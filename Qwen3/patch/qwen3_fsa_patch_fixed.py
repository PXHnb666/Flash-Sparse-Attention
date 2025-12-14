# qwen3_fsa_patch_fixed.py
# Minimal monkey patch to use Flash-Sparse-Attention (FSA) inside Qwen3Attention.forward for PREFILL.
# Supported path: batch prefill (no KV cache). For decoding (one-step) or paged-KV cache, falls back to original.
#
# Usage:
#   from qwen3_fsa_patch import enable_fsa_for_qwen3, disable_fsa_for_qwen3
#   model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16).cuda()
#   enable_fsa_for_qwen3(model, block_size=64, topk=16)
#
# Requirements:
#   pip install triton>=3.0 flash-attn  # built for your torch/cu toolchain
#   pip install git+https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention.git
#
# Notes:
#   - This patch handles only the "prefill" path (no past_key_value) and assumes left-padded False (contiguous sequences).
#   - If inputs contain padding, we compute per-sample lengths from attention_mask to build cu_seqlens.
#   - For decoding (past_key_value not None), we fall back to the original forward to ensure correctness.
#   - Head dim must be <= 256 per FSA kernel constraint.
#   - Test on NVIDIA A100/H100 (bf16/fp16).

from typing import Optional, Tuple, Callable, Dict
import types
import inspect
import math

import torch
import torch.nn as nn
import time

try:
    # FSA import paths (adjust if project layout changes)
    from fsa.module.fsa import FlashSparseAttention, RopeConfig
except Exception as e:
    raise ImportError(
        "Failed to import Flash-Sparse-Attention. "
        "Please install it: pip install git+https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention.git"
    ) from e

# Import Qwen3 pieces for type hints and fallback paths.
try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3DecoderLayer,
        Qwen3ForCausalLM,
        apply_rotary_pos_emb,
        repeat_kv,
    )
except ImportError:
    from transformers.models.qwen.modeling_qwen import (
        QwenAttention as Qwen3Attention,
        QwenDecoderLayer as Qwen3DecoderLayer,
        QwenForCausalLM as Qwen3ForCausalLM,
        apply_rotary_pos_emb,
        repeat_kv,
    )


def _is_left_padded(mask_row: torch.Tensor) -> bool:
    return bool(mask_row.numel() > 0 and mask_row[0] == 0 and mask_row[-1] == 1)


def _ensure_fsa_dtype(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.bfloat16, torch.float16):
        return t
    target = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return t.to(dtype=target)


def _pack_sequences(x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert (B, T, H*D) or (B, T, H, D) into packed tokens + cu_seqlens for FSA.
    Returns: (x_packed, cu_seqlens, lengths)

    - Supports right-padding and left-padding.
    - attn_mask can be bool/long/float; will be interpreted as 1(valid)/0(pad).
    """
    B, T = x.shape[0], x.shape[1]

    if attn_mask is None:
        lengths = torch.full((B,), T, dtype=torch.int32, device=x.device)
    else:
        if attn_mask.dtype not in (torch.bool, torch.long, torch.int32, torch.int64):
            # float mask -> 1/0
            attn_mask = (attn_mask != 0).to(torch.long)
        elif attn_mask.dtype is torch.bool:
            attn_mask = attn_mask.to(torch.long)

        lengths = attn_mask.to(dtype=torch.int32).sum(dim=1).to(torch.int32)

    # cu_seqlens: [0, len0, len0+len1, ...]
    cu = torch.zeros(B + 1, dtype=torch.int32, device=x.device)
    cu[1:] = torch.cumsum(lengths, dim=0)

    # Pack by slicing each sequence prefix/suffix of length[i]
    if attn_mask is None:
        x_packed = x.reshape(B * T, *x.shape[2:]).contiguous()
    else:
        chunks = []
        for i in range(B):
            Li = int(lengths[i].item())
            if Li == 0:
                continue
            mask_i = attn_mask[i]
            left_pad = _is_left_padded(mask_i)
            if left_pad:
                start = T - Li
                sl = x[i, start:T]
            else:
                sl = x[i, :Li]
            chunks.append(sl)
        x_packed = torch.cat(chunks, dim=0).contiguous() if chunks else x.new_zeros((0, *x.shape[2:]))

    return x_packed, cu, lengths


def _build_fsa(cfg, device, dtype, block_size: int, topk: int) -> FlashSparseAttention:
    rope_cfg = RopeConfig(
        max_position_embeddings=getattr(cfg, "max_position_embeddings", 131072),
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        rope_theta=getattr(cfg, "rope_theta", 10000.0),
        rope_scaling=getattr(cfg, "rope_scaling", {}) or {},
    )

    base_kwargs = dict(
        hidden_size=cfg.hidden_size,
        num_q_heads=cfg.num_attention_heads,
        num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        block_size=block_size,
        topk=topk,
        rope_config=rope_cfg,
    )

    # Adapt to newer FSA signatures (kernel_size, kernel_stride, init_blocks, local_blocks, window_size, ...)
    try:
        sig_params = inspect.signature(FlashSparseAttention.__init__).parameters
    except (ValueError, TypeError):
        sig_params = {}
    extra = {}
    if "kernel_size" in sig_params:
        extra.update(dict(
            kernel_size=block_size,
            kernel_stride=max(1, block_size // 2),
            init_blocks=1,
            local_blocks=1,
            window_size=block_size * 2,
        ))

    fsa = FlashSparseAttention(**base_kwargs, **extra).to(device=device, dtype=dtype)
    return fsa


def _build_rope_config_from_qwen(cfg) -> RopeConfig:
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    if rope_scaling is None:
        rope_scaling = {"rope_type": "default"}
    else:
        rope_scaling = dict(rope_scaling)
        rope_scaling.setdefault("rope_type", rope_scaling.get("type", "default"))

    return RopeConfig(
        max_position_embeddings=getattr(cfg, "max_position_embeddings", 131072),
        head_dim=head_dim,
        rope_theta=getattr(cfg, "rope_theta", 10000.0),
        rope_scaling=rope_scaling,
    )


class Qwen3FSAAttention(Qwen3Attention):
    """
    Drop-in replacement for Qwen3Attention that routes the PREFILL path through Flash-Sparse-Attention (FSA).
    Decoding / cache usage falls back to the standard attention implementation.
    """

    def __init__(self, config, layer_idx: int, fsa_kwargs: Optional[Dict[str, int]] = None):
        super().__init__(config, layer_idx)
        self._fallback_forward = super().forward

        assert (
            self.num_key_value_groups * config.num_key_value_heads == config.num_attention_heads
        ), "Invalid key-value head grouping configuration."

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        defaults = {
            "kernel_size": 32,
            "kernel_stride": 16,
            "block_size": 64,
            "topk": 16,
            "init_blocks": 1,
            "local_blocks": 2,
            "window_size": 512,
        }
        if fsa_kwargs:
            defaults.update({k: v for k, v in fsa_kwargs.items() if v is not None})

        # FSA kernels restrict kernel_size and block_size; normalize to supported values.
        supported_kernel_sizes = (16, 32, 64, 128)
        requested_kernel_size = defaults["kernel_size"]
        if requested_kernel_size not in supported_kernel_sizes:
            new_kernel_size = min(supported_kernel_sizes, key=lambda k: (abs(k - requested_kernel_size), k))
            requested_stride = defaults.get("kernel_stride", new_kernel_size // 2)
            requested_stride = min(max(1, requested_stride), new_kernel_size)
            defaults["kernel_size"] = new_kernel_size
            defaults["kernel_stride"] = math.gcd(new_kernel_size, requested_stride) or 1
            print(
                f"[FSA] kernel_size {requested_kernel_size} not supported; "
                f"using {new_kernel_size} with stride {defaults['kernel_stride']}."
            )
        else:
            # Ensure stride divides kernel_size.
            stride = defaults.get("kernel_stride", defaults["kernel_size"] // 2)
            stride = min(max(1, stride), defaults["kernel_size"])
            defaults["kernel_stride"] = math.gcd(defaults["kernel_size"], stride) or 1

        supported_block_sizes = (32, 64, 128, 256)
        requested_block_size = defaults["block_size"]
        if requested_block_size not in supported_block_sizes:
            new_block_size = min(supported_block_sizes, key=lambda b: (abs(b - requested_block_size), b))
            defaults["block_size"] = new_block_size
            print(
                f"[FSA] block_size {requested_block_size} not supported; using {new_block_size}."
            )

        self.fsa = FlashSparseAttention(
            hidden_size=config.hidden_size,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            head_dim=head_dim,
            kernel_size=defaults["kernel_size"],
            kernel_stride=defaults["kernel_stride"],
            block_size=defaults["block_size"],
            topk=defaults["topk"],
            init_blocks=defaults["init_blocks"],
            local_blocks=defaults["local_blocks"],
            window_size=defaults["window_size"],
            rope_config=_build_rope_config_from_qwen(config),
        )

        # Share qkv / output projection weights with the base attention.
        if hasattr(self.fsa, "proj_q"):
            self.fsa.proj_q.weight = self.q_proj.weight
            if self.q_proj.bias is not None and hasattr(self.fsa.proj_q, "bias"):
                self.fsa.proj_q.bias = self.q_proj.bias
        if hasattr(self.fsa, "proj_k"):
            self.fsa.proj_k.weight = self.k_proj.weight
            if self.k_proj.bias is not None and hasattr(self.fsa.proj_k, "bias"):
                self.fsa.proj_k.bias = self.k_proj.bias
        if hasattr(self.fsa, "proj_v"):
            self.fsa.proj_v.weight = self.v_proj.weight
            if self.v_proj.bias is not None and hasattr(self.fsa.proj_v, "bias"):
                self.fsa.proj_v.bias = self.v_proj.bias
        if hasattr(self.fsa, "proj_o"):
            self.fsa.proj_o.weight = self.o_proj.weight
            if self.o_proj.bias is not None and hasattr(self.fsa.proj_o, "bias"):
                self.fsa.proj_o.bias = self.o_proj.bias

        # Keep auxiliary FSA parameters aligned with attention dtype/device.
        base_param = next(self.parameters())
        self.fsa.to(device=base_param.device, dtype=base_param.dtype)
        self._fsa_forward_params = list(inspect.signature(self.fsa.forward).parameters)
        self.bench_sync = False  # set True externally for micro-bench syncing
        self.validate_fsa = False  # set True to compare with fallback attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """FSA-accelerated *prefill* attention.

        Important: HuggingFace Qwen3 creates a DynamicCache when `use_cache=True`, so `past_key_values`
        is **not None** even during the very first (prefill) forward.

        We treat a layer as "decode" only if that layer already has cached KV (seq_len > 0).
        """

        def _get_layer_cache_len(pkv, layer_idx: int):
            """Best-effort: return cached KV length for *this* layer.

            Returns:
              - int >= 0 if we can infer
              - None if we cannot infer
            """
            if pkv is None:
                return 0

            # 1) Try per-layer `get_seq_length(layer_idx)` (newer Cache APIs)
            if hasattr(pkv, "get_seq_length"):
                try:
                    return int(pkv.get_seq_length(layer_idx))
                except TypeError:
                    pass
                except Exception:
                    pass

            # 2) Try common DynamicCache internals
            for attr in ("key_cache", "keys", "k_cache"):
                if hasattr(pkv, attr):
                    kc = getattr(pkv, attr)
                    try:
                        # If caches are appended lazily, missing entries imply empty.
                        if isinstance(kc, (list, tuple)) and len(kc) <= layer_idx:
                            return 0
                        k = kc[layer_idx]
                        if k is None:
                            return 0
                        if hasattr(k, "shape") and len(getattr(k, "shape")) >= 2:
                            return int(k.shape[-2])
                    except Exception:
                        pass

            # 3) Try __getitem__
            try:
                item = pkv[layer_idx]
                if isinstance(item, (tuple, list)) and len(item) >= 1:
                    k = item[0]
                    if k is None:
                        return 0
                    return int(k.shape[-2])
            except Exception:
                pass

            # 4) Legacy tuple-of-tuples cache
            if isinstance(pkv, (tuple, list)) and len(pkv) > layer_idx:
                try:
                    k = pkv[layer_idx][0]
                    return int(k.shape[-2])
                except Exception:
                    pass

            # 5) As a last resort, try global `get_seq_length()` (may be layer-0 only)
            if hasattr(pkv, "get_seq_length"):
                try:
                    return int(pkv.get_seq_length())
                except Exception:
                    pass

            return None

        layer_idx = int(getattr(self, "layer_idx", 0))
        layer_past_len = _get_layer_cache_len(past_key_values, layer_idx)

        # Decode path (this layer already has cache) -> defer to original attention.
        if layer_past_len not in (None, 0):
            if layer_idx == 0 and not getattr(self, "_debug_decode_printed", False):
                print(f"[FSA] Decode detected (layer0 past_len={layer_past_len}), fallback to baseline attention.")
                self._debug_decode_printed = True
            return self._fallback_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )


        # If user explicitly disables cache in inference, keep the old conservative fallback.
        if (not self.training) and (kwargs.get("use_cache", None) is False):
            return self._fallback_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # Accept 4D causal masks by treating them as fully valid (no padding).
        attn_mask_for_pack = attention_mask
        if attn_mask_for_pack is not None and attn_mask_for_pack.dim() == 4:
            attn_mask_for_pack = None

        # Debug print once to confirm FSA path is taken.
        if layer_idx == 0 and not getattr(self, "_debug_printed", False):
            print("[FSA] Using FSA forward for layer 0 (prefill path).")
            self._debug_printed = True

        # Prefill path: pack hidden_states -> FSA(x, cu_seqlens) -> unpack.
        mask_2d = attn_mask_for_pack if (attn_mask_for_pack is not None and attn_mask_for_pack.dim() == 2) else None

        B, T, C = hidden_states.shape
        x = _ensure_fsa_dtype(hidden_states)

        if mask_2d is not None:
            if mask_2d.dtype not in (torch.bool, torch.long, torch.int32, torch.int64):
                mask_2d = (mask_2d != 0).to(torch.long)
            elif mask_2d.dtype is torch.bool:
                mask_2d = mask_2d.to(torch.long)

        if mask_2d is None:
            lengths = torch.full((B,), T, dtype=torch.int32, device=x.device)
            cu = torch.arange(0, (B + 1) * T, step=T, device=x.device, dtype=torch.int32)
            x_packed = x.reshape(B * T, C)
            cu_for_unpack, lengths_for_unpack = cu, lengths
        else:
            x_packed, cu, lengths = _pack_sequences(x, mask_2d)
            cu_for_unpack, lengths_for_unpack = cu, lengths

        if self.bench_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            # Your installed FlashSparseAttention expects (x, cu_seqlens).
            attn_out_packed = self.fsa(x_packed, cu_for_unpack)
        except TypeError:
            # Fallback for possible older signatures.
            attn_out_packed = self.fsa(x_packed)
        if self.bench_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        assert attn_out_packed.size(0) == int(cu_for_unpack[-1].item()), "Packed output size mismatch"

        if mask_2d is None:
            attn_out = attn_out_packed.view(B, T, C)
        else:
            attn_out = x.new_zeros((B, T, C))
            start = 0
            for i in range(B):
                Li = int(lengths_for_unpack[i])
                if Li <= 0:
                    continue
                mask_i = mask_2d[i]
                left_pad = _is_left_padded(mask_i)
                first_idx = mask_i.size(0) - Li if left_pad else 0
                attn_out[i, first_idx:first_idx + Li] = attn_out_packed[start:start + Li]
                start += Li

        # If a cache object is provided (e.g., DynamicCache), populate it so that subsequent decode works.
        # Baseline Qwen3Attention updates KV whenever `past_key_values is not None`.
        if past_key_values is not None:
            if position_embeddings is None:
                # Can't reliably update cache without cos/sin; fall back for correctness.
                return self._fallback_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            cos, sin = position_embeddings
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            # Recompute KV in the same way as baseline attention (k_norm + RoPE; v no RoPE)
            key_states = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(x).view(hidden_shape).transpose(1, 2)
            _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            _ = past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)

        # NOTE: we do not return attn_weights (keep None) to match the original patch's behavior.
        return attn_out, None


def patch_qwen3_with_fsa(
    model: "Qwen3ForCausalLM",
    *,
    kernel_size: int = 32,
    kernel_stride: int = 16,
    block_size: int = 64,
    topk: int = 16,
    init_blocks: int = 1,
    local_blocks: int = 2,
    window_size: int = 512,
    verbose: bool = True,
):
    """
    Replace each decoder layer's self_attn with Qwen3FSAAttention (prefill via FSA, decode falls back).
    """
    if not isinstance(model, Qwen3ForCausalLM):
        raise TypeError("patch_qwen3_with_fsa expects a Qwen3ForCausalLM model.")

    layers = getattr(model, "model", None)
    if layers is None or not hasattr(layers, "layers"):
        raise AttributeError("Could not locate decoder layers on the provided model.")

    fsa_kwargs = dict(
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        window_size=window_size,
    )

    num_patched = 0
    log_dtype, log_device = None, None
    for idx, layer in enumerate(layers.layers):
        if not isinstance(layer, Qwen3DecoderLayer):
            continue
        if isinstance(layer.self_attn, Qwen3FSAAttention):
            continue
        if getattr(layer.self_attn, "_fsa_enabled", False):
            continue

        old_attn = layer.self_attn
        new_attn = Qwen3FSAAttention(model.config, layer_idx=idx, fsa_kwargs=fsa_kwargs)

        # Match device/dtype with the existing attention.
        ref_param = next(old_attn.parameters())
        new_attn.to(device=ref_param.device, dtype=ref_param.dtype)

        # Copy projection and norm weights to preserve checkpoint compatibility.
        new_attn.q_proj.load_state_dict(old_attn.q_proj.state_dict())
        new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
        new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
        new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
        new_attn.q_norm.load_state_dict(old_attn.q_norm.state_dict())
        new_attn.k_norm.load_state_dict(old_attn.k_norm.state_dict())
        new_attn.q_norm.to(dtype=ref_param.dtype, device=ref_param.device)
        new_attn.k_norm.to(dtype=ref_param.dtype, device=ref_param.device)

        layer.self_attn = new_attn
        layer.self_attn._fsa_enabled = True
        num_patched += 1
        log_dtype, log_device = ref_param.dtype, ref_param.device

    if verbose:
        suffix = ""
        if log_dtype is not None:
            suffix = f" ({log_dtype}, device={log_device})"
        print(f"[FSA] Patched {num_patched} Qwen3 decoder layers with Flash-Sparse-Attention{suffix}.")

    return model


def enable_fsa_for_qwen3(model: nn.Module, block_size: int = 64, topk: int = 16, verbose: bool = True):
    """
    Deprecated: the monkey-patch path is not compatible with current Qwen3 signatures.
    Please use `patch_qwen3_with_fsa` instead.
    """
    raise RuntimeError("enable_fsa_for_qwen3 is deprecated for current Qwen3; use patch_qwen3_with_fsa(model).")


def disable_fsa_for_qwen3(model: nn.Module, verbose: bool = True):
    restored = 0
    for m in model.modules():
        if m.__class__.__name__ == "Qwen3Attention" and hasattr(m, "_orig_forward"):
            m.forward = m._orig_forward
            delattr(m, "_orig_forward")
            if hasattr(m, "_fsa_enabled"):
                delattr(m, "_fsa_enabled")
            restored += 1
    if verbose:
        print(f"[FSA] Restored {restored} Qwen3Attention modules to original forward.")
