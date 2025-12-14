#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-8B FSA Patch Analysis (v3)

Goal
- Load local Qwen3-8B weights (ModelScope cache path works)
- Apply qwen3_fsa_patch_fixed.patch_qwen3_with_fsa
- Count *unique* linear matrices in decoder blocks + lm_head
- Distinguish:
  (A) newly introduced Linear modules by patch
  (B) newly introduced *unique* linear weight tensors (true new matrices)
- Identify which matmuls are replaced by FSA (attention core only)
- Provide a FLOPs delta table for prefill (batch=1) vs dense causal attention

Key fix vs v2
- Uses STORAGE identity (data_ptr + offset + shape + stride + dtype) instead of id(Parameter)
  to detect weight sharing across modules.

This script is FLOPs accounting (matmul-centric), not a runtime benchmark.
"""

from __future__ import annotations

import os
import sys
import json
import math
from typing import Dict, List, Tuple, Iterable, Any, Optional

import torch
import torch.nn as nn

# -------------------------
# USER EDITABLE DEFAULTS
# -------------------------
MODEL_PATH = "/home/peixiaohan/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
PATCH_DIR  = "/home/peixiaohan/LLM/Flash-Sparse-Attention"

FSA_CFG = dict(
    block_size=64,
    topk=16,
    kernel_size=32,
    kernel_stride=16,
    init_blocks=1,
    local_blocks=2,
    window_size=512,
)

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
BATCH = 1
DTYPE = torch.bfloat16


# -------------------------
# helpers
# -------------------------

def pick_device() -> torch.device:
    # NOTE: respects CUDA_VISIBLE_DEVICES; if the env is wrong, torch.cuda.is_available() is False.
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_patch_import_path():
    if PATCH_DIR and PATCH_DIR not in sys.path:
        sys.path.insert(0, PATCH_DIR)


def iter_named_linears(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            yield name, mod


def _storage_data_ptr(t: torch.Tensor) -> int:
    # untyped_storage is preferred in newer PyTorch.
    try:
        return t.untyped_storage().data_ptr()
    except Exception:
        return t.storage().data_ptr()  # type: ignore[attr-defined]


def weight_storage_key(w: torch.Tensor) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...], str]:
    """A stable key that treats re-wrapped Parameters sharing the same storage as identical."""
    return (
        _storage_data_ptr(w),
        int(w.storage_offset()),
        tuple(int(x) for x in w.shape),
        tuple(int(x) for x in w.stride()),
        str(w.dtype),
    )


def unique_linear_weight_tensors(
    model: nn.Module,
    name_prefixes: Tuple[str, ...],
) -> Dict[Tuple[int, int, Tuple[int, ...], Tuple[int, ...], str], Dict[str, Any]]:
    """Returns mapping: storage_key -> {names, shape, dtype, numel}."""
    out: Dict[Tuple[int, int, Tuple[int, ...], Tuple[int, ...], str], Dict[str, Any]] = {}
    for name, lin in iter_named_linears(model):
        if not name.startswith(name_prefixes):
            continue
        w = lin.weight
        k = weight_storage_key(w)
        if k not in out:
            out[k] = {
                "names": [name],
                "shape": list(w.shape),
                "dtype": str(w.dtype),
                "numel": int(w.numel()),
            }
        else:
            out[k]["names"].append(name)
    return out


def collect_linear_modules(
    model: nn.Module,
    name_prefixes: Tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    """Returns mapping: module_name -> {weight_key, shape, dtype, numel}."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, lin in iter_named_linears(model):
        if not name.startswith(name_prefixes):
            continue
        w = lin.weight
        out[name] = {
            "weight_key": weight_storage_key(w),
            "shape": list(w.shape),
            "dtype": str(w.dtype),
            "numel": int(w.numel()),
        }
    return out


def summarize_unique(uw: Dict[Any, Dict[str, Any]]) -> Tuple[int, int]:
    return len(uw), sum(v["numel"] for v in uw.values())


# -------------------------
# FLOPs model (prefill, causal)
# -------------------------

def flops_linear(batch: int, seqlen: int, in_f: int, out_f: int) -> float:
    return 2.0 * batch * seqlen * in_f * out_f


def avg_keys_dense_causal(S: int) -> float:
    # token t attends to (t+1) keys
    return (S + 1) / 2.0


def avg_keys_window_causal(S: int, w: int) -> float:
    w = min(w, S)
    # average of min(t+1,w)
    # closed form: if S<=w -> (S+1)/2; else -> [w(w+1)/2 + (S-w)w]/S
    if S <= w:
        return (S + 1) / 2.0
    return (w * (w + 1) / 2.0 + (S - w) * w) / S


def compressed_len(S: int, kernel: int, stride: int) -> int:
    if S < kernel:
        return 0
    return (S - kernel) // stride + 1


def avg_keys_compressed_causal(S: int, kernel: int, stride: int) -> float:
    # windows end at start+kernel-1 where start=i*stride.
    # available compressed keys at token t: count of i s.t. i*stride + (kernel-1) <= t.
    # count = floor((t-(kernel-1))/stride)+1 for t>=kernel-1, else 0.
    if S <= 0:
        return 0.0
    total = 0
    for t in range(S):
        if t < kernel - 1:
            continue
        total += (t - (kernel - 1)) // stride + 1
    return total / S


def avg_keys_sparse_causal(S: int, keys_cap: int) -> float:
    keys_cap = min(keys_cap, S)
    total = 0
    for t in range(S):
        total += min(t + 1, keys_cap)
    return total / S


def flops_attn_core(batch: int, S: int, n_q_heads: int, head_dim: int, avg_keys: float) -> float:
    # QK^T + AV (matmul only), causal accounted in avg_keys
    return 4.0 * batch * n_q_heads * S * avg_keys * head_dim


def flops_linear_compress_kv(batch: int, S: int, kv_heads: int, head_dim: int, kernel: int, stride: int) -> float:
    # linear_compress consumes K/V already in per-head space: [S, kv_heads, head_dim]
    # Each compressed output position computes (kernel*head_dim) x (head_dim) per kv head.
    # FLOPs per output per head: 2*(kernel*D)*D
    Sc = compressed_len(S, kernel, stride)
    return 4.0 * batch * kv_heads * Sc * kernel * head_dim * head_dim  # x2 for K and V


def flops_gate(batch: int, S: int, hidden: int) -> float:
    return 2.0 * batch * S * hidden * 3


def build_flops_table(model_cfg: Dict[str, int], fsa_cfg: Dict[str, int], seq_lens: List[int], batch: int) -> List[Dict[str, Any]]:
    H = model_cfg["hidden_size"]
    nH = model_cfg["num_attention_heads"]
    kvH = model_cfg["num_key_value_heads"]
    L = model_cfg["num_hidden_layers"]
    head_dim = H // nH
    interm = model_cfg["intermediate_size"]
    vocab = model_cfg["vocab_size"]

    blk = fsa_cfg["block_size"]
    topk = fsa_cfg["topk"]
    window = fsa_cfg["window_size"]
    kernel = fsa_cfg["kernel_size"]
    stride = fsa_cfg["kernel_stride"]

    rows = []
    for S in seq_lens:
        # proj per layer (same in dense and patched)
        q_proj = flops_linear(batch, S, H, H)
        k_proj = flops_linear(batch, S, H, kvH * head_dim)
        v_proj = flops_linear(batch, S, H, kvH * head_dim)
        o_proj = flops_linear(batch, S, H, H)
        attn_proj = (q_proj + k_proj + v_proj + o_proj) * L

        mlp = (flops_linear(batch, S, H, interm) + flops_linear(batch, S, interm, H)) * L
        lm = flops_linear(batch, S, H, vocab)

        dense_core = flops_attn_core(batch, S, nH, head_dim, avg_keys_dense_causal(S)) * L
        dense_total = attn_proj + mlp + lm + dense_core

        # FSA replacement cores (this repo computes ALL three branches then gates)
        Sc = compressed_len(S, kernel, stride)
        avg_c = avg_keys_compressed_causal(S, kernel, stride)
        comp_kv = flops_linear_compress_kv(batch, S, kvH, head_dim, kernel, stride) * L
        comp_core = flops_attn_core(batch, S, nH, head_dim, avg_c) * L

        # topk_sparse_attention attends to topk blocks; init/local are forced-included inside the same topk.
        keys_sparse = topk * blk
        avg_s = avg_keys_sparse_causal(S, keys_sparse)
        sparse_core = flops_attn_core(batch, S, nH, head_dim, avg_s) * L

        avg_w = avg_keys_window_causal(S, window)
        slide_core = flops_attn_core(batch, S, nH, head_dim, avg_w) * L

        gate = flops_gate(batch, S, H) * L

        fsa_core = comp_kv + comp_core + sparse_core + slide_core + gate
        fsa_total = attn_proj + mlp + lm + fsa_core

        rows.append({
            "seq_len": S,
            "dense_total": dense_total,
            "dense_core": dense_core,
            "fsa_total": fsa_total,
            "fsa_core": fsa_core,
            "delta_total": fsa_total - dense_total,
            "delta_total_pct": (fsa_total / dense_total - 1.0) * 100.0,
            "delta_core": fsa_core - dense_core,
            "delta_core_pct": (fsa_core / dense_core - 1.0) * 100.0,
            "breakdown": {
                "attn_proj": attn_proj,
                "mlp": mlp,
                "lm_head": lm,
                "dense_core": dense_core,
                "fsa_gate": gate,
                "fsa_linear_compress_kv": comp_kv,
                "fsa_compressed_core": comp_core,
                "fsa_sparse_core": sparse_core,
                "fsa_sliding_core": slide_core,
                "compressed_len": Sc,
                "avg_keys_dense": avg_keys_dense_causal(S),
                "avg_keys_compressed": avg_c,
                "avg_keys_sparse": avg_s,
                "avg_keys_window": avg_w,
            }
        })
    return rows


# -------------------------
# main
# -------------------------

def main():
    set_patch_import_path()

    from transformers import AutoConfig, AutoModelForCausalLM
    from qwen3_fsa_patch_fixed import patch_qwen3_with_fsa

    device = pick_device()

    print("=" * 96)
    print("Qwen3-8B FSA Patch Analysis v3")
    print("=" * 96)
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"PATCH_DIR : {PATCH_DIR}")
    print(f"DEVICE    : {device}")
    print(f"DTYPE     : {DTYPE}")
    print(f"FSA_CFG   : {FSA_CFG}")
    print(f"SEQ_LENS  : {SEQ_LENS} (batch={BATCH})")

    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    prefixes = ("model.layers.", "lm_head")

    # baseline
    base_unique = unique_linear_weight_tensors(model, prefixes)
    base_mods = collect_linear_modules(model, prefixes)
    base_cnt, base_numel = summarize_unique(base_unique)

    # patch
    patch_qwen3_with_fsa(model, verbose=True, **FSA_CFG)

    pat_unique = unique_linear_weight_tensors(model, prefixes)
    pat_mods = collect_linear_modules(model, prefixes)
    pat_cnt, pat_numel = summarize_unique(pat_unique)

    # delta: unique matrices
    base_keys = set(base_unique.keys())
    pat_keys = set(pat_unique.keys())
    new_weight_keys = sorted(list(pat_keys - base_keys), key=lambda x: (x[2], x[0]))

    # delta: module names
    base_names = set(base_mods.keys())
    pat_names = set(pat_mods.keys())
    new_module_names = sorted(list(pat_names - base_names))

    shared_new_modules = []
    real_new_modules = []
    for n in new_module_names:
        wk = pat_mods[n]["weight_key"]
        if wk in base_keys:
            shared_new_modules.append(n)
        else:
            real_new_modules.append(n)

    print("\n" + "=" * 96)
    print("SUMMARY (Linear weights, STORAGE-unique)")
    print("=" * 96)
    print(f"[UNIQUE Linear weight tensors] baseline: {base_cnt}  numel={base_numel}")
    print(f"[UNIQUE Linear weight tensors] patched : {pat_cnt}  numel={pat_numel}")
    print(f"[UNIQUE Linear weight tensors] delta   : {pat_cnt - base_cnt}  numel={pat_numel - base_numel}")

    print("\n[New Linear modules introduced by patch]")
    print(f"  total new modules: {len(new_module_names)}")
    print(f"  - share existing baseline weights: {len(shared_new_modules)} (expected: proj_q/k/v/o wrappers)")
    print(f"  - introduce NEW weight tensors:    {len(real_new_modules)} (expected: gate)\n")

    if real_new_modules:
        print("  NEW-WEIGHT Linear modules (first 20):")
        for n in real_new_modules[:20]:
            meta = pat_mods[n]
            print(f"    - {n}: shape={meta['shape']} numel={meta['numel']} dtype={meta['dtype']}")
        if len(real_new_modules) > 20:
            print(f"    ... ({len(real_new_modules) - 20} more)")

    if new_weight_keys:
        print("\n[New UNIQUE Linear weight tensors (true new matrices)]")
        for k in new_weight_keys[:20]:
            v = pat_unique[k]
            print(f"  - shape={v['shape']} numel={v['numel']} dtype={v['dtype']} names={v['names'][:3]}{'...' if len(v['names'])>3 else ''}")
        if len(new_weight_keys) > 20:
            print(f"  ... ({len(new_weight_keys) - 20} more)")

    # Non-linear FSA params related to matmul-ish ops
    extra = []
    total_extra = 0
    for n, p in model.named_parameters():
        if ".self_attn.fsa." not in n:
            continue
        if any(k in n for k in ("compress_key", "compress_value", "intra_block_pe")):
            extra.append({"name": n, "shape": list(p.shape), "numel": int(p.numel()), "dtype": str(p.dtype)})
            total_extra += int(p.numel())

    print("\n" + "=" * 96)
    print("FSA extra params (non-Linear but matmul-related)")
    print("=" * 96)
    print(f"count={len(extra)} total_numel={total_extra} (~{total_extra * 2 / 1024 / 1024:.2f} MiB in bf16)")
    print("top-5 largest:")
    for it in sorted(extra, key=lambda x: -x["numel"])[:5]:
        print(f"  - {it['name']}: shape={it['shape']} numel={it['numel']} dtype={it['dtype']}")

    # FLOPs
    model_cfg = dict(
        hidden_size=int(cfg.hidden_size),
        num_attention_heads=int(getattr(cfg, "num_attention_heads")),
        num_key_value_heads=int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads"))),
        num_hidden_layers=int(getattr(cfg, "num_hidden_layers")),
        intermediate_size=int(getattr(cfg, "intermediate_size")),
        vocab_size=int(getattr(cfg, "vocab_size")),
    )

    rows = build_flops_table(model_cfg, FSA_CFG, SEQ_LENS, BATCH)

    def to_T(x: float) -> float:
        return x / 1e12

    print("\n" + "=" * 96)
    print("FLOPs (matmul-centric, PREFILL, causal) delta table")
    print("=" * 96)
    for r in rows:
        S = r["seq_len"]
        print(
            f"S={S:5d} | dense_total={to_T(r['dense_total']):8.3f}T  fsa_total={to_T(r['fsa_total']):8.3f}T  "
            f"Δtotal={to_T(r['delta_total']):+8.3f}T ({r['delta_total_pct']:+6.2f}%) | "
            f"Δcore={to_T(r['delta_core']):+8.3f}T ({r['delta_core_pct']:+6.2f}%)"
        )

    out_path = "qwen3_fsa_analysis_v3.json"
    report = {
        "model_path": MODEL_PATH,
        "patch_dir": PATCH_DIR,
        "device": str(device),
        "dtype": str(DTYPE),
        "model_config": model_cfg,
        "fsa_config": FSA_CFG,
        "linear_weight_analysis": {
            "baseline": {"unique": base_cnt, "numel": base_numel},
            "patched": {"unique": pat_cnt, "numel": pat_numel},
            "delta": {"unique": pat_cnt - base_cnt, "numel": pat_numel - base_numel},
            "new_linear_modules": new_module_names,
            "new_linear_modules_shared_weights": shared_new_modules,
            "new_linear_modules_new_weights": real_new_modules,
            "new_unique_linear_weight_tensors": [pat_unique[k] for k in new_weight_keys],
            "fsa_extra_params_non_linear": {"total_numel": total_extra, "items": extra},
        },
        "flops": [
            {
                "seq_len": r["seq_len"],
                "dense_total_T": to_T(r["dense_total"]),
                "dense_core_T": to_T(r["dense_core"]),
                "fsa_total_T": to_T(r["fsa_total"]),
                "fsa_core_T": to_T(r["fsa_core"]),
                "delta_total_T": to_T(r["delta_total"]),
                "delta_total_pct": r["delta_total_pct"],
                "delta_core_T": to_T(r["delta_core"]),
                "delta_core_pct": r["delta_core_pct"],
                "breakdown_T": {k: (to_T(v) if isinstance(v, (int, float)) else v) for k, v in r["breakdown"].items()},
            }
            for r in rows
        ],
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved JSON report: {out_path}")


if __name__ == "__main__":
    main()
