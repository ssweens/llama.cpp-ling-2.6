#!/usr/bin/env python3
"""CPU-only layer/component probe for Ling-2.6-flash / BailingMoeV2.5.

This intentionally avoids loading the full 104B model. It loads only the tensors
needed for a small set of high-signal parity checks:

* selected HF -> GGUF tensor conversion checks;
* Lightning/simple-GLA slope metadata check;
* MoE router top-k/group-limited routing check;
* layer-0 linear-attention attention-block comparison between HF-dequantized
  weights and the converted GGUF Q8_0 weights.

The probe is diagnostic rather than a full correctness proof. It is designed to
run CPU-only on the existing FP8 HF checkpoint and Q8_0 GGUF.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gguf-py"))

from gguf import GGUFReader, dequantize  # noqa: E402


@dataclass(frozen=True)
class Paths:
    hf: Path
    gguf: Path


class TensorStore:
    def __init__(self, paths: Paths):
        self.paths = paths
        self.config = json.loads((paths.hf / "config.json").read_text())
        self.index = json.loads((paths.hf / "model.safetensors.index.json").read_text())["weight_map"]
        self.gguf_reader = GGUFReader(paths.gguf)
        self.gguf_tensors = {t.name: t for t in self.gguf_reader.tensors}
        self._hf_cache: dict[str, torch.Tensor] = {}
        self._gguf_cache: dict[str, torch.Tensor] = {}

    def hf_raw(self, name: str) -> torch.Tensor:
        if name in self._hf_cache:
            return self._hf_cache[name]
        fn = self.index[name]
        with safe_open(self.paths.hf / fn, framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        self._hf_cache[name] = t
        return t

    def hf(self, name: str) -> torch.Tensor:
        """Load an HF tensor, dequantizing FP8 sidecar scales when present."""
        t = self.hf_raw(name)
        scale_name = name + "_scale_inv"
        if scale_name in self.index:
            scale = self.hf_raw(scale_name).float()
            block_size = self.config.get("quantization_config", {}).get("weight_block_size")
            t = t.float()
            if block_size is not None:
                # convert_hf_to_gguf.py dequant_simple(): repeat scales by block,
                # then trim to the weight shape.
                dim_offset = scale.ndim - len(block_size)
                for i, size in enumerate(block_size):
                    scale = scale.repeat_interleave(size, dim_offset + i)
                slices = tuple(slice(0, size) for size in t.shape)
                scale = scale[slices]
            while scale.ndim < t.ndim:
                scale = scale.unsqueeze(-1)
            return (t * scale).float()
        return t.float()

    def gguf(self, name: str) -> torch.Tensor:
        if name in self._gguf_cache:
            return self._gguf_cache[name]
        t = self.gguf_tensors[name]
        arr = dequantize(t.data, t.tensor_type).copy()
        out = torch.from_numpy(arr).float()
        self._gguf_cache[name] = out
        return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)) * weight.float()


def group_rms_norm(x: torch.Tensor, weight: torch.Tensor, n_groups: int, eps: float) -> torch.Tensor:
    shape = x.shape
    xg = x.float().view(*shape[:-1], n_groups, shape[-1] // n_groups)
    xg = xg * torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xg.reshape(shape) * weight.float()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_split_half(x: torch.Tensor, positions: torch.Tensor, rope_dim: int, base: float) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    freqs = torch.outer(positions.float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, None, :]
    sin = emb.sin()[:, None, :]
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat((x_rot, x_pass), dim=-1)


def rope_interleave_mla(x: torch.Tensor, positions: torch.Tensor, rope_dim: int, base: float) -> torch.Tensor:
    """HF BailingMoeV2.5 MLA interleaved RoPE for x [T,H,D]."""
    inv = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    freqs = torch.outer(positions.float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, None, :]
    sin = emb.sin()[:, None, :]
    xt = x.permute(1, 0, 2).unsqueeze(0)  # [1,H,T,D]
    b, h, s, d = xt.shape
    xt = xt.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    xt = xt.squeeze(0).permute(1, 0, 2)  # [T,H,D]
    return xt * cos + rotate_half(xt) * sin


def simple_gla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    """Reference recurrent simple-GLA for q/k/v [T,H,D], slope [H]."""
    T, H, D = q.shape
    state = torch.zeros((H, D, D), dtype=torch.float32)
    out = torch.empty_like(v, dtype=torch.float32)
    decay = torch.exp(slope.float())
    for t in range(T):
        for h in range(H):
            state[h].mul_(decay[h])
            state[h].add_(torch.outer(k[t, h].float(), v[t, h].float()))
            out[t, h] = state[h].T @ q[t, h].float()
    return out


def slopes(n_heads: int) -> torch.Tensor:
    def pow2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    def get(n: int) -> list[float]:
        if math.log2(n).is_integer():
            return pow2(n)
        p = 2 ** math.floor(math.log2(n))
        return pow2(p) + get(2 * p)[0::2][: n - p]

    return torch.tensor(get(n_heads), dtype=torch.float32)


def stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    diff = (a - b).abs()
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-8)
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float((diff / denom).max().item()) if diff.numel() else 0.0,
        "cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0).item()) if diff.numel() else 1.0,
    }


def print_stats(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
    s = stats(a, b)
    print(
        f"{label:34s} max_abs={s['max_abs']:.6g} "
        f"mean_abs={s['mean_abs']:.6g} max_rel={s['max_rel']:.6g} cos={s['cos']:.8f}"
    )


def check_tensor_conversion(store: TensorStore) -> None:
    pairs = [
        ("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"),
        ("model.layers.0.attention.query_key_value.weight", "blk.0.attn_qkv.weight"),
        ("model.layers.0.attention.query_layernorm.weight", "blk.0.attn_q_norm.weight"),
        ("model.layers.0.attention.key_layernorm.weight", "blk.0.attn_k_norm.weight"),
        ("model.layers.0.attention.g_proj.weight", "blk.0.attn_g_proj.weight"),
        ("model.layers.0.attention.g_norm.weight", "blk.0.attn_g_norm.weight"),
        ("model.layers.0.attention.dense.weight", "blk.0.attn_output.weight"),
        ("model.layers.7.attention.q_a_proj.weight", "blk.7.attn_q_a.weight"),
        ("model.layers.7.attention.q_b_proj.weight", "blk.7.attn_q_b.weight"),
        ("model.layers.7.attention.q_a_layernorm.weight", "blk.7.attn_q_a_norm.weight"),
        ("model.layers.7.attention.kv_a_proj_with_mqa.weight", "blk.7.attn_kv_a_mqa.weight"),
        ("model.layers.7.attention.kv_a_layernorm.weight", "blk.7.attn_kv_a_norm.weight"),
        ("model.layers.7.attention.kv_b_proj.weight", "blk.7.attn_kv_b.weight"),
        ("model.layers.7.attention.dense.weight", "blk.7.attn_output.weight"),
        ("model.layers.7.mlp.gate.weight", "blk.7.ffn_gate_inp.weight"),
        ("model.layers.7.mlp.gate.expert_bias", "blk.7.exp_probs_b.bias"),
    ]
    print("\n== selected HF -> GGUF tensor conversion ==")
    for hf_name, gguf_name in pairs:
        print_stats(gguf_name, store.hf(hf_name), store.gguf(gguf_name))


def check_slopes(store: TensorStore) -> None:
    cfg = store.config
    n_heads = cfg["num_attention_heads"]
    n_layers = cfg["num_hidden_layers"]
    print("\n== linear-attn slope metadata ==")
    for il in [0, 1, 6, 8, 30]:
        expected = -slopes(n_heads) * (1 - (il - 1) / (n_layers - 1) + 1e-5)
        got = store.gguf(f"blk.{il}.attn_g_decay.weight")
        print_stats(f"blk.{il}.attn_g_decay", expected, got)


def check_router(store: TensorStore, seed: int, tokens: int) -> None:
    cfg = store.config
    torch.manual_seed(seed)
    x = torch.randn(tokens, cfg["hidden_size"], dtype=torch.float32) * 0.02
    w_hf = store.hf("model.layers.7.mlp.gate.weight")
    b_hf = store.hf("model.layers.7.mlp.gate.expert_bias")
    w_gg = store.gguf("blk.7.ffn_gate_inp.weight")
    b_gg = store.gguf("blk.7.exp_probs_b.bias")

    def route(w: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(x @ w.T)
        biased = scores + b
        n_group = cfg.get("n_group", 8)
        topk_group = cfg.get("topk_group", 4)
        top_k = cfg["num_experts_per_tok"]
        group_scores = biased.view(tokens, n_group, -1).topk(2, dim=-1).values.sum(dim=-1)
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, True)
        score_mask = group_mask.unsqueeze(-1).expand(tokens, n_group, cfg["num_experts"] // n_group).reshape(tokens, -1)
        masked = biased.masked_fill(~score_mask, float("-inf"))
        _, idx = torch.topk(masked, k=top_k, dim=-1)
        weights = torch.gather(scores, dim=1, index=idx)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * cfg["routed_scaling_factor"]
        return idx, weights

    idx_hf, wt_hf = route(w_hf, b_hf)
    idx_gg, wt_gg = route(w_gg, b_gg)
    print("\n== layer-7 MoE router ==")
    print(f"topk exact match: {bool(torch.equal(idx_hf, idx_gg))}")
    print(f"HF topk[0]:   {idx_hf[0].tolist()}")
    print(f"GGUF topk[0]: {idx_gg[0].tolist()}")
    print_stats("router weights", wt_hf, wt_gg)


def linear_attention_block(store: TensorStore, prefix: str, x: torch.Tensor) -> dict[str, torch.Tensor]:
    cfg = store.config
    H = cfg["num_attention_heads"]
    D = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    rope_dim = cfg["rotary_dim"]
    base = cfg["rope_theta"]
    getter = store.hf if prefix == "hf" else store.gguf
    if prefix == "hf":
        names = {
            "attn_norm": "model.layers.0.input_layernorm.weight",
            "qkv": "model.layers.0.attention.query_key_value.weight",
            "q_norm": "model.layers.0.attention.query_layernorm.weight",
            "k_norm": "model.layers.0.attention.key_layernorm.weight",
            "g_norm": "model.layers.0.attention.g_norm.weight",
            "g_proj": "model.layers.0.attention.g_proj.weight",
            "out": "model.layers.0.attention.dense.weight",
        }
        slope = -slopes(H) * (1 - (0 - 1) / (cfg["num_hidden_layers"] - 1) + 1e-5)
    else:
        names = {
            "attn_norm": "blk.0.attn_norm.weight",
            "qkv": "blk.0.attn_qkv.weight",
            "q_norm": "blk.0.attn_q_norm.weight",
            "k_norm": "blk.0.attn_k_norm.weight",
            "g_norm": "blk.0.attn_g_norm.weight",
            "g_proj": "blk.0.attn_g_proj.weight",
            "out": "blk.0.attn_output.weight",
        }
        slope = getter("blk.0.attn_g_decay.weight")

    x_norm = rms_norm(x, getter(names["attn_norm"]), eps)
    qkv = x_norm @ getter(names["qkv"]).T
    q, k, v = qkv.view(x.shape[0], H + 2 * H, D).split([H, H, H], dim=1)
    q = rms_norm(q, getter(names["q_norm"]), eps)
    k = rms_norm(k, getter(names["k_norm"]), eps)
    pos = torch.arange(x.shape[0], dtype=torch.float32)
    q = rope_split_half(q, pos, rope_dim, base)
    k = rope_split_half(k, pos, rope_dim, base)
    # seg_la applies softmax_scale internally; mirror the llama.cpp graph by
    # pre-scaling q before the simple scan.
    q = q * (D ** -0.5)
    o = simple_gla(q, k, v, slope)
    o = o.reshape(x.shape[0], H * D)
    o = group_rms_norm(o, getter(names["g_norm"]), cfg["group_norm_size"], eps)
    gate = torch.sigmoid(x_norm @ getter(names["g_proj"]).T)
    o = o * gate
    out = o @ getter(names["out"]).T
    return {"x_norm": x_norm, "qkv": qkv, "scan": o, "out": out}


def check_linear0(store: TensorStore, seed: int, tokens: int) -> None:
    cfg = store.config
    torch.manual_seed(seed)
    x = torch.randn(tokens, cfg["hidden_size"], dtype=torch.float32) * 0.02
    hf = linear_attention_block(store, "hf", x)
    gg = linear_attention_block(store, "gguf", x)
    print("\n== layer-0 linear-attn attention block (HF-dequant vs GGUF Q8_0) ==")
    for key in ["x_norm", "qkv", "scan", "out"]:
        print_stats(key, hf[key], gg[key])
        for label, val in [("hf", hf[key]), ("gguf", gg[key])]:
            if not torch.isfinite(val).all():
                raise RuntimeError(f"non-finite values in {label}.{key}")
    print(f"out norms: hf={hf['out'].norm().item():.6g}, gguf={gg['out'].norm().item():.6g}")


def mla_attention_block(store: TensorStore, prefix: str, x: torch.Tensor) -> dict[str, torch.Tensor]:
    cfg = store.config
    H = cfg["num_attention_heads"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    qk_dim = qk_nope + qk_rope
    v_dim = cfg["v_head_dim"]
    kv_rank = cfg["kv_lora_rank"]
    q_rank = cfg["q_lora_rank"]
    eps = cfg["rms_norm_eps"]
    base = cfg["rope_theta"]
    getter = store.hf if prefix == "hf" else store.gguf
    if prefix == "hf":
        names = {
            "attn_norm": "model.layers.7.input_layernorm.weight",
            "q_a": "model.layers.7.attention.q_a_proj.weight",
            "q_a_norm": "model.layers.7.attention.q_a_layernorm.weight",
            "q_b": "model.layers.7.attention.q_b_proj.weight",
            "kv_a": "model.layers.7.attention.kv_a_proj_with_mqa.weight",
            "kv_a_norm": "model.layers.7.attention.kv_a_layernorm.weight",
            "kv_b": "model.layers.7.attention.kv_b_proj.weight",
            "out": "model.layers.7.attention.dense.weight",
        }
    else:
        names = {
            "attn_norm": "blk.7.attn_norm.weight",
            "q_a": "blk.7.attn_q_a.weight",
            "q_a_norm": "blk.7.attn_q_a_norm.weight",
            "q_b": "blk.7.attn_q_b.weight",
            "kv_a": "blk.7.attn_kv_a_mqa.weight",
            "kv_a_norm": "blk.7.attn_kv_a_norm.weight",
            "kv_b": "blk.7.attn_kv_b.weight",
            "out": "blk.7.attn_output.weight",
        }

    x_norm = rms_norm(x, getter(names["attn_norm"]), eps)
    q_lat = rms_norm(x_norm @ getter(names["q_a"]).T, getter(names["q_a_norm"]), eps)
    q = (q_lat @ getter(names["q_b"]).T).view(x.shape[0], H, qk_dim)
    q_nope, q_rot = q.split([qk_nope, qk_rope], dim=-1)

    kv = x_norm @ getter(names["kv_a"]).T
    kv_lat, k_rot = kv.split([kv_rank, qk_rope], dim=-1)
    kv_lat_norm = rms_norm(kv_lat, getter(names["kv_a_norm"]), eps)
    kv_dec = (kv_lat_norm @ getter(names["kv_b"]).T).view(x.shape[0], H, qk_nope + v_dim)
    k_nope, v = kv_dec.split([qk_nope, v_dim], dim=-1)

    pos = torch.arange(x.shape[0], dtype=torch.float32)
    q_rot = rope_interleave_mla(q_rot, pos, qk_rope, base)
    k_rot = rope_interleave_mla(k_rot[:, None, :], pos, qk_rope, base).expand(-1, H, -1)

    q_full = torch.cat((q_nope, q_rot), dim=-1)
    k_full = torch.cat((k_nope, k_rot), dim=-1)
    scores = torch.einsum("thd,shd->hts", q_full, k_full) * (qk_dim ** -0.5)
    causal = torch.triu(torch.ones(x.shape[0], x.shape[0], dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal[None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    attn = torch.einsum("hts,shd->thd", probs, v).reshape(x.shape[0], H * v_dim)
    out = attn @ getter(names["out"]).T
    return {"x_norm": x_norm, "q": q_full, "k": k_full, "attn": attn, "out": out}


def check_mla7(store: TensorStore, seed: int, tokens: int) -> None:
    cfg = store.config
    torch.manual_seed(seed + 1)
    x = torch.randn(tokens, cfg["hidden_size"], dtype=torch.float32) * 0.02
    hf = mla_attention_block(store, "hf", x)
    gg = mla_attention_block(store, "gguf", x)
    print("\n== layer-7 MLA attention block, decompressed causal reference (HF-dequant vs GGUF Q8_0) ==")
    for key in ["x_norm", "q", "k", "attn", "out"]:
        print_stats(key, hf[key], gg[key])
        for label, val in [("hf", hf[key]), ("gguf", gg[key])]:
            if not torch.isfinite(val).all():
                raise RuntimeError(f"non-finite values in {label}.{key}")
    print(f"out norms: hf={hf['out'].norm().item():.6g}, gguf={gg['out'].norm().item():.6g}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, default=Path("/home/bigkahuna/models/hf/inclusionAI/Ling-2.6-flash-fp8"))
    ap.add_argument("--gguf", type=Path, default=Path("/home/bigkahuna/models/gguf/Ling-2.6-flash-fp8-Q8_0.gguf"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    store = TensorStore(Paths(args.hf, args.gguf))
    check_tensor_conversion(store)
    check_slopes(store)
    check_router(store, args.seed, args.tokens)
    check_linear0(store, args.seed, args.tokens)
    check_mla7(store, args.seed, args.tokens)
    print("\nOK: CPU-only Ling-2.6 layer probe completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
