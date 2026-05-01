#!/usr/bin/env python3
"""Validate Ling-2.6-flash / BailingMoeV2.5 GGUF metadata for Phases 1-3.

This is a structural smoke test only. It intentionally does not run inference;
Phase 5 graph building and Phase 4 simple_gla_scan are still pending.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gguf-py"))

from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

ARCH = "bailingmoe2.5"


def field(reader: GGUFReader, name: str):
    f = reader.get_field(name)
    if f is None:
        raise AssertionError(f"missing GGUF field: {name}")
    return f.contents()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    args = ap.parse_args()

    reader = GGUFReader(args.gguf)
    tensors = {t.name: t for t in reader.tensors}

    arch = field(reader, "general.architecture")
    assert arch == ARCH, f"architecture mismatch: {arch!r} != {ARCH!r}"

    n_layer = field(reader, f"{ARCH}.block_count")
    head_kv = field(reader, f"{ARCH}.attention.head_count_kv")
    group_norm_groups = field(reader, f"{ARCH}.attention.group_norm_groups")
    nextn = field(reader, f"{ARCH}.nextn_predict_layers")
    expert_group_field = reader.get_field(f"{ARCH}.expert_group_count")
    expert_group_used_field = reader.get_field(f"{ARCH}.expert_group_used_count")

    assert n_layer == 33, f"expected 33 layers (32 main + 1 MTP), got {n_layer}"
    assert nextn == 1, f"expected nextn_predict_layers=1, got {nextn}"
    assert len(head_kv) == 33, f"expected head_count_kv length 33, got {len(head_kv)}"
    assert head_kv[:32].count(0) == 28, f"expected 28 recurrent layers, got {head_kv[:32].count(0)}"
    assert [i for i, v in enumerate(head_kv[:32]) if v == 1] == [7, 15, 23, 31], head_kv
    assert head_kv[32] == 1, f"expected MTP layer head_count_kv=1, got {head_kv[32]}"
    assert group_norm_groups == 4, f"expected group_norm_groups=4, got {group_norm_groups}"
    if expert_group_field is not None:
        expert_groups = expert_group_field.contents()
        assert expert_groups == 8, f"expected expert_group_count=8, got {expert_groups}"
    else:
        expert_groups = None
    if expert_group_used_field is not None:
        expert_groups_used = expert_group_used_field.contents()
        assert expert_groups_used == 4, f"expected expert_group_used_count=4, got {expert_groups_used}"
    else:
        expert_groups_used = None

    recurrent_layers = [i for i, v in enumerate(head_kv[:32]) if v == 0]
    mla_layers = [i for i, v in enumerate(head_kv) if v == 1]

    for il in recurrent_layers:
        name = f"blk.{il}.attn_g_decay.weight"
        assert name in tensors, f"missing linear-attn decay tensor: {name}"
        t = tensors[name]
        assert list(t.shape) == [32], f"{name} shape {list(t.shape)} != [32]"
        assert t.tensor_type == GGMLQuantizationType.F32, f"{name} qtype {t.tensor_type} != F32"
        for suffix in ["attn_qkv.weight", "attn_g_proj.weight", "attn_g_norm.weight"]:
            assert f"blk.{il}.{suffix}" in tensors, f"missing linear-attn tensor blk.{il}.{suffix}"

    for il in mla_layers:
        for suffix in ["attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight"]:
            assert f"blk.{il}.{suffix}" in tensors, f"missing MLA tensor blk.{il}.{suffix}"
        assert list(tensors[f"blk.{il}.attn_k_b.weight"].shape) == [128, 512, 32], \
            f"blk.{il}.attn_k_b.weight shape {list(tensors[f'blk.{il}.attn_k_b.weight'].shape)} != [128, 512, 32]"
        assert list(tensors[f"blk.{il}.attn_v_b.weight"].shape) == [512, 128, 32], \
            f"blk.{il}.attn_v_b.weight shape {list(tensors[f'blk.{il}.attn_v_b.weight'].shape)} != [512, 128, 32]"

    mtp = 32
    for suffix in ["nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "layer_output_norm.weight"]:
        assert f"blk.{mtp}.{suffix}" in tensors, f"missing MTP tensor blk.{mtp}.{suffix}"

    print("OK: Ling-2.6-flash GGUF structural validation passed")
    print(f"  file: {args.gguf}")
    print(f"  tensors: {len(reader.tensors)}")
    print(f"  recurrent layers: {recurrent_layers}")
    print(f"  MLA/MTP layers: {mla_layers}")
    print(f"  group_norm_groups: {group_norm_groups}")
    if expert_groups is None or expert_groups_used is None:
        print("  expert groups: missing in GGUF (loader fallback uses 8 groups / 4 used)")
    else:
        print(f"  expert groups: {expert_groups}, used: {expert_groups_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
