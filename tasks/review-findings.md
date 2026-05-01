# Review Findings: Ling-2.6-flash Plan

Companion to `tasks/todo.md`. Captures **decisions made**, **prior-plan corrections**, **non-obvious gotchas**, and **open questions a reviewer might raise**. Read this before implementing if anything in `todo.md` looks surprising — the rationale lives here.

---

## 1. Corrections to the original draft plan

These were wrong or imprecise in earlier drafts and have been fixed in `todo.md`:

| # | Original assumption | Reality | Fixed in `todo.md` |
|---|---|---|---|
| 1 | MLA RoPE uses `GGML_ROPE_TYPE_NEOX` due to `rope_interleave=True` | MLA uses interleaved/default mode (`mode=0`); **linear-attn** uses NeoX (split-half) | §5.4, §5.5 |
| 2 | All layers share one tensor naming scheme | Linear-attn layers use V2 names (`query_key_value`, `dense`, ...); MLA layers use DeepSeek-V3 names (`q_a_proj`, `kv_b_proj`, ...) | §1.2, §2.5 |
| 3 | Slope formula = simple `-build_slope_tensor(H) * (1 - il/L)` | Actual HF code uses `(layer_idx - 1)`, which gives a >1 multiplier at `il=0`, plus `+1e-5` term | §2.3 |
| 4 | `kv_b_proj` loaded as-is | Must be split into transposed `k_b_proj` + `v_b_proj` for absorbed MLA | §2.3 |
| 5 | `num_key_value_heads` left at 32 | Must be forced to 1 to enable MQA-style absorbed MLA | §2.2 |
| 6 | Encode hybrid pattern via dedicated `recurrent_layer_arr` field | Existing convention is **per-layer `head_count_kv` list with 0 = recurrent** | §2.2, §3.2 |
| 7 | `g_proj` consumes the post-attention output | `g_proj` consumes the `input_layernorm` output of the layer (parallel input gate, not output gate of the attention) | §5.3, §5.4 |
| 8 | Effort estimate ~2–4 weeks | Revised to ~1.5–3 weeks once V2 reuse is properly accounted for; the new ggml op is the dominant cost | §4 (entire phase) |

---

## 2. Decisions locked in

### 2.1 Architecture string: `"bailingmoe2.5"`
Alternatives considered:
- `"bailing-hybrid"` (matches HF `model_type`) — rejected: breaks the existing `bailingmoe`/`bailingmoe2` family naming.
- `"bailingmoe25"` — rejected: ambiguous, hard to read.
- `"bailingmoe2_5"` — possible, but `.` is the more common separator in arch strings (e.g. nothing currently uses `_` as a version separator).

**Locked in: `"bailingmoe2.5"`.** Reviewer may push back; if so, `"bailingmoe2_5"` is the fallback.

### 2.2 GLA op name: `ggml_simple_gla_scan`
Matches `flash-linear-attention`'s `simple_gla` terminology (the algorithm Ling uses). Alternatives:
- `ggml_lightning_attn` — rejected: "Lightning Attention" refers to multiple variants; ambiguous.
- `ggml_gla_scan` — rejected: full GLA has data-dependent gating; ours is the *fixed-decay* simpler case.

### 2.3 Chat template: rely on runtime Jinja for V1 PR
V2.5's `chat_template.jinja` adds tool-calling and multi-step tool-response detection beyond what `LLM_CHAT_TEMPLATE_BAILING2` handles. Options:
- (a) Extend the C++ template to handle tools — significant code, hard to maintain in C++.
- (b) Add a new `LLM_CHAT_TEMPLATE_BAILING25` arm with a partial implementation — half-measure.
- (c) Rely on runtime Jinja support (already in tree).

**Decision: (c).** The C++ builtin is a fast-path optimization; correctness comes from Jinja. Add a regression test in `tests/test-chat-template`.

### 2.4 GPU backends deferred
First PR ships CPU-only `ggml_simple_gla_scan`. Rationale: CUDA/Metal kernels for new scan ops typically take 2-3× as long as the CPU reference and add review surface. Track follow-up in a separate issue.

---

## 3. Non-obvious gotchas (in priority order)

### 3.1 The `(layer_idx - 1)` term in the slope formula looks like a bug
HF source:
```python
slope = -BailingMoeV2_5LinearAttention.build_slope_tensor(self.num_heads) * (
    1 - (self.layer_idx - 1) / (self.config.num_hidden_layers - 1) + 1e-5
)
```
For `layer_idx=0`, this gives multiplier `1 - (-1)/31 + 1e-5 ≈ 1.0323`, i.e., **stronger** decay than at any deeper layer. That's likely an off-by-one that shipped — but we **must replicate it byte-for-byte** because the trained weights expect this exact slope distribution. Do **not** "fix" it.

### 3.2 GroupRMSNorm vs RMSNorm
`g_norm` weight has shape `[4096]` (per-channel scale) but normalization happens within groups of 4 channels. This is **not** equivalent to:
- Plain RMSNorm over 4096 channels (would normalize all together).
- Per-head RMSNorm (would normalize over `head_dim=128`).
Implementation: reshape to `[4, 1024, T*B]`, apply `ggml_rms_norm` (which normalizes the inner-most axis), reshape back, multiply by per-channel scale. **Don't write a new ggml op for this.** §5.4 in `todo.md` shows the exact reshape.

### 3.3 MTP layer always uses MLA
The layer-type formula `(il+1) % G == 0` would suggest MTP at `il=32` is *not* MLA (`33 % 8 = 1 ≠ 0`). But HF hard-codes MLA in `BailingMoeV2_5MTPLayer`. Don't apply the formula to MTP; treat it as MLA unconditionally. The MTP file's tensor names (`q_a_proj`, `kv_b_proj`, etc.) already confirm this.

### 3.4 `lm_head` is shared between main model and MTP
The MTP `safetensors` file has no `lm_head.weight`. HF code uses `self.lm_head(mtp_hidden_states)` — same module. In the converter, do **not** look for an MTP-specific output head; in the graph builder, reuse `model.output` for the MTP logits projection.

### 3.5 Position-tracking anchor in hybrid memory
`past_seen_tokens` for the whole model must be read from the MLA layer's KV cache, not from any linear-attn layer's recurrent state (which doesn't track positions). Use `get_seq_length(layer_idx = layer_group_size - 1)`. Kimi-Linear has the same pattern; copy it verbatim.

### 3.6 `n_embd_s()` does NOT include `head_dim_qk` from MLA
The recurrent state size is determined by the **linear-attn** branch, not the MLA branch. Linear-attn uses `head_dim=128` (full, not `qk_head_dim=192`), so:
```
n_embd_s() = head_dim_k_linear * head_dim_v_linear * n_head_linear
           = 128 * 128 * 32
           = 524288 F32 elements per sequence per layer
```
Do not confuse with MLA's `n_embd_head_k_mla = 192`.

### 3.7 The HF non-interleave branch has `x = 1/0` — production model REQUIRES `rope_interleave=True`
In `BailingMoeV2_5MultiLatentAttention.forward`:
```python
if self.config.rope_interleave:
    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
else:
    x = 1 / 0   # crash
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
```
We must implement only the interleave path for MLA. If a future variant flips this flag, our code must detect it and choose the right ggml RoPE mode.

### 3.8 `apply_rotary_pos_emb_interleave` is an in-place layout converter
The function takes input that is laid out as GPT-J interleaved (`a0, b0, a1, b1, ...`), converts to split-halves via `view + transpose + reshape`, then applies NeoX-style rotation, and **returns in split-halves layout**. This means downstream attention sees split-halves Q/K. Net effect end-to-end is **NeoX rotation applied to GPT-J-laid-out input weights**.

In ggml terms, two equivalent options:
1. Call `ggml_rope_ext` with `mode=0` (default/interleaved) directly on the original tensor — the rotation pairs match.
2. Pre-permute `q_b_proj` and `kv_a_proj_with_mqa` weights at convert time so they output split-halves directly, then use `mode=GGML_ROPE_TYPE_NEOX`.

**Recommend option 1** for V1 (simpler converter). Verify with numerical parity in §6.2; switch to option 2 if option 1 fails parity.

### 3.9 Quantization F32 pinning is critical for the slope and norm tensors
The slope tensor is small (`[32]` per layer × 28 layers = 896 F32 values total) but quantizing it would destroy the exponential decay precision. Norms and the router are similarly sensitive. See §6.4 for the full list.

---

## 4. Open questions worth flagging to upstream reviewers

### 4.1 Should `simple_gla_scan` live in ggml core or a new `ggml-fla` (flash-linear-attn) bucket?
KDA, RWKV6/7, Mamba SSM scan, and Qwen3-Next gated-delta-net all live alongside each other in `ggml-cpu/ops/`. New op fits the same bucket. No new namespace needed.

### 4.2 Naming: `attn_g_decay` vs `attn_decay` vs `attn_slope`?
"Slope" is the HF term but is misleading — it's a log-decay rate. "Decay" is clearer. Locked to `attn_g_decay` to match the HF variable name `g` and other GLA-family ops (`g_norm`, `g_proj`).

### 4.3 Should we expose MTP logits via `llama_decode`?
Existing nextn-supporting models (DeepSeek-V3, GLM-4.5) currently discard MTP logits at runtime — they're useful only for speculative decoding. Mirror that behavior; don't plumb MTP logits through the public API in V1.

### 4.4 Can we share more code with `bailingmoe2`?
The MoE block is byte-identical. Two options:
- (a) Refactor `build_moe_bailingmoe2(...)` as a free function and call it from both. Cleanest.
- (b) Copy-paste with a TODO. Keeps the V2 PR untouched.

Recommend (a) **as a separate refactor PR before** the V2.5 PR, to keep diffs reviewable.

### 4.5 Lite/smaller variant compatibility
`BailingMoeV2_5Config` defaults differ from the Ling-2.6-flash config (e.g., `num_attention_heads=16`, `num_hidden_layers=20`, `q_lora_rank=None`). A future Lite checkpoint could:
- Lack `q_lora_rank` → MLA path would use the non-LoRA Q projection (`q_proj` instead of `q_a_proj`/`q_b_proj`). Converter must handle both.
- Have `num_kv_heads_for_linear_attn != num_attention_heads` → linear-attn would need real GQA via `repeat_kv`. Op signature already supports this; converter must not assert equality.

`todo.md §2.5` flags asserting tensor presence rather than relying on the formula — that's the right guard for these variants.

---

## 5. Verification protocol rationale

### 5.1 Why F32 for parity testing
Hybrid recurrent-attention models accumulate FP errors over the sequence length. BF16 vs F16 vs F32 each give different cumulative drift. F32 on both reference and our implementation removes one source of variance and isolates algorithmic correctness.

### 5.2 Why bisect linear-attn and MLA layers separately
The two attention types share the same input/output norms and FFN block, but otherwise have no overlap. If parity fails on one but not the other, we know exactly which branch is broken. Forcing layer 0 (always linear) and layer 7 (always MLA) gives the cleanest isolated tests.

### 5.3 Why argmax-match as a hard gate
Logit RMS error can be small while the top-1 token differs (when the top two logits are close). For language models, top-1 mismatch on >0 positions is a real bug. Tolerance on logit values is for diagnostics; argmax-match is the hard gate.
