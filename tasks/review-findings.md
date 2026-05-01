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
| **9** | **GroupRMSNorm: `group_norm_size=4` means 1024 groups of 4 channels each** | **REVERSED. It means 4 groups of 1024 channels each. The HF reshape uses `(group_norm_size, hidden/group_norm_size) = (4, 1024)` with normalization over the inner-1024 axis. Confirmed by SGLang's fla `RMSNormGated(group_size = hidden/group_norm_size = 1024)` where `group_size` is channels-per-group.** | §2.2, §5.4 |
| **10** | **Layer pattern formula needs `or il >= L//G * G` tail clause** | **SGLang deployment-canonical impl uses ONLY `(il+1) % G != 0` and asserts `L % G == 0`. For Ling-2.6-flash (L=32, G=8) both forms agree. Drop the tail clause; assert clean multiples instead.** | §2.2, §2.4 |
| **11** | **Layer-attn `q` is unscaled** | **For seg_la backend, the kernel applies `softmax_scale = head_dim^(-0.5)` internally to q. For our ggml impl we will pre-scale q in the graph builder before the scan op.** | §5.4 |
| **12** | **Linear-attn computation in fp16/bf16** | **SGLang casts QKV to F32 (`qkv = qkv.to(torch.float32)`) before the linear-attn computation. Match this in graph for numerical parity.** | §5.4 |

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

### 2.5 Slope sign convention: HF / fla style (already-negated)
Two possible storage conventions for the per-layer per-head decay tensor:
- **HF / fla** (what we use): bake `g = -base_slopes * scale` (negative) at convert time. The scan op directly applies `S = exp(g) * S + ...`.
- **SGLang seg_la**: stores positive `slope` in a buffer; kernel does `decay_scale = -tl.load(slope) ; ratio = exp(decay_scale)`. Same math, different storage sign.

We match HF/fla because (a) the HF modeling code is the closest to training-time numerics, (b) one less negation in the kernel. **Document this in the `ggml_simple_gla_scan` op header.**

### 2.6 Slope formula: HF off-by-one preserved
**Conflict**: HF reference (`modeling_bailing_moe_v2_5.py`) uses `1 - (layer_idx - 1)/(num_layers - 1) + 1e-5`. SGLang's `LightningAttentionBackend._build_slope_tensor` (a generic Lightning Attention impl) uses `1 - layer_id/(num_hidden_layers - 1) + 1e-5` (no `-1` offset). They differ by one layer of phase.

For V1 we use **HF's formula** (the off-by-one variant). Rationale:
- HF code is shipped alongside the checkpoint and is closest to training-time semantics.
- SGLang's `lightning_backend.py` is generic and may not be the actual V2.5 deployment path (V2.5's seg_la backend has its own `decay_scales` initialization that I have not been able to locate in the SGLang fork).
- If numerical parity fails on the linear-attn branch, swap to SGLang's formula and re-test. This is a **single-line change in the converter** (`(il - 1)` → `il`).

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

## 4. Findings from SGLang reference (antgroup/sglang `ling_2_6` branch)

Here are the deployment-canonical conventions, drawn from `models/bailing_moe_linear.py`, `models/bailing_moe_nextn.py`, `configs/bailing_hybrid.py`, and `layers/attention/linear/{seg_la.py, lightning_attn.py, lightning_backend.py}`:

### 4.1 Recurrent state geometry
Mamba2-style state per layer: shape `[B, num_heads, head_dim, head_dim]` = `[B, 32, 128, 128]` for Ling-2.6-flash. **No conv1d** (unlike Kimi KDA). `n_embd_s = 524288` per sequence per layer (~2 MiB at F32). Confirmed.

### 4.2 RoPE conventions
- Linear-attn: `is_neox_style=True` → ggml `GGML_ROPE_TYPE_NEOX`. Applied on partial dim (rotary_dim = head_dim * partial_rotary_factor = 64).
- MLA: `is_neox_style = not config.rope_interleave`. With `rope_interleave=True` → `is_neox_style=False` → ggml mode 0 (default/interleaved). Applied only on `qk_rope_head_dim=64` of the q_pe / k_pe slice.

### 4.3 Linear-attn forward (canonical)
```
qkv = QKV(x).to(float32)         # cast to F32 for numerical stability
q, k, v = split(qkv, ...)
if use_qk_norm: q,k = RMSNorm_per_head(q,k)
q, k = rotary_emb_neox(positions, q, k)   # only first 64 of head_dim rotated
# q is NOT explicitly scaled here; the kernel applies softmax_scale = head_dim^(-0.5) internally
hidden = seg_la_attn(q, k, v, decay_scales, state)
gate = g_proj(hidden_states_input)        # NB: input to attention block, NOT post-attn
hidden = g_norm(hidden, gate)             # fused RMSNormGated with sigmoid activation
hidden = dense(hidden)
```

For our ggml graph we'll pre-scale q in the builder (so the scan op signature stays clean), use F32 for the QKV matmuls, and split the fused `RMSNormGated` into separate `ggml_rms_norm` + `ggml_mul(sigmoid(gate))`.

### 4.4 GroupRMSNorm semantics
```
x_grouped = rearrange(x, "... (g d) -> ... g d", d=group_size)
rstd = 1 / sqrt(x_grouped.square().mean(dim=-1, keepdim=True) + eps)
```
With `group_size = hidden / group_norm_size = 4096 / 4 = 1024` (channels per group). RMS computed over the 1024-channel inner axis. **Multiplied by per-channel weight `[hidden]`.**

In ggml: reshape `[hidden, T*B]` → `[1024, 4, T*B]`, `ggml_rms_norm` over axis 0, reshape back, `ggml_mul` with weight.

### 4.5 MTP (NEXTN) details
- `BailingMoEModelNextN` has its own `enorm`, `hnorm`, `eh_proj`, `final_layernorm`, ONE decoder layer (always MLA, `attention_type=1`), and a `lm_head` that is **runtime-tied** to the main model's `lm_head` via `set_embed_and_head` (used by SGLang's eagle_worker for speculative decoding).
- The MTP file `model-mtp-layer.safetensors` has NO `embed_tokens` and NO `lm_head` — confirmed by tensor index and matches SGLang's tying logic.
- SGLang exposes MTP via `--speculative-algorithm NEXTN` with `--speculative-num-steps 3`, `--speculative-num-draft-tokens 4`. In llama.cpp, MTP integration with the speculative-decoding API is a **follow-up PR**; the V1 PR will load and graph-build the MTP layer but not hook it into draft sampling.
- **MTP MLA layer**: same q-LoRA / kv-LoRA dimensions as main MLA layers. Same `kv_b_proj` split applies.
- **Tool-call format**: SGLang uses `--tool-call-parser qwen25`, suggesting the V2.5 chat template's tool-call XML format is qwen2.5-compatible.

### 4.6 SGLang weight-loader transformations (informational)
- `attention.dense → attention.out_proj` for linear-attn layers; `attention.dense → attention.o_proj` for MLA layers (and MTP). Both originate from the same checkpoint name. We don't need to mirror this rename in our converter; our schema uses one `ATTN_OUT` for both.
- `q_a_proj` and `kv_a_proj_with_mqa` are fused into `fused_qkv_a_proj_with_mqa` at runtime when `q_lora_rank` is set. **Performance optimization, not required for correctness**. Defer to follow-up PR if profiling shows it matters.
- `slope` keys are explicitly skipped during weight loading (`if "slope" in name: continue`). Slopes are **not in the checkpoint** — they are non-persistent buffers per HF `register_buffer(persistent=False)`. We synthesize them at convert time. Already in plan.

### 4.7 Long-context: YARN at runtime, not at convert
The HF `config.json` has `rope_scaling: null` and `max_position_embeddings: 131072`. The README's recommended SGLang launch command overrides at runtime to:
```
rope_scaling = { rope_type: "yarn", factor: 2.0, rope_theta: 6000000, partial_rotary_factor: 0.5, original_max_position_embeddings: 131072 }
```
Yielding 262144 effective context. **This is a deployment-time override, not embedded in the checkpoint.** Our converter should NOT bake YARN; the resulting GGUF will inherit `max_position_embeddings = 131072` and standard rope. Users can override via llama-cli's `--rope-scaling` / `--rope-freq-scale` runtime flags (or by patching the GGUF metadata) to enable YARN context extension.

Document this in the README/usage notes when shipping.

## 5. Open questions worth flagging to upstream reviewers

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

## 6. Verification protocol rationale

### 5.1 Why F32 for parity testing
Hybrid recurrent-attention models accumulate FP errors over the sequence length. BF16 vs F16 vs F32 each give different cumulative drift. F32 on both reference and our implementation removes one source of variance and isolates algorithmic correctness.

### 5.2 Why bisect linear-attn and MLA layers separately
The two attention types share the same input/output norms and FFN block, but otherwise have no overlap. If parity fails on one but not the other, we know exactly which branch is broken. Forcing layer 0 (always linear) and layer 7 (always MLA) gives the cleanest isolated tests.

### 5.3 Why argmax-match as a hard gate
Logit RMS error can be small while the top-1 token differs (when the top two logits are close). For language models, top-1 mismatch on >0 positions is a real bug. Tolerance on logit values is for diagnostics; argmax-match is the hard gate.
