# Implementation Plan: Ling-2.6-flash (BailingMoeV2.5) support in llama.cpp

Hybrid MLA + Lightning-Attention-2 MoE architecture (`BailingMoeV2_5ForCausalLM`, HF `model_type = "bailing_hybrid"`).

**Reference HF model**: `inclusionAI/Ling-2.6-flash` (`config.json`, `modeling_bailing_moe_v2_5.py`).

**Reference llama.cpp components**:
- `convert_hf_to_gguf.py::BailingMoeV2Model`, `KimiLinearModel`
- `src/models/bailingmoe2.cpp` (MoE block)
- `src/models/deepseek2.cpp` (MLA with q-LoRA)
- `src/models/kimi-linear.cpp` (hybrid memory dispatch)
- `src/llama-chat.cpp::LLM_CHAT_TEMPLATE_BAILING2` (chat template)

Companion document: `tasks/review-findings.md` (open decisions, prior-plan corrections, edge-case rationale).

---

## Phase 0: Naming & conventions (decide before coding)

- [ ] **Architecture enum/string**: use `LLM_ARCH_BAILINGMOE2_5` ↔ `"bailingmoe2.5"`. Locked in by the existing `bailingmoe`/`bailingmoe2` precedent. (See `review-findings.md §16` for rejected alternatives.)
- [ ] **Slope tensor name**: store per-layer at `blk.{i}.attn_g_decay` (F32, shape `[n_head]`). Add to `gguf-py/gguf/constants.py` as `MODEL_TENSOR.ATTN_G_DECAY`.
- [ ] **GLA op name**: `ggml_simple_gla_scan` (matches `flash-linear-attention`'s `simple_gla` terminology).

---

## Phase 1: GGUF schema & tensor mapping

### 1.1 `gguf-py/gguf/constants.py` ✅
- [x] Add `MODEL_ARCH.BAILINGMOE2_5 = auto()` and `"bailingmoe2.5"` mapping.
- [x] Add new `MODEL_TENSOR` entries: `ATTN_G_PROJ`, `ATTN_G_NORM`, `ATTN_G_DECAY`.
- [x] Define `MODEL_ARCH.BAILINGMOE2_5` tensor list (35 entries).
  Notes from implementation:
  - Reused existing `NEXTN_EH_PROJ`, `NEXTN_ENORM`, `NEXTN_HNORM` for MTP block.
  - Mapped MTP `final_layernorm` to existing `LAYER_OUT_NORM` (matches bailingmoe2's mapping of the same source name).
  - `NEXTN_EMBED_TOKENS`, `NEXTN_SHARED_HEAD_HEAD`, `NEXTN_SHARED_HEAD_NORM` deliberately omitted — V2.5 MTP shares `tok_embd` and `lm_head` with main model (verified against `model-mtp-layer.safetensors` index).
  - Single `ATTN_OUT`, `ATTN_NORM`, `FFN_NORM` covers both attention types (HF uses identical names across linear-attn and MLA layers).

### 1.2 `gguf-py/gguf/tensor_mapping.py` ✅
- [x] Added 3 new tensor mappings: `ATTN_G_PROJ`, `ATTN_G_NORM`, `ATTN_G_DECAY`.
- [x] Extended existing MLA mappings (`ATTN_Q_A`, `ATTN_Q_B`, `ATTN_KV_A_MQA`, `ATTN_KV_B`, `ATTN_K_B`, `ATTN_V_B`, `ATTN_Q_A_NORM`, `ATTN_KV_A_NORM`) with bailingmoe2.5 source names (`model.layers.{bid}.attention.*` instead of deepseek2's `self_attn.*`).
- [x] Smoke test (28/28 sample HF tensor names map correctly across linear-attn + MLA + MTP layers + common tensors).
- [x] `pytest gguf-py/tests/` passes (5/5).

### 1.3 `gguf-py/gguf/gguf_writer.py` and constants
- [ ] Add new KVs (or confirm existing reused):
    - `bailingmoe2.5.attention.layer_group_size` (uint32) — for documentation/sanity; runtime uses per-layer `head_count_kv` list (§2.4)
    - `bailingmoe2.5.attention.group_norm_size` (uint32) — for GroupRMSNorm group factor (=4)
    - `bailingmoe2.5.expert_gating_func` = `sigmoid` (already supported via existing KV)
    - All MLA KVs already supported by `add_q_lora_rank`, `add_kv_lora_rank`, `add_key_length_mla`, `add_value_length_mla`

---

## Phase 2: Python converter (`convert_hf_to_gguf.py`) ✅

### 2.1 Subclass `BailingMoeV2Model` ✅
- [x] Register: `@ModelBase.register("BailingMoeV2_5ForCausalLM")`.
- [x] `model_arch = gguf.MODEL_ARCH.BAILINGMOE2_5`.
- [x] `__init__` inherited from `BailingMoeV2Model` (already sets block_count from nextn).

### 2.2 `set_gguf_parameters` ✅
- [x] Force `num_key_value_heads=1` before super (mirrors KimiLinear).
- [x] `setdefault("norm_topk_prob", True)` for the missing V2.5 config key.
- [x] Per-layer `head_count_kv` list (28 zeros + 4 ones for L=32 G=8, plus 1 for MTP).
- [x] MLA KVs: `add_q_lora_rank`, `add_kv_lora_rank`, `add_key_length_mla`, `add_value_length_mla`.
- [x] MoE / leading-dense / expert-weights / nextn KVs inherited from V2 super.
- [x] `add_group_norm_groups((n_head * head_dim) // group_norm_size)`.
- [x] `add_rope_dimension_count` inherited from V2 super (head_dim * partial_rotary_factor = 64).
- [x] `add_rope_freq_base` inherited from TextModel super (reads `rope_theta=6_000_000`).

### 2.3 `modify_tensors` overrides ✅
- [x] `kv_b_proj` split into transposed `k_b_proj` + `v_b_proj`, original kept for non-absorbed fallback. All three flow through V2's super for expert/bias handling.
- [x] Expert bias rename + expert stacking inherited from V2's `modify_tensors`.
- [x] Slope tensors yielded via `generate_extra_tensors` (see 2.4) and routed through the standard `modify_tensors` pipeline.
- [x] Slope formula bit-exactness verified vs hand-computed reference: max abs diff = 0.0 for n=32.

### 2.4 MTP weight loading ✅
- [x] `generate_extra_tensors` loads `model-mtp-layer.safetensors` and yields its tensors with HF-style names. Logs a warning (not error) if the file is missing.
- [x] MTP tensor names handled via the existing schema mapping (Phase 1):
  - `enorm` → `NEXTN_ENORM`, `hnorm` → `NEXTN_HNORM`, `eh_proj` → `NEXTN_EH_PROJ`
  - `final_layernorm` → `LAYER_OUT_NORM` (matches V2's same mapping)
  - MLA + MoE tensors share schema with main-stack equivalents (bid=32)
- [x] Slope tensors for the 28 linear-attn layers also yielded from `generate_extra_tensors`.
- [x] Verified MTP file has no `lm_head` → main `output` is shared (Phase 5 graph builder will reuse it).

### 2.5 Robustness: derive layer type from tensor presence (DEFERRED)
- [ ] After loading all tensors, assert per-layer (recurrent has qkv+g_proj+g_norm; MLA has q_a_proj+kv_b_proj). Cross-check against the formula-driven head_count_kv list.
- Status: **deferred to follow-up**. Mismatch will surface as a clear missing-tensor error in the C++ loader; the assertion is defensive polish.

### 2.6 Vocab
- [x] `_set_vocab_gpt2()` inherited from V2 (no override needed).
- [ ] **Verify pretokenizer hash**: run `convert_hf_to_gguf_update.py` against an actual checkpoint; confirm matches the existing `bailingmoe2` entry. (Requires HF download; deferred.)
- [ ] **Verify EOS**: confirm `eos_token_id=156895` resolves to `<|role_end|>` and is correctly emitted in GGUF. (Requires HF download; deferred.)

---

## Phase 3: C++ architecture & hparams

### 3.1 `src/llama-arch.{h,cpp}`
- [ ] Add `LLM_ARCH_BAILINGMOE2_5` enum value.
- [ ] Add `{ LLM_ARCH_BAILINGMOE2_5, "bailingmoe2.5" }` to the name table.
- [ ] Register all per-layer tensors in the schema map (mirrors V2's entry plus MLA + linear-attn-extras).
- [ ] Add `LLM_ARCH_BAILINGMOE2_5` to the hybrid-arch list (alongside `LLM_ARCH_KIMI_LINEAR`, `LLM_ARCH_QWEN3NEXT`) in `llama-arch.cpp`'s recurrent-arch checks.

### 3.2 `src/llama-hparams.{h,cpp}`
- [ ] Reuse existing `recurrent_layer_arr` (already present for hybrid archs). Populate from the per-layer `head_count_kv` list during model load: `recurrent_layer_arr[il] = (head_count_kv[il] == 0)`.
- [ ] No new hparams fields needed; reuse `n_embd_head_k_mla`, `n_embd_head_v_mla`, `n_lora_q`, `n_lora_kv`, `n_rot`.

### 3.3 `src/llama-model.cpp`
- [ ] Add `case LLM_ARCH_BAILINGMOE2_5` in `load_hparams` — mirror the BAILINGMOE2 case plus MLA hparams loading from `KIMI_LINEAR`.
- [ ] Add `case LLM_ARCH_BAILINGMOE2_5` in `load_tensors` — per layer, gate on `is_recurrent(il)`:
    - Recurrent: load `attn_qkv`, `attn_q_norm`, `attn_k_norm`, `attn_out`, `attn_g_proj`, `attn_g_norm`, `attn_g_decay`.
    - MLA: load `attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_a_norm`, `attn_kv_b`, `attn_k_b`, `attn_v_b`, `attn_out`.
    - All layers: `attn_norm`, `ffn_norm`.
    - First `n_layer_dense_lead` layers: dense `ffn_{gate,up,down}`. Others: MoE block (`ffn_gate_inp`, `ffn_exp_probs_b`, expert tensors, shared expert tensors).
    - MTP layer (il=32): MLA tensors + MoE block + four MTP norms (`enorm`, `hnorm`, `eh_proj`, `final_layernorm`).
- [ ] Set `model.output = model.lm_head` (no MTP-specific output head).
- [ ] Set hybrid memory params:
    - `n_embd_r() = 0` (no conv state, unlike Kimi).
    - `n_embd_s() = qk_head_dim * v_head_dim * n_head = 192 * 128 * 32` — wait, **state dimensions are `head_dim_k × head_dim_v × n_head`** for the recurrence. Linear-attn uses full Q/K head dim (128, after RoPE applied to first 64), so `n_embd_s() = 128 * 128 * 32 = 524288` F32 elements per sequence.
- [ ] Add `case LLM_ARCH_BAILINGMOE2_5` to the model-build dispatch: `llm = std::make_unique<llm_build_bailingmoe2_5>(*this, params)`.
- [ ] Add to graph type / output type case lists (model-print, output-name, etc. — follow `LLM_ARCH_KIMI_LINEAR` precedent).

### 3.4 Chat template (`src/llama-chat.cpp`)
- [ ] **Diff** V2.5's `chat_template.jinja` against the existing `LLM_CHAT_TEMPLATE_BAILING2`:
    - Both: `<role>SYSTEM</role>...<|role_end|><role>HUMAN</role>...<|role_end|><role>ASSISTANT</role>...`
    - V2.5 adds: tool-calling block (`<tools>...</tools>`, `<tool_call>...</tool_call>`), `detailed thinking on/off` toggle, multi-step tool-response detection.
- [ ] **Decide**: (a) extend `LLM_CHAT_TEMPLATE_BAILING2` to handle the new toggles, (b) add `LLM_CHAT_TEMPLATE_BAILING25` arm, or (c) rely on runtime Jinja for full fidelity.
    - Recommendation: (c) for V1 PR (Jinja support is already in tree). Add a `tests/test-chat-template` Jinja-mode regression test to ensure parity.

---

## Phase 4: New ggml operator `ggml_simple_gla_scan`

### 4.1 API
- [ ] Add to `ggml/include/ggml.h`:
    ```c
    GGML_API struct ggml_tensor * ggml_simple_gla_scan(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,        // [D_k, H, T, B] F32/F16
        struct ggml_tensor  * k,        // [D_k, H, T, B]
        struct ggml_tensor  * v,        // [D_v, H, T, B]
        struct ggml_tensor  * g,        // [H]            F32 — log-decay per head (already negative)
        struct ggml_tensor  * state);   // [D_k, D_v, H, B] F32 — IN/OUT
    ```
- [ ] Output: `[D_v, H, T, B]`. State updated in-place (or via `ggml_cpy` pattern, mirroring `ggml_kda_scan`).

### 4.2 Mathematical contract
Per `(b, h)` independently, iterating `t = 0..T-1`:
```
S ← exp(g[h]) · S + outer(k[:,h,t,b], v[:,h,t,b])     # S: [D_k, D_v]
o[:,h,t,b] ← S^T · q[:,h,t,b]                          # [D_v]
```
Initial `S` taken from the input `state` tensor. Final `S` written back to `state`.

**`g` is pre-negated and pre-ramped** (the slope baking from §2.3 produces `g` directly usable here).

### 4.3 Mode dispatch (matches HF)
- [ ] If `T <= 64`: recurrent path (sequential per-token loop). Optimal for decode.
- [ ] If `T > 64`: chunked path (block size 64 typical) — per-chunk matmul accumulation. Optimal for prefill.

### 4.4 CPU kernel (`ggml/src/ggml-cpu/`)
- [ ] Reference implementation in `ggml-cpu.c` (or new `ggml-cpu/ops/simple-gla.c`).
- [ ] Follow the layout of `ggml_kda_scan` / `ggml_rwkv_wkv6` for thread sharding (typically thread per `(B, H)` pair).
- [ ] F32 accumulation, F32 state storage (regardless of input dtype).

### 4.5 GPU kernels (defer to follow-up PR if needed)
- [ ] CUDA: model after `ggml/src/ggml-cuda/kda.cu` or `wkv6.cu`. Tile per `(B, H)`, register-tile the state.
- [ ] Metal: similar.
- [ ] Vulkan/SYCL: lower priority.

### 4.6 Tests
- [ ] Add unit test in `tests/test-backend-ops.cpp` comparing CPU output against a numpy/python reference for random inputs at `T ∈ {1, 8, 64, 128, 1024}`.
- [ ] Numerical tolerance: rtol=1e-4, atol=1e-5 in F32.

---

## Phase 5: Graph builder `src/models/bailingmoe2_5.cpp`

### 5.1 Skeleton
- [ ] Subclass `llm_graph_context` (NOT `llm_build_delta_net_base` — we don't use the delta-net algebra). Mirror `llm_build_kimi_linear` for the hybrid-memory bookkeeping.
- [ ] Header: declare `llm_build_bailingmoe2_5` in `src/models/models.h`.

### 5.2 Hybrid memory setup (copy verbatim from `kimi-linear.cpp`)
- [ ] `auto * inp_k = build_inp_mem_hybrid_k();`
- [ ] `auto * inp_rs = inp_k->get_recr();`
- [ ] `auto * inp_attn_k = inp_k->get_attn();`
- [ ] **Position tracking**: anchor `past_seen_tokens` on the MLA layer cache (`get_seq_length(layer_idx = layer_group_size - 1)`) — exactly as kimi-linear does, otherwise positions drift between branches.

### 5.3 Per-layer dispatch
For each `il` in `0 .. n_layer - num_nextn_predict_layers - 1`:
```cpp
cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);  // input_layernorm output, used by g_proj too
ggml_tensor * x_norm = cur;  // capture for g_proj input
if (hparams.is_recurrent(il)) {
    cur = build_linear_attn(cur, layer, il, inp_rs, x_norm);
} else {
    cur = build_mla(cur, layer, il, inp_attn_k);
}
// residual + FFN/MoE block
```

### 5.4 Linear attention branch (`build_linear_attn`)
- [ ] Fused QKV: `qkv = ggml_mul_mat(layer.attn_qkv, cur)`. Shape `[16384, T, B]`.
- [ ] Reshape & split: `q[D=128, H=32, T, B]`, `k[D=128, H=32, T, B]`, `v[D=128, H=32, T, B]` (kv_heads_for_linear == n_heads in this checkpoint; assert).
- [ ] **QK-norm** (per-head RMSNorm over `D=128`): `q = build_norm(q, layer.attn_q_norm, NULL, LLM_NORM_RMS, il)`; same for `k` with `attn_k_norm`. Weight shape `[128]`.
- [ ] **RoPE on first 64 dims, NeoX (split-half) layout**: `q_rope = ggml_rope_ext(q, ..., n_rot=64, mode=GGML_ROPE_TYPE_NEOX, freq_base=6e6, ...)`. Same for `k`. (HF's linear-attn uses `apply_rotary_pos_emb`, the split-half variant.)
- [ ] Load `g = layer.attn_g_decay` (already F32, shape `[32]`).
- [ ] Get/build state: `state = build_rs(inp_rs, mctx_cur->get_s_l(il), n_embd_s(), n_seqs)`, reshape to `[D_k=128, D_v=128, H=32, n_seqs]`.
- [ ] **Run new op**: `o = ggml_simple_gla_scan(ctx0, q, k, v, g, state)` — output `[128, 32, T, B]`, state updated.
- [ ] **GroupRMSNorm** (group size = 4):
    ```cpp
    // o has shape [4096, T*B] flattened
    // Reshape to [4, 1024, T*B] so ggml_rms_norm normalizes the inner-4 axis
    o = ggml_reshape_3d(ctx0, o, 4, 1024, n_tokens);
    o = ggml_rms_norm(ctx0, o, eps);
    o = ggml_reshape_2d(ctx0, o, 4096, n_tokens);
    o = ggml_mul(ctx0, o, layer.attn_g_norm);  // per-channel learned scale [4096]
    ```
- [ ] **Sigmoid output gate**: `g_proj_out = ggml_mul_mat(layer.attn_g_proj, x_norm)`; `o = o * sigmoid(g_proj_out)`. `x_norm` is the input-layernorm output (NOT the post-attention output).
- [ ] **Dense output**: `o = ggml_mul_mat(layer.attn_out, o)`.
- [ ] State writeback handled by `build_rs` machinery (mirror kimi-linear).

### 5.5 MLA branch (`build_mla`)
Direct adaptation of `src/models/deepseek2.cpp` (with q-LoRA, since `q_lora_rank=1536`).
- [ ] Q LoRA path: `q = q_b(q_a_norm(q_a(x_norm)))`.
- [ ] KV compression: `kv_cmpr_pe = kv_a_mqa(x_norm)`; split into `kv_cmpr [512]` and `k_pe [64]`.
- [ ] Normalize: `kv_cmpr = kv_a_norm(kv_cmpr)`.
- [ ] **RoPE — interleaved/default mode (NOT NeoX)**: `q_pe = ggml_rope_ext(q_pe, ..., mode=0, freq_base=6e6, ...)`; same for `k_pe`. (HF uses `apply_rotary_pos_emb_interleave` which net-effects to GPT-J interleaved layout.)
    - **Verification step**: numerical-parity test (Phase 6) is the gate. If parity fails on the MLA path, swap the mode flag and/or pre-permute `q_b_proj`/`kv_a_proj_with_mqa` weights at convert time.
- [ ] **Absorbed-MLA path** (using `wk_b`, `wv_b` from §2.3 split): identical to kimi-linear's MLA branch.
- [ ] Output projection: `attn_out = dense(attn_output)`.

### 5.6 FFN / MoE block (copy from `bailingmoe2.cpp`)
- [ ] Layer 0 (`il < hparams.n_layer_dense_lead`): dense `build_ffn` with SiLU.
- [ ] Other layers: `build_moe_ffn` with `expert_gating_func=sigmoid`, `expert_weights_norm=true`, `expert_weights_scale=2.5`, `n_expert=256`, `n_expert_used=8`. Plus shared expert add.

### 5.7 MTP layer (il = 32)
After the main residual stream is finalized (post-`output_norm`):
```cpp
ggml_tensor * mtp_in_emb = build_inp_embd(model.tok_embd);  // shifted input embeds
mtp_in_emb = roll_tensor(mtp_in_emb, -1);  // shift left by 1, pad with 0
ggml_tensor * e = build_norm(mtp_in_emb, layer32.enorm, NULL, LLM_NORM_RMS, 32);
ggml_tensor * h = build_norm(main_hidden, layer32.hnorm, NULL, LLM_NORM_RMS, 32);
ggml_tensor * x = ggml_concat(ctx0, e, h, /*dim=*/0);  // last dim = 8192
x = ggml_mul_mat(layer32.eh_proj, x);                  // -> 4096
ggml_tensor * residual = x;
x = build_norm(x, layer32.input_layernorm, NULL, LLM_NORM_RMS, 32);
x = build_mla(x, layer32, 32, inp_attn_k);             // MLA always for MTP
x = ggml_add(residual, x);
residual = x;
x = build_norm(x, layer32.post_attention_layernorm, NULL, LLM_NORM_RMS, 32);
x = build_moe_ffn(x, ...);  // reuse MoE machinery; MTP layer has its own expert weights
x = ggml_add(residual, x);
x = build_norm(x, layer32.final_layernorm, NULL, LLM_NORM_RMS, 32);
mtp_logits = ggml_mul_mat(model.output, x);  // SHARED lm_head
```
- [ ] Expose `mtp_logits` separately (mirror DeepSeek-V3 / GLM-4.5 nextn handling).

### 5.8 Output head
- [ ] `result_norm = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1)`.
- [ ] `result_output = ggml_mul_mat(model.output, result_norm)` (lm_head; not tied to embeddings).

---

## Phase 6: Verification & testing

### 6.1 Convert + smoke test
- [ ] Run `convert_hf_to_gguf.py` on a downloaded checkpoint. Output should list 33 layers (32 main + 1 MTP).
- [ ] Inspect with `gguf-dump.py`:
    - Per-layer `head_count_kv` list contains 28 zeros and 4 ones (linear:MLA = 28:4 for L=32, G=8) plus 1 trailing one for MTP.
    - Slope tensors present at all 28 recurrent layers, F32, shape `[32]`.
    - MLA splits: `attn_k_b`, `attn_v_b` present for the 4 MLA layers + MTP.

### 6.2 Numerical parity vs HF transformers
- [ ] Build a side-by-side test harness:
    - HF reference: `BailingMoeV2_5ForCausalLM.from_pretrained(..., torch_dtype=torch.float32)`, eager attention.
    - llama.cpp: F32 inference, single thread, deterministic (no flash-attn fallback).
- [ ] Inputs: 5 fixed prompts of length 256.
- [ ] Compare:
    - Logits at positions {0, 1, 64, 128, 255}: rtol=1e-3, atol=1e-4.
    - Top-1 token argmax must match in 100% of compared positions.
- [ ] If parity fails, bisect by:
    1. Embeddings only (truncate to 1 layer): verify token embedding identity.
    2. One linear-attn layer end-to-end (force layer 0): isolates `simple_gla_scan` + GroupRMSNorm + RoPE-NeoX correctness.
    3. One MLA layer end-to-end (force layer 7): isolates MLA RoPE convention + absorbed-KV path.
    4. Full main stack: isolates layer-pattern bookkeeping.
    5. Add MTP: isolates MTP graph.

### 6.3 Runtime tests
- [ ] `llama-cli` 64-token completion at FP16 for sanity.
- [ ] `llama-bench` quick run for tok/s sanity (compare vs Kimi-Linear of similar size as ballpark).
- [ ] `tests/test-chat-template` (Jinja mode) to verify chat-template parity.

### 6.4 Quantization
- [ ] **Pin F32**: all `*_norm.weight` tensors (`attn_norm`, `ffn_norm`, `attn_q_norm`, `attn_k_norm`, `attn_g_norm`, `attn_q_a_norm`, `attn_kv_a_norm`, `enorm`, `hnorm`, `final_layernorm`, `output_norm`), `attn_g_decay`, `ffn_gate_inp` (router), `ffn_exp_probs_b` (expert bias).
- [ ] Quantize normally (Q4_K_M as default test target): all `*_proj`, `query_key_value`, `dense`, `g_proj`, `q_a/q_b`, `kv_a/kv_b`, `k_b`/`v_b`, MoE expert tensors, `tok_embd`, `output`, `eh_proj`.
- [ ] Run perplexity on wiki.test.raw at Q4_K_M; flag if >5% degradation vs F16.

### 6.5 Documentation
- [ ] Add Ling-2.6-flash to `README.md` supported-models table.
- [ ] Add a short note in `docs/development/HOWTO-add-model.md` (or wherever Kimi-Linear's notes live) explaining the slope-baking convention, in case future Lightning-Attention models reuse the op.

---

## Phase 7: PR mechanics
- [ ] Open a feature-request issue on `ggml-org/llama.cpp` first (none exists for V2.5) to allow upstream review of the new op signature before deep implementation.
- [ ] Split into two PRs if reviewer requests: (1) `ggml_simple_gla_scan` op + tests, (2) BailingMoeV2.5 model integration consuming the op.
- [ ] Mark GPU backends as TODO in PR description if shipping CPU-only; track in a follow-up issue.