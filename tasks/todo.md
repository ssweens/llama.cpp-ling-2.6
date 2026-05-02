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
- [x] Per-layer `head_count_kv` list using SGLang's simpler predicate `(il+1) % G != 0`.
  Asserts `L % G == 0` (matches SGLang's deployment assumption). Tail clause from
  HF reference dropped (would never fire for L=32 G=8 anyway).
- [x] MLA KVs: `add_q_lora_rank`, `add_kv_lora_rank`, `add_key_length_mla`, `add_value_length_mla`.
- [x] MoE / leading-dense / expert-weights / nextn KVs inherited from V2 super.
- [x] `add_group_norm_groups(hparams["group_norm_size"])` — stores **number of groups** (=4),
  not channels-per-group. Verified semantics against SGLang `fla.RMSNormGated` and HF
  `BailingMoeV2_5GroupRMSNorm.forward`. See `review-findings.md` §1#9 and §4.4.
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

## Phase 3: C++ architecture & hparams ✅ (except graph dispatch, deferred to Phase 5)

### 3.1 `src/llama-arch.{h,cpp}` ✅
- [x] Add `LLM_ARCH_BAILINGMOE2_5` enum value.
- [x] Add `{ LLM_ARCH_BAILINGMOE2_5, "bailingmoe2.5" }` to the name table.
- [x] Add 3 new tensor enums: `ATTN_G_PROJ`, `ATTN_G_NORM`, `ATTN_G_DECAY`.
- [x] Add `LLM_ARCH_BAILINGMOE2_5` to `llm_arch_is_hybrid`.
- [x] Add `LLM_ARCH_BAILINGMOE2_5` to `llm_arch_supports_sm_tensor` false list (matches KIMI_LINEAR precedent).
- [x] Wire global tensor name strings + `LLM_TENSOR_INFOS` entries (MUL_MAT/MUL/SSM_SCAN buffer hints).

### 3.2 `src/llama-hparams.{h,cpp}` ✅
- [x] Reuse existing `recurrent_layer_arr` (set by load_hparams from per-layer `head_count_kv == 0`).
- [x] No new hparams fields needed; reuse `n_embd_head_k_mla_impl`, `n_embd_head_v_mla_impl`, `n_lora_q`, `n_lora_kv`, `n_rot`, `n_norm_groups`, `nextn_predict_layers`.
- [x] Add `attn_g_proj`, `attn_g_norm`, `attn_g_decay` tensor pointers to `struct llama_layer`.

### 3.3 `src/llama-model.cpp` ✅ (loader; graph dispatch stubbed)
- [x] `load_hparams` case for `LLM_ARCH_BAILINGMOE2_5`: reads V2 MoE KVs + MLA dims (key_length_mla, value_length_mla, q_lora_rank, kv_lora_rank) + group_norm_groups. Populates `recurrent_layer_arr`. Normalizes absorbed-MLA KV-cache key width to `kv_lora_rank + n_rot = 576` while keeping linear-attn head_dim as `hidden/n_head = 128`. Sets simple-GLA recurrent state via existing SSM fields (`ssm_d_state=128`, `ssm_d_inner=4096`, so `n_embd_s=524288`; `ssm_d_conv=0`, so `n_embd_r=0`). Type detection by n_layer (32 or 33 → LLM_TYPE_100B_A6B for Ling-2.6-flash).
- [x] `load_tensors` case dispatches per-layer on `is_mtp` and `is_recurrent`:
    - Linear-attn: attn_qkv, attn_out, attn_q_norm, attn_k_norm, attn_g_proj, attn_g_norm, attn_g_decay.
    - MLA (and MTP): wq_a, attn_q_a_norm, wq_b, wkv_a_mqa, attn_kv_a_norm, wkv_b + wk_b + wv_b (converter emits all three; graph can choose absorbed path), wo.
    - FFN: dense for first n_layer_dense_lead layers, MoE for the rest. MTP always MoE.
    - MTP: enorm, hnorm, eh_proj, layer_out_norm. tok_embd and output (lm_head) shared with main.
- [x] Add `LLM_ARCH_BAILINGMOE2_5` to NEOX rope_type list.
- [x] **Graph dispatch case**: stubbed with a `runtime_error` describing Phase 5 status. Loader works; inference will throw a clear message until Phase 5 lands.
- [x] Build verified clean (`cmake --build build --target llama -j 4`).

### 3.4 Chat template (`src/llama-chat.cpp`)
- [ ] **Diff** V2.5's `chat_template.jinja` against the existing `LLM_CHAT_TEMPLATE_BAILING2`:
    - Both: `<role>SYSTEM</role>...<|role_end|><role>HUMAN</role>...<|role_end|><role>ASSISTANT</role>...`
    - V2.5 adds: tool-calling block (`<tools>...</tools>`, `<tool_call>...</tool_call>`), `detailed thinking on/off` toggle, multi-step tool-response detection.
- [ ] **Decide**: (a) extend `LLM_CHAT_TEMPLATE_BAILING2` to handle the new toggles, (b) add `LLM_CHAT_TEMPLATE_BAILING25` arm, or (c) rely on runtime Jinja for full fidelity.
    - Recommendation: (c) for V1 PR (Jinja support is already in tree). Add a `tests/test-chat-template` Jinja-mode regression test to ensure parity.

---

## Phase 4: New ggml operator `ggml_simple_gla_scan`

### 4.1 API ✅
- [x] Add to `ggml/include/ggml.h`:
    ```c
    GGML_API struct ggml_tensor * ggml_simple_gla_scan(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,        // [D_k, H, T, B] F32
        struct ggml_tensor  * k,        // [D_k, H, T, B] F32
        struct ggml_tensor  * v,        // [D_v, H, T, B] F32
        struct ggml_tensor  * g,        // [H]            F32 — log-decay per head (already negative)
        struct ggml_tensor  * state);   // [D_k, D_v, H, B] F32 — input state
    ```
- [x] Return a packed F32 tensor, mirroring `ggml_gated_delta_net`:
    - first `D_v * H * T * B` elements: output viewable as `[D_v, H, T, B]`
    - remaining `D_k * D_v * H * B` elements: new state viewable as `[D_k, D_v, H, B]`
    - Phase 5 graph builder will slice the new-state view and copy it back with `ggml_cpy`.

### 4.2 Mathematical contract ✅
Per `(b, h)` independently, iterating `t = 0..T-1`:
```
S ← exp(g[h]) · S + outer(k[:,h,t,b], v[:,h,t,b])     # S: [D_k, D_v]
o[:,h,t,b] ← S^T · q[:,h,t,b]                          # [D_v]
```
Initial `S` taken from the input `state` tensor. Final `S` is returned in the packed new-state region (Phase 5 copies it back to recurrent memory).

**`g` is pre-negated and pre-ramped** (the slope baking from §2.3 produces `g` directly usable here).

### 4.3 Mode dispatch (matches HF) ✅ for V1 reference / follow-up for perf
- [x] V1 CPU reference uses the recurrent/sequential loop for all `T`. This is correct and sufficient for Phase 5 functional bring-up.
- [ ] Follow-up optimization: if `T > 64`, add a chunked path (block size 64 typical) for prefill throughput.

### 4.4 CPU kernel (`ggml/src/ggml-cpu/`) ✅
- [x] Reference implementation in `ggml/src/ggml-cpu/ops.cpp`.
- [x] Follow `ggml_gated_delta_net` sharding pattern: parallel chunks over `(B, H)` pairs.
- [x] F32 accumulation and F32 state/output storage. V1 API accepts F32 inputs; Phase 5 graph casts QKV to F32 before calling the op.

### 4.5 GPU kernels
- [x] CUDA F32 reference/perf kernel for `D_k == D_v ∈ {64, 128}` with one CUDA block per `(B, H)` pair. It keeps the `[D_k, D_v]` state row for one value channel in registers and writes the packed output/new-state layout expected by the CPU op. Verified the state layout is `[D_k, D_v, H, B]`; an early transposed load/store failed backend-op parity and was corrected before committing.
- [ ] CUDA follow-up optimization: tune register pressure / occupancy and add a chunked prefill path for larger `T` if profiling shows `simple_gla_scan` remains material.
- [ ] Metal: similar.
- [ ] Vulkan/SYCL: lower priority.

### 4.6 Tests ✅
- [x] Add a dedicated CPU reference test (`tests/test-simple-gla.cpp`) that compares packed op output + final state against a direct C++ reference for deterministic inputs at `T ∈ {1, 4, 8}` and `B ∈ {1, 2}`.
- [x] Add `tests/test-backend-ops.cpp` coverage so backend support/perf harness can see `GGML_OP_SIMPLE_GLA_SCAN`.
- [x] Add CUDA-sized backend-op cases (`D=64`, `D=128`) so CUDA support is actually exercised; `./build/bin/test-backend-ops -o SIMPLE_GLA_SCAN` passes on CUDA0/CUDA1/CUDA2.
- [x] Numerical tolerance: rtol=1e-4, atol=1e-5 in F32.

---

## Phase 5: Graph builder `src/models/bailingmoe2_5.cpp`

### 5.1 Skeleton ✅
- [x] Subclass `llm_graph_context` (NOT `llm_build_delta_net_base` — we don't use the delta-net algebra). Mirror `llm_build_kimi_linear` for the hybrid-memory bookkeeping.
- [x] Header: declare `llm_build_bailingmoe2_5` in `src/models/models.h`.
- [x] Add `src/models/bailingmoe2_5.cpp` and wire `LLM_ARCH_BAILINGMOE2_5` graph dispatch in `llama-model.cpp`.

### 5.2 Hybrid memory setup ✅
- [x] `auto * inp_k = build_inp_mem_hybrid_k();`
- [x] `auto * inp_rs = inp_k->get_recr();`
- [x] `auto * inp_attn_k = inp_k->get_attn();`
- [x] Context-init smoke confirms 4 MLA KV-cache layers (7/15/23/31), MTP layer 32 without KV cache, and 28 recurrent layers with 56 MiB RS buffer.
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

### 5.4 Linear attention branch (`build_linear_attn`) ✅
- [x] Fused QKV: `qkv = ggml_mul_mat(layer.attn_qkv, cur)`. Shape `[3 * n_head * head_dim = 12288, T, B]`.
  **Cast to F32** to match SGLang's numerical precision (`qkv = qkv.to(float32)`); ggml will keep it F32 through the scan.
- [x] Reshape & split: `q[D=128, H=32, T, B]`, `k[D=128, H=32, T, B]`, `v[D=128, H=32, T, B]` (kv_heads_for_linear == n_heads in this checkpoint; assert).
- [x] **QK-norm** (per-head RMSNorm over `D=128`): `q = build_norm(q, layer.attn_q_norm, NULL, LLM_NORM_RMS, il)`; same for `k` with `attn_k_norm`. Weight shape `[128]`.
- [x] **RoPE on first 64 dims, NeoX (split-half) layout**: `q_rope = ggml_rope_ext(q, ..., n_rot=64, mode=GGML_ROPE_TYPE_NEOX, freq_base=6e6, ...)`. Same for `k`. (Confirmed via SGLang `is_neox_style=True` for linear-attn.)
- [x] **Pre-scale q** by `1/sqrt(head_dim)`. Rationale: SGLang's `seg_la` kernel applies `softmax_scale = head_dim^(-0.5)` internally; we apply it explicitly before our scan op so the op signature stays clean. `q = ggml_scale(q, 1.0/sqrt(head_dim))`.
- [x] Load `g = layer.attn_g_decay` (already F32, shape `[32]`, already negated per HF/fla convention; see `review-findings.md` §2.5).
- [x] Get/build state: `state = build_rs(inp_rs, mctx_cur->get_s_l(il), n_embd_s(), n_seqs)`, reshape to `[D_k=128, D_v=128, H=32, n_seqs]`.
- [x] **Run new op**: `packed = ggml_simple_gla_scan(ctx0, q, k, v, g, state)`; slice packed output `[128,32,T,B]` and packed new-state `[128,128,32,B]`, then copy new-state back to recurrent memory.
- [x] **GroupRMSNorm** (4 groups of 1024 channels each — see `review-findings.md` §4.4): implemented as reshape `[1024,4,n_tokens]` → `ggml_rms_norm` → reshape `[4096,n_tokens]` → learned scale `attn_g_norm`.
- [x] **Sigmoid output gate**: `g_proj_out = ggml_mul_mat(layer.attn_g_proj, x_norm)`; `o = o * sigmoid(g_proj_out)`. `x_norm` is the input-layernorm output (NOT the post-attention output). Equivalent to SGLang's fused `RMSNormGated(activation="sigmoid")`.
- [x] **Dense output**: `o = ggml_mul_mat(layer.attn_out, o)`.
- [x] State writeback handled by `build_rs` machinery (mirror kimi-linear).

### 5.5 MLA branch ✅ (`build_mla`)
Direct adaptation of `src/models/deepseek2.cpp` (with q-LoRA, since `q_lora_rank=1536`).
- [x] Q LoRA path: `q = q_b(q_a_norm(q_a(x_norm)))`.
- [x] KV compression: `kv_cmpr_pe = kv_a_mqa(x_norm)`; split into `kv_cmpr [512]` and `k_pe [64]`.
- [x] Normalize: `kv_cmpr = kv_a_norm(kv_cmpr)`.
- [x] **RoPE — interleaved/default mode (NOT NeoX)**: `q_pe = ggml_rope_ext(q_pe, ..., mode=0, freq_base=6e6, ...)`; same for `k_pe`. (HF uses `apply_rotary_pos_emb_interleave` which net-effects to GPT-J interleaved layout.)
    - **Verification step**: numerical-parity test (Phase 6) is the gate. If parity fails on the MLA path, swap the mode flag and/or pre-permute `q_b_proj`/`kv_a_proj_with_mqa` weights at convert time.
- [x] **Absorbed-MLA path** (using `wk_b`, `wv_b` from §2.3 split): identical to kimi-linear's MLA branch. Phase 5 validation found the base KV-cache key length must be `kv_lora_rank + qk_rope_head_dim = 576`, while linear-attn still uses `hidden/n_head = 128`; converter and loader now encode/normalize that.
- [x] Output projection: `attn_out = dense(attn_output)`.

### 5.6 FFN / MoE block (copy from `bailingmoe2.cpp`) ✅
- [x] Layer 0 (`il < hparams.n_layer_dense_lead`): dense `build_ffn` with SiLU.
- [x] Other layers: `build_moe_ffn` with `expert_gating_func=sigmoid`, `expert_weights_norm=true`, `expert_weights_scale=2.5`, `n_expert=256`, `n_expert_used=8`. Plus shared expert add.

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

### 5.8 Output head ✅
- [x] `result_norm = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1)`.
- [x] `result_output = ggml_mul_mat(model.output, result_norm)` (lm_head; not tied to embeddings).

### 5.9 Phase 5 smoke status ✅ (main path)
- [x] CPU-only no-decode context/graph smoke: `./build/bin/llama-simple -m /home/bigkahuna/models/gguf/Ling-2.6-flash-fp8-Q8_0.gguf -n 0 -ngl 0 hello`.
  Result: loads all 573 tensors, initializes 576 MiB MLA K cache + 56 MiB simple-GLA recurrent state, reserves BailingMoeV2.5 graph successfully (3048 nodes, 1 split), exits without decode.
- [ ] Actual decode/logit validation remains Phase 6 (CPU full-token eval is slow; use GPU or layer-isolated tests for parity).

---

## Phase 6: Verification & testing

### 6.1 Convert + smoke test ✅ (Phase 1-3 validated; superseded by Phase 5 graph smoke)
- [x] Download `inclusionAI/Ling-2.6-flash-fp8` for Phase 1-3 validation. Rationale: same `BailingMoeV2_5ForCausalLM` / `bailing_hybrid` architecture as BF16, includes the MTP layer, ~109GB source checkpoint instead of ~200GB, ungated. Converter dequantizes FP8 tensors internally, then writes GGUF (`--outtype q8_0` for this smoke test).
  - Note: `inclusionAI/Ling-2.6-flash-int4` is smaller (~65GB) but omits `model-mtp-layer.safetensors` / `model.layers.32.*`, so it cannot validate Phase 3 MTP loading despite `num_nextn_predict_layers=1` in its config.
- [x] Run `convert_hf_to_gguf.py` on the downloaded FP8 checkpoint. Output GGUF: `/home/bigkahuna/models/gguf/Ling-2.6-flash-fp8-Q8_0.gguf` (573 tensors, ~107GB on disk / 114.3G logical tensor payload).
- [x] Inspect with `tasks/validate_ling26_gguf.py`:
    - Per-layer `head_count_kv` list contains 28 zeros and 4 ones (linear:MLA = 28:4 for L=32, G=8) plus 1 trailing one for MTP.
    - `group_norm_groups == 4`.
    - Slope tensors present at all 28 recurrent layers, F32, shape `[32]`.
    - MLA splits: `attn_k_b`, `attn_v_b` present for the 4 MLA layers + MTP, with corrected shapes `[128,512,32]` / `[512,128,32]`.
- [x] Exercise C++ loader on the converted GGUF with `./build/bin/llama-simple -m ... -n 1 -ngl 0 hello` before Phase 5. Result at that checkpoint: metadata + all 573 tensors loaded, hybrid KV/recurrent memory initialized (56 MiB RS buffer), then context creation reached the intentional `bailingmoe2.5 graph builder is not yet implemented` runtime error. Phase 5 now replaces this with a real main-path graph builder; see §5.9.

### 6.2 Numerical parity vs HF transformers
- [ ] Build a side-by-side test harness:
    - HF reference: `BailingMoeV2_5ForCausalLM.from_pretrained(..., torch_dtype=torch.float32)`, eager attention.
    - llama.cpp: F32 inference, single thread, deterministic (no flash-attn fallback).
- [ ] Inputs: 5 fixed prompts of length 256.
- [ ] Compare:
    - Logits at positions {0, 1, 64, 128, 255}: rtol=1e-3, atol=1e-4.
    - Top-1 token argmax must match in 100% of compared positions.
- [ ] CPU-only status (2026-04-30): full HF-vs-llama parity is not practical on this host without a smaller layer-isolated harness; the available checkpoint + Q8 GGUF already consume ~200GB on disk and full-model HF F32/BF16 CPU load would exceed the intended validation budget.
- [ ] CPU runtime sanity found the graph executes without crashes, but greedy 64-token output from the Q8_0 GGUF is degenerate (`0000...`). Treat this as a parity/quality blocker until layer-isolated comparisons identify whether the issue is linear-attn math, MLA RoPE/absorbed KV, MoE routing, or FP8->Q8 conversion quality.
- [x] Add CPU-only layer/component probe harness: `tasks/ling26_layer_probe.py`. It loads only selected safetensors/GGUF tensors and checks HF→GGUF tensor conversion, slope metadata, layer-7 MoE routing, layer-0 linear-attn attention-block math, and layer-7 MLA decompressed causal-reference math on short synthetic hidden states.
- [x] Initial probe result (`python3 tasks/ling26_layer_probe.py --tokens 4 --threads 4`): selected tensor conversion, slope metadata, MoE router, layer-0 linear-attn attention block, and layer-7 MLA attention block all match closely between HF-dequantized weights and GGUF Q8_0 (cosine >= 0.99995 for probed block outputs). This makes converter tensor mapping, slope baking, group-limited router metadata, simple-GLA reference math, and MLA decompressed tensor flow unlikely causes of the degenerate full-model output.
- [ ] Active quality-debug pass (2026-05-01): isolate why full graph generates degenerate tokens despite passing attention-subblock probes.
    - [ ] Check logits/output-head on the prompt token(s): inspect top-k distribution and whether special/control tokens dominate before sampler state can affect output.
    - [x] Extend `tasks/ling26_layer_probe.py` from attention-subblock-only to full layer-0 block including residual + dense FFN/MoE branch. Result: layer-0 full block remains close (cos ≈ 0.999972), so dense FFN mapping/residual order are unlikely causes.
    - [x] Add layer-7 absorbed-MLA-vs-decompressed parity probe if layer 0 is clean, because the existing probe validates decompressed MLA math but not llama.cpp's absorbed KV-cache graph path. Result: absorbed split-kv path matches decompressed GGUF reference (out cos ≈ 0.999999), so split `k_b/v_b` math is unlikely.
    - [x] Add sparse layer-7 MoE expert-output probe for selected routed experts; the current harness validates router top-k/weights but not packed expert tensor layout or expert FFN math. Result: selected-expert MoE output remains close (out cos ≈ 0.999957), so packed expert tensor layout is unlikely.
    - [x] Probe linear-attention decay sign by temporarily negating `attn_g_decay` in the graph and rebuilding. Result: output got worse, so keep the original negative log-decay convention.
    - [x] Root cause found (2026-05-01): C++ linear-attention Q/K/V views used `d_inner` as the token stride after fused `query_key_value`, but HF/SGLang lay each token row out as `[Q | K | V]` with stride `3 * d_inner`. Multi-token prompts therefore read later tokens from the wrong QKV segment. Fixed by using a `3 * d_inner` token stride and materializing contiguous 4D Q/K/V views before `ggml_simple_gla_scan`.
    - [x] If block probes pass, inspect full-stack bookkeeping: layer pattern, final norm, logits projection, MTP layer inclusion/exclusion, and tokenizer/template control-token handling.
    - [x] Runtime prompt/template sanity after stride fix: Q2_K full-offload chat prompt now produces coherent output (`Llamas are gentle, social herd animals native to the Andes.`). Q8_0 CPU chat prompt also produces coherent output (`Llamas are gentle, social herd animals known for their distinctive banana-shaped ears and soft, woolly coats.`). Logs: `tasks/logs/ling26_qkv_stride_fix_q2k_chat_sentence.log`, `tasks/logs/ling26_qkv_stride_fix_q8_cpu_chat_sentence.log`.
- [ ] If parity fails, bisect by:
    1. Embeddings only (truncate to 1 layer): verify token embedding identity.
    2. One linear-attn layer end-to-end (force layer 0): isolates `simple_gla_scan` + GroupRMSNorm + RoPE-NeoX correctness. Initial harness covers the attention sub-block but not the FFN/residual.
    3. One MLA layer end-to-end (force layer 7): isolates MLA RoPE convention + absorbed-KV path. Initial harness covers decompressed causal MLA reference but not llama.cpp's absorbed KV-cache graph path directly.
    4. Full main stack: isolates layer-pattern bookkeeping.
    5. Add MTP: isolates MTP graph.

### 6.3 Runtime tests (CPU-only pass)
- [x] `llama-simple` one-token decode, Q8_0, `-ngl 0`: graph executes and decodes 1 token without runtime errors. Logs:
    - `tasks/logs/ling26_phase6_cpu_decode_n1.log`
    - `tasks/logs/ling26_phase6_cpu_decode_n1_after_moe_defaults.log`
- [x] 64-token CPU-only completion at Q8_0 (substituted for FP16 because disk is too tight for an FP16/BF16 GGUF): completed without runtime errors, but output was degenerate zeros; keep as a Phase 6 quality blocker. Log: `tasks/logs/ling26_phase6_cpu_completion64.log`.
- [x] `llama-bench` quick CPU-only microbench (`-ngl 0 -p 1 -n 1 -r 1 -b 1 -ub 1 -t 16 -fa 0`): `pp1 0.74 t/s`, `tg1 1.37 t/s`. Log: `tasks/logs/ling26_phase6_cpu_bench_p1n1.log`.
- [x] `tests/test-chat-template` passed via `ctest --test-dir build -R test-chat-template --output-on-failure`.
- [ ] `llama-cli` 64-token completion at FP16 remains not run in this CPU-only pass: this build has `LLAMA_BUILD_SERVER=OFF`, so the `llama-cli` target is not available, and the current disk budget cannot hold an FP16/BF16 GGUF for Ling-2.6-flash.

### 6.4 Quantization
- [ ] **Pin F32**: all `*_norm.weight` tensors (`attn_norm`, `ffn_norm`, `attn_q_norm`, `attn_k_norm`, `attn_g_norm`, `attn_q_a_norm`, `attn_kv_a_norm`, `enorm`, `hnorm`, `final_layernorm`, `output_norm`), `attn_g_decay`, `ffn_gate_inp` (router), `ffn_exp_probs_b` (expert bias).
- [ ] Quantize normally (Q4_K_M as default test target): all `*_proj`, `query_key_value`, `dense`, `g_proj`, `q_a/q_b`, `kv_a/kv_b`, `k_b`/`v_b`, MoE expert tensors, `tok_embd`, `output`, `eh_proj`.
- [x] Disk/VRAM-constrained smoke quant: generated Q2_K from the validated Q8_0 GGUF with `--allow-requantize --leave-output-tensor`. Output: `/home/bigkahuna/models/gguf/Ling-2.6-flash-fp8-Q2_K.gguf` (~37 GiB). Dry-run estimate was 37,753 MiB (~36.9 GiB). IQ2_XXS/IQ2_XS are smaller but require an imatrix for actual quantization in this build; Q3_K_S/Q3+ were too close to free disk.
- [x] Validate Q2_K GGUF with `tasks/validate_ling26_gguf.py`: passed structural checks (573 tensors, expected recurrent/MLA pattern, group_norm_groups=4, expert groups=8/4).
- [x] Build a full CUDA-enabled binary in the existing `build/` dir (`GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, full arch list `75-virtual;80-virtual;86-real;89-real;120a-real;121a-real`). `ccache` installed/capped at 2 GiB before build.
- [x] CPU smoke on Q2_K: `./build/bin/llama-simple -m ...Q2_K.gguf -n 1 -ngl 0 hello` passed; eval ~90 ms/token on CPU for the 1-token smoke.
- [x] CPU quality smoke on Q2_K with GPUs hidden: `CUDA_VISIBLE_DEVICES= ./build/bin/llama-simple -m ...Q2_K.gguf -n 32 -ngl 0 "Write one short sentence about llamas."` completed but still produced degenerate output (`0x1000000...`). This confirms the degenerate generation is not specific to Q8_0; it is a graph/parity issue or checkpoint/export issue that survives Q2_K.
- [x] CUDA partial-offload smoke on Q2_K: `CUDA_VISIBLE_DEVICES=0,1 ./build/bin/llama-simple -m ...Q2_K.gguf -n 1 -ngl 8 hello` passed. Current box had other Python GPU workloads using ~16 GiB per 5090, so full offload initially failed with CUDA0 OOM; partial offload placed 8/34 layers on 2x5090 and decoded successfully. Log: `tasks/logs/ling26_q2k_cuda_2x5090_ngl8_decode_n1.log`.
- [x] CUDA full-offload Q2_K VRAM smoke after GPUs freed: `CUDA_VISIBLE_DEVICES=0,1 ./build/bin/llama-simple -m ...Q2_K.gguf -n 1 -ngl 99 hello` passed on 2x RTX 5090. Offloaded 34/34 layers; model buffers were CUDA0 18,488.73 MiB and CUDA1 19,063.05 MiB; eval ~81.66 ms/token for the 1-token smoke. Log: `tasks/logs/ling26_q2k_cuda_2x5090_full_decode_n1.log`.
- [x] CUDA full-offload Q2_K short completion before the QKV-stride fix: `CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_DISABLE_GRAPHS=1 ./build/bin/llama-simple -m ...Q2_K.gguf -n 16 -ngl 99 "Write one short sentence about llamas."` passed at ~24.8 tok/s decode, but output was poor (`cess` plus blanks). Log: `tasks/logs/ling26_q2k_cuda_2x5090_full_completion16_no_graphs.log`.
- [x] CUDA full-offload Q2_K coherent-output validation after the QKV-stride fix: `CUDA_VISIBLE_DEVICES=0,1 ./build/bin/llama-completion -m ...Q2_K.gguf -ngl 99 -fa auto -n 64 -p "Write one short sentence about llamas." --temp 0 --top-k 1 -st --jinja --no-warmup` produced `Llamas are gentle, social herd animals native to the Andes.` at ~24 tok/s decode. Log: `tasks/logs/ling26_qkv_stride_fix_q2k_chat_sentence.log`.
- [x] CPU Q8_0 coherent-output validation after the QKV-stride fix: `CUDA_VISIBLE_DEVICES= ./build/bin/llama-completion -m ...Q8_0.gguf -ngl 0 -fa auto -n 32 -p "Write one short sentence about llamas." --temp 0 --top-k 1 -st --jinja --no-warmup` produced `Llamas are gentle, social herd animals known for their distinctive banana-shaped ears and soft, woolly coats.` Log: `tasks/logs/ling26_qkv_stride_fix_q8_cpu_chat_sentence.log`.
- [ ] Run perplexity on wiki.test.raw at Q4_K_M; flag if >5% degradation vs F16.

### 6.5 CUDA performance pass
- [x] Root bottleneck identified: `GGML_OP_SIMPLE_GLA_SCAN` was CPU-only, forcing GPU↔CPU splits in every recurrent layer. Baseline coherent Q2_K 2x5090 full-offload run had `CUDA_Host compute buffer size = 3954.04 MiB`, `graph splits = 61`, and decode `25.00 tok/s` (`40.00 ms/token`). Log: `tasks/logs/ling26_qkv_stride_fix_q2k_long_pack_animals.log`.
- [x] Added CUDA support for `ggml_simple_gla_scan`. Corrected CUDA state load/store after backend-op parity exposed a transposed `[D_k, D_v]` layout bug.
- [x] Correct CUDA run on Q2_K 2x5090 full offload with `-c 4096`: `CUDA_Host compute buffer size = 48.04 MiB`, `graph splits = 46`, coherent output, decode `34.32 tok/s` (`29.13 ms/token`). Log: `tasks/logs/ling26_simple_gla_cuda_correct_q2k_long_ctx4096.log`.
- [x] `llama-bench` after CUDA simple-GLA: `pp64 641.10 ± 15.19 t/s`, `tg128 34.24 ± 0.02 t/s` on 2x RTX 5090, Q2_K full offload. Log: `tasks/logs/ling26_simple_gla_cuda_vec_q2k_bench_p64n128.log`.
- [x] Placement experiments: 3-GPU layer split was slower (~24.5 tok/s because the 3090 adds pipeline latency); 2-GPU row split was slower (~23.4 tok/s); tensor split is not implemented for `bailingmoe2.5`; `--no-host` and `-mg 1` did not improve decode.
- [ ] Further speed work: single-stream decode remains MoE/matmul-launch bound at batch=1; likely follow-ups are MTP/NEXTN speculative decoding and/or deeper MoE kernel/fusion work, not more CPU fallback removal.
- [x] Tensor-split enablement for `bailingmoe2.5` conservative smoke path:
    - [x] Remove the architecture-level `-sm tensor` block for `LLM_ARCH_BAILINGMOE2_5`.
    - [x] Keep BailingMoe2.5 attention/KV/recurrent state tensors mirrored for correctness; tensor split still applies to FFN/MoE/output paths. Attempts to split MLA KV-A before RMSNorm and recurrent state copies exposed invalid split-state/shape interactions, so deeper attention splitting remains follow-up work.
    - [x] Add meta-backend support for fully mirrored Flash Attention inputs and for non-meta no-op view nodes inside meta graphs.
    - [x] Build quality gate: `cmake --build build --target llama-completion -j 4` passed.
    - [x] CUDA tensor-split smoke: `CUDA_VISIBLE_DEVICES=0,1 ./build/bin/llama-completion -m /home/bigkahuna/models/gguf/Ling-2.6-flash-fp8-Q2_K.gguf -ngl 99 -sm tensor -fa on -c 512 -n 1 -p 'hello' --temp 0 --top-k 1 -st --no-warmup` passed, output `Hello`. Log: `tasks/logs/ling26_tensor_split_cuda_n1_ctx512_simplified.log`.
    - [x] CUDA tensor-split short quality smoke: same 2x5090 setup, `-n 24`, prompt `Write one short sentence about llamas.`, produced coherent output: `Llamas are gentle, social herd animals from the Andes known for their soft wool and curious expressions.` Log: `tasks/logs/ling26_tensor_split_cuda_short_sentence_ctx512.log`.

### 6.6 Documentation
- [ ] Add Ling-2.6-flash to `README.md` supported-models table.
- [ ] Add a short note in `docs/development/HOWTO-add-model.md` (or wherever Kimi-Linear's notes live) explaining the slope-baking convention, in case future Lightning-Attention models reuse the op.

---

## Phase 7: PR mechanics
- [ ] Open a feature-request issue on `ggml-org/llama.cpp` first (none exists for V2.5) to allow upstream review of the new op signature before deep implementation.
- [ ] Split into two PRs if reviewer requests: (1) `ggml_simple_gla_scan` op + tests, (2) BailingMoeV2.5 model integration consuming the op.
- [ ] Mark GPU backends as TODO in PR description if shipping CPU-only; track in a follow-up issue.

## Phase 8: Long-context (YARN) and MTP integration (follow-up PRs)

These are **deliberately deferred** out of the V1 PR to keep its scope reviewable. Capture them as separate follow-ups.

### 8.1 YARN at runtime (not at convert)
- HF config has `rope_scaling: null` and `max_position_embeddings: 131072` (the trained context).
- SGLang README recommends overriding at launch with `rope_scaling = {rope_type: "yarn", factor: 2.0, original_max_position_embeddings: 131072}` for 262144 effective context.
- **Decision**: do NOT bake YARN into the converter. Document in README that long-context (>131k) requires user to enable YARN via llama-cli flags or by patching GGUF metadata. Verify llama.cpp's existing YARN support handles this configuration unchanged.
- See `review-findings.md` §4.7.

### 8.2 MTP / NEXTN speculative decoding
- V1 PR loads MTP weights and graph-builds the MTP layer, but does NOT hook into the speculative-decoding API. The `mtp_logits` output is computed and discarded (matches the existing DeepSeek-V3 / GLM-4.5 pattern).
- Follow-up PR: integrate MTP as draft-token source in llama.cpp's speculative decoding API. Reference: SGLang `--speculative-algorithm NEXTN` / `--speculative-num-steps 3` / `--speculative-num-draft-tokens 4`. EAGLE-3 / NEXTN style.

### 8.3 `fused_qkv_a_proj_with_mqa` MLA optimization
- SGLang fuses `q_a_proj + kv_a_proj_with_mqa` into one matmul (`fused_qkv_a_proj_with_mqa`). For Ling-2.6-flash this is `[hidden=4096] -> [q_lora_rank + kv_lora_rank + qk_rope_head_dim = 1536 + 512 + 64 = 2112]`.
- V1 keeps them separate. Profile and decide whether the fusion is worth a follow-up converter + graph change.