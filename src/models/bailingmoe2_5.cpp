#include "models.h"

#include "llama-memory-recurrent.h"

llm_build_bailingmoe2_5::llm_build_bailingmoe2_5(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_head = hparams.n_head();
    const int64_t head_dim = hparams.n_embd / n_head; // linear-attn head_dim, independent of absorbed MLA cache width
    const int64_t d_inner = n_head * head_dim;

    GGML_ASSERT(head_dim == hparams.n_embd_head_v());
    GGML_ASSERT(hparams.is_mla());

    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t q_lora_rank = hparams.n_lora_q;
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const int64_t n_norm_groups = hparams.n_norm_groups;
    const int64_t group_size = d_inner / n_norm_groups;

    GGML_ASSERT(n_norm_groups > 0);
    GGML_ASSERT(d_inner % n_norm_groups == 0);
    GGML_ASSERT(q_lora_rank > 0);
    GGML_ASSERT(kv_lora_rank > 0);

    const float kq_scale_mla = 1.0f / sqrtf((float) n_embd_head_k_mla);
    const float q_scale_gla = 1.0f / sqrtf((float) head_dim);

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_k = build_inp_mem_hybrid_k();
    auto * inp_rs = inp_k->get_recr();
    auto * inp_attn_k = inp_k->get_attn();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < n_transformer_layers; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * x_norm = cur;

        if (hparams.is_recurrent(il)) {
            // === Linear attention / Lightning-Attention-2 simple GLA ===
            ggml_tensor * qkv = ggml_mul_mat(ctx0, layer.wqkv, x_norm);
            cb(qkv, "qkv", il);
            qkv = ggml_cast(ctx0, qkv, GGML_TYPE_F32);
            cb(qkv, "qkv_f32", il);

            // HF/SGLang split qkv from rows laid out as [Q | K | V]; keep token stride at 3*d_inner.
            const size_t qkv_token_stride = ggml_row_size(qkv->type, 3 * d_inner);
            ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, head_dim, n_head, n_tokens,
                    ggml_row_size(qkv->type, head_dim),
                    qkv_token_stride,
                    0);
            ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, head_dim, n_head, n_tokens,
                    ggml_row_size(qkv->type, head_dim),
                    qkv_token_stride,
                    ggml_row_size(qkv->type, d_inner));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, head_dim, n_head, n_tokens,
                    ggml_row_size(qkv->type, head_dim),
                    qkv_token_stride,
                    ggml_row_size(qkv->type, 2 * d_inner));

            Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            // Linear-attn uses NeoX/split-half RoPE on the first n_rot dims.
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            // SGLang's seg_la kernel applies head_dim^-0.5 internally; our op keeps
            // the signature clean and expects pre-scaled q.
            Qcur = ggml_scale(ctx0, Qcur, q_scale_gla);
            cb(Qcur, "Qcur_scaled", il);

            Qcur = ggml_cont_4d(ctx0, Qcur, head_dim, n_head, n_seq_tokens, n_seqs);
            Kcur = ggml_cont_4d(ctx0, Kcur, head_dim, n_head, n_seq_tokens, n_seqs);
            Vcur = ggml_cont_4d(ctx0, Vcur, head_dim, n_head, n_seq_tokens, n_seqs);

            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();
            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            ggml_tensor * scan = ggml_simple_gla_scan(ctx0, Qcur, Kcur, Vcur, layer.attn_g_decay, state);
            cb(scan, "simple_gla_scan", il);

            ggml_tensor * output = ggml_view_4d(ctx0, scan,
                    head_dim, n_head, n_seq_tokens, n_seqs,
                    ggml_row_size(scan->type, head_dim),
                    ggml_row_size(scan->type, head_dim * n_head),
                    ggml_row_size(scan->type, head_dim * n_head * n_seq_tokens),
                    0);
            ggml_tensor * new_state = ggml_view_4d(ctx0, scan,
                    head_dim, head_dim, n_head, n_seqs,
                    ggml_row_size(scan->type, head_dim),
                    ggml_row_size(scan->type, head_dim * head_dim),
                    ggml_row_size(scan->type, head_dim * head_dim * n_head),
                    ggml_row_size(scan->type, head_dim * n_head * n_seq_tokens * n_seqs));
            cb(output, "simple_gla_output", il);
            cb(new_state, "simple_gla_new_state", il);

            ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0, new_state,
                        ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                            kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            ggml_tensor * o = ggml_cont_2d(ctx0, output, d_inner, n_tokens);

            // GroupRMSNorm: group_norm_size stores number of groups (4), so each
            // group normalizes group_size=1024 contiguous channels.
            o = ggml_reshape_3d(ctx0, o, group_size, n_norm_groups, n_tokens);
            o = ggml_rms_norm(ctx0, o, hparams.f_norm_rms_eps);
            o = ggml_reshape_2d(ctx0, o, d_inner, n_tokens);
            o = ggml_mul(ctx0, o, layer.attn_g_norm);
            cb(o, "simple_gla_group_norm", il);

            ggml_tensor * gate = ggml_mul_mat(ctx0, layer.attn_g_proj, x_norm);
            gate = ggml_sigmoid(ctx0, gate);
            o = ggml_mul(ctx0, o, gate);
            cb(o, "simple_gla_gated", il);

            o = ggml_cont_2d(ctx0, o, d_inner, n_tokens);
            cur = ggml_mul_mat(ctx0, layer.wo, o);
            cb(cur, "attn_out", il);
        } else {
            // === MLA layer (DeepSeek-V3 style with q-LoRA + absorbed KV cache) ===
            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, x_norm);
            cb(q, "q_a", il);
            q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(q, "q_a_norm", il);
            q = ggml_mul_mat(ctx0, layer.wq_b, q);
            cb(q, "q_b", il);

            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k_mla),
                    ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
                    0);
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                    ggml_row_size(q->type, n_embd_head_k_mla),
                    ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
                    ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_nope, "q_nope", il);
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, x_norm);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);
            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(kv_cmpr, "kv_cmpr", il);
            cb(k_pe, "k_pe", il);

            // MLA branch uses the interleaved/default RoPE convention.
            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NORMAL,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NORMAL,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe_rope", il);
            cb(k_pe, "k_pe_rope", il);

            kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr_norm", il);

            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
            ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
            ggml_tensor * Vcur = kv_cmpr;
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn_k,
                    layer.wo, NULL, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
            cb(cur, "mla_out", il);
        }

        if (il == n_transformer_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up, NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp, NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
