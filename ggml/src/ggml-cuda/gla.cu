#include "common.cuh"
#include "gla.cuh"

template<int HEAD_SIZE>
static __global__ void gated_linear_attn_f32(const int B, const int T, const int C, const int H, const float scale,
     const float * k, const float * v, const float * r, const float * td, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = HEAD_SIZE;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _td[head_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        __syncthreads();

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4 & k = (float4 &)(_k[j]);
            const float4 & r = (float4 &)(_r[j]);
            const float4 & td = (float4 &)(_td[j]);
            float4 & s = (float4 &)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            s.x = s.x * td.x + kv.x;
            s.y = s.y * td.y + kv.y;
            s.z = s.z * td.z + kv.z;
            s.w = s.w * td.w + kv.w;

            y += r.x * s.x;
            y += r.y * s.y;
            y += r.z * s.z;
            y += r.w * s.w;
        }
        dst[t] = y * scale;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

template<int HEAD_SIZE>
static __global__ void simple_gla_scan_f32(const int B, const int T, const int H,
        const float * q, const float * k, const float * v, const float * g, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = HEAD_SIZE;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int C = H * head_size;
    const int state_size = head_size * head_size;

    float state[head_size];
    __shared__ float _q[head_size], _k[head_size];

    #pragma unroll
    for (int i = 0; i < head_size; ++i) {
        state[i] = s[((batch_i * H + head_i) * state_size) + tid * head_size + i];
    }

    const float decay = expf(g[head_i]);

    for (int t = 0; t < T; ++t) {
        const int offset = (batch_i * T + t) * C + head_i * head_size + tid;

        __syncthreads();
        _q[tid] = q[offset];
        _k[tid] = k[offset];
        __syncthreads();

        const float vj = v[offset];
        float y = 0.0f;
        for (int i = 0; i < head_size; i += 4) {
            const float4 kv = *reinterpret_cast<const float4 *>(&_k[i]);
            const float4 qv = *reinterpret_cast<const float4 *>(&_q[i]);
            float4 sv = *reinterpret_cast<float4 *>(&state[i]);

            sv.x = sv.x * decay + kv.x * vj;
            sv.y = sv.y * decay + kv.y * vj;
            sv.z = sv.z * decay + kv.z * vj;
            sv.w = sv.w * decay + kv.w * vj;

            y += qv.x * sv.x;
            y += qv.y * sv.y;
            y += qv.z * sv.z;
            y += qv.w * sv.w;

            *reinterpret_cast<float4 *>(&state[i]) = sv;
        }
        dst[offset] = y;
    }

    const int output_elems = B * T * C;
    #pragma unroll
    for (int i = 0; i < head_size; ++i) {
        dst[output_elems + ((batch_i * H + head_i) * state_size) + tid * head_size + i] = state[i];
    }
}

void ggml_cuda_op_simple_gla_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * g = dst->src[3];
    const ggml_tensor * s = dst->src[4];

    const int64_t D = q->ne[0];
    const int64_t H = q->ne[1];
    const int64_t T = q->ne[2];
    const int64_t B = q->ne[3];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(s->type == GGML_TYPE_F32);
    GGML_ASSERT(k->ne[0] == D && k->ne[1] == H && k->ne[2] == T && k->ne[3] == B);
    GGML_ASSERT(v->ne[0] == D && v->ne[1] == H && v->ne[2] == T && v->ne[3] == B);
    GGML_ASSERT(s->ne[0] == D && s->ne[1] == D && s->ne[2] == H && s->ne[3] == B);
    GGML_ASSERT(D == 64 || D == 128);

    const float * q_d = (const float *) q->data;
    const float * k_d = (const float *) k->data;
    const float * v_d = (const float *) v->data;
    const float * g_d = (const float *) g->data;
    const float * s_d = (const float *) s->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    if (D == 64) {
        simple_gla_scan_f32<64><<<B * H, D, 0, stream>>>(B, T, H, q_d, k_d, v_d, g_d, s_d, dst_d);
    } else {
        simple_gla_scan_f32<128><<<B * H, D, 0, stream>>>(B, T, H, q_d, k_d, v_d, g_d, s_d, dst_d);
    }
}

void ggml_cuda_op_gated_linear_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * k_d  = (const float *)dst->src[0]->data;
    const float * v_d  = (const float *)dst->src[1]->data;
    const float * r_d  = (const float *)dst->src[2]->data;
    const float * td_d = (const float *)dst->src[3]->data;
    const float * s_d  = (const float *)dst->src[4]->data;

    const int64_t B = dst->src[4]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float scale;
    memcpy(&scale, (float*)dst->op_params, sizeof(float));

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);


    if (C / H == 64) {
        gated_linear_attn_f32<64><<<B * H, C / H, 0, stream>>>(B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } else {
        gated_linear_attn_f32<128><<<B * H, C / H, 0, stream>>>(B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    }
}
