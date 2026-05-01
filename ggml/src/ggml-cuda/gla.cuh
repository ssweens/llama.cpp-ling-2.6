#include "common.cuh"

void ggml_cuda_op_gated_linear_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_simple_gla_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
