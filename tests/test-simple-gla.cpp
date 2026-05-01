#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static size_t idx4(int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t ne0, int64_t ne1, int64_t ne2) {
    return (size_t) (i0 + ne0 * (i1 + ne1 * (i2 + ne2 * i3)));
}

static std::vector<float> reference_simple_gla(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & g,
        const std::vector<float> & state,
        int64_t Dk,
        int64_t Dv,
        int64_t H,
        int64_t T,
        int64_t B) {
    const size_t output_elems = (size_t) (Dv * H * T * B);
    const size_t state_elems  = (size_t) (Dk * Dv * H * B);

    std::vector<float> packed(output_elems + state_elems, 0.0f);
    std::vector<float> S((size_t) (Dk * Dv));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t j = 0; j < Dv; ++j) {
                for (int64_t i = 0; i < Dk; ++i) {
                    S[(size_t) (i + Dk * j)] = state[idx4(i, j, h, b, Dk, Dv, H)];
                }
            }

            const float decay = std::exp(g[(size_t) h]);
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t j = 0; j < Dv; ++j) {
                    const float vj = v[idx4(j, h, t, b, Dv, H, T)];
                    for (int64_t i = 0; i < Dk; ++i) {
                        S[(size_t) (i + Dk * j)] = decay * S[(size_t) (i + Dk * j)] +
                            k[idx4(i, h, t, b, Dk, H, T)] * vj;
                    }
                }

                for (int64_t j = 0; j < Dv; ++j) {
                    float sum = 0.0f;
                    for (int64_t i = 0; i < Dk; ++i) {
                        sum += S[(size_t) (i + Dk * j)] * q[idx4(i, h, t, b, Dk, H, T)];
                    }
                    packed[idx4(j, h, t, b, Dv, H, T)] = sum;
                }
            }

            for (int64_t j = 0; j < Dv; ++j) {
                for (int64_t i = 0; i < Dk; ++i) {
                    packed[output_elems + idx4(i, j, h, b, Dk, Dv, H)] = S[(size_t) (i + Dk * j)];
                }
            }
        }
    }

    return packed;
}

static bool run_case(int64_t Dk, int64_t Dv, int64_t H, int64_t T, int64_t B) {
    const size_t q_elems     = (size_t) (Dk * H * T * B);
    const size_t v_elems     = (size_t) (Dv * H * T * B);
    const size_t state_elems = (size_t) (Dk * Dv * H * B);

    std::vector<float> q(q_elems);
    std::vector<float> k(q_elems);
    std::vector<float> v(v_elems);
    std::vector<float> g((size_t) H);
    std::vector<float> state(state_elems);

    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.01f * (float) ((int) (i % 17) - 8);
        k[i] = 0.02f * (float) ((int) (i % 13) - 6);
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = 0.015f * (float) ((int) (i % 19) - 9);
    }
    for (int64_t h = 0; h < H; ++h) {
        g[(size_t) h] = -0.05f * (float) (h + 1);
    }
    for (size_t i = 0; i < state.size(); ++i) {
        state[i] = 0.001f * (float) ((int) (i % 23) - 11);
    }

    const std::vector<float> expected = reference_simple_gla(q, k, v, g, state, Dk, Dv, H, T, B);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "failed to init ggml context\n");
        return false;
    }

    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Dk, H, T, B);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Dk, H, T, B);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Dv, H, T, B);
    ggml_tensor * tg = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Dk, Dv, H, B);
    ggml_tensor * out = ggml_simple_gla_scan(ctx, tq, tk, tv, tg, ts);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "failed to allocate backend buffer\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(tq, q.data(),     0, q.size()     * sizeof(float));
    ggml_backend_tensor_set(tk, k.data(),     0, k.size()     * sizeof(float));
    ggml_backend_tensor_set(tv, v.data(),     0, v.size()     * sizeof(float));
    ggml_backend_tensor_set(tg, g.data(),     0, g.size()     * sizeof(float));
    ggml_backend_tensor_set(ts, state.data(), 0, state.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));

    bool ok = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float tol = 1e-5f + 1e-4f * std::fabs(expected[i]);
        if (diff > tol) {
            std::fprintf(stderr,
                    "simple_gla mismatch Dk=%lld Dv=%lld H=%lld T=%lld B=%lld idx=%zu actual=%g expected=%g diff=%g tol=%g\n",
                    (long long) Dk, (long long) Dv, (long long) H, (long long) T, (long long) B,
                    i, actual[i], expected[i], diff, tol);
            ok = false;
            break;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return ok;
}

int main() {
    bool ok = true;
    ok = run_case(4, 4, 2, 1, 1) && ok;
    ok = run_case(4, 4, 2, 4, 1) && ok;
    ok = run_case(8, 4, 3, 8, 2) && ok;

    if (!ok) {
        return EXIT_FAILURE;
    }

    std::puts("OK: ggml_simple_gla_scan CPU reference tests passed");
    return EXIT_SUCCESS;
}
