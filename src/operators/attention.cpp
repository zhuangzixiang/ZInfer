// attention: scaled_dot_product_attention(q,k,v: [B,H,L,D]);
// S = QK^T/√D → softmax → O = PV, 梯度按标准公式逐步回传
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include <cmath>
#include <vector>

namespace znn {

extern const Operator g_attention_operator;

static void attention_infer(Span<const TensorMeta*> in, const void*,
                            std::vector<Shape>& outs) {
    for (size_t i = 1; i < in.size(); ++i) {
        if (!(*in[i]->shape == *in[0]->shape)) {
            throw std::runtime_error("znn: attention expects same-shape q/k/v");
        }
    }
    outs.push_back(*in[0]->shape);
}

// 单头小工具: C = A[M,K] @ B[K,N](朴素)
static void gemm(const float* a, const float* b, float* c,
                 int64_t M, int64_t K, int64_t N, float scale, bool at) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int64_t k = 0; k < K; ++k) {
                acc += at ? a[k * M + m] * b[k * N + n] : a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = acc * scale;
        }
    }
}

static void attention_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                          const void*) {
    const Tensor& q = *in[0];
    const Tensor& k = *in[1];
    const Tensor& v = *in[2];
    Tensor& y = *out[0];
    const int64_t B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    std::vector<float> s(static_cast<size_t>(L * L));
    std::vector<float> p(static_cast<size_t>(L * L));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const float* qp = q.data<float>() + ((b * H + h) * L) * D;
            const float* kp = k.data<float>() + ((b * H + h) * L) * D;
            const float* vp = v.data<float>() + ((b * H + h) * L) * D;
            float* op = y.data<float>() + ((b * H + h) * L) * D;
            gemm(qp, kp, s.data(), L, D, L, scale, false);       // S = Q @ K^T
            for (int64_t i = 0; i < L; ++i) {                    // 行 softmax(数值稳定)
                float m = s[i * L];
                for (int64_t j = 1; j < L; ++j) m = m < s[i * L + j] ? s[i * L + j] : m;
                float z = 0.f;
                for (int64_t j = 0; j < L; ++j) {
                    p[i * L + j] = expf(s[i * L + j] - m);
                    z += p[i * L + j];
                }
                for (int64_t j = 0; j < L; ++j) p[i * L + j] /= z;
            }
            gemm(p.data(), vp, op, L, L, D, 1.f, false);         // O = P @ V
        }
    }
}

static void attention_grad(BackwardContext& c) {
    const Tensor& q = *c.saved_inputs[0];
    const Tensor& k = *c.saved_inputs[1];
    const Tensor& v = *c.saved_inputs[2];
    const Tensor& g = *c.grad_outputs[0];
    Tensor& dq = *c.grad_inputs[0];
    Tensor& dk = *c.grad_inputs[1];
    Tensor& dv = *c.grad_inputs[2];
    const int64_t B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    std::vector<float> s(static_cast<size_t>(L * L));
    std::vector<float> p(static_cast<size_t>(L * L));
    std::vector<float> dp(static_cast<size_t>(L * L));
    std::vector<float> ds(static_cast<size_t>(L * L));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const float* qp = q.data<float>() + ((b * H + h) * L) * D;
            const float* kp = k.data<float>() + ((b * H + h) * L) * D;
            const float* vp = v.data<float>() + ((b * H + h) * L) * D;
            const float* gp = g.data<float>() + ((b * H + h) * L) * D;
            float* dqp = dq.data<float>() + ((b * H + h) * L) * D;
            float* dkp = dk.data<float>() + ((b * H + h) * L) * D;
            float* dvp = dv.data<float>() + ((b * H + h) * L) * D;
            gemm(qp, kp, s.data(), L, D, L, scale, false);       // 重算前向 S/P
            for (int64_t i = 0; i < L; ++i) {
                float m = s[i * L];
                for (int64_t j = 1; j < L; ++j) m = m < s[i * L + j] ? s[i * L + j] : m;
                float z = 0.f;
                for (int64_t j = 0; j < L; ++j) {
                    p[i * L + j] = expf(s[i * L + j] - m);
                    z += p[i * L + j];
                }
                for (int64_t j = 0; j < L; ++j) p[i * L + j] /= z;
            }
            gemm(p.data(), gp, dvp, L, L, D, 1.f, true);         // dV = P^T @ dO
            gemm(gp, vp, dp.data(), L, D, L, 1.f, true);         // dP = dO @ V^T
            for (int64_t i = 0; i < L; ++i) {                    // softmax 反传
                float dot = 0.f;
                for (int64_t j = 0; j < L; ++j) dot += dp[i * L + j] * p[i * L + j];
                for (int64_t j = 0; j < L; ++j) {
                    ds[i * L + j] = p[i * L + j] * (dp[i * L + j] - dot);
                }
            }
            gemm(ds.data(), kp, dqp, L, L, D, scale, false);     // dQ = dS @ K ·scale
            gemm(ds.data(), qp, dkp, L, L, D, scale, true);      // dK = dS^T @ Q·scale
        }
    }
}

const Operator g_attention_operator = {
    "attention",
    attention_infer,
    attention_grad,
    attention_ref,
    {},                          // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal) {
    if (causal) {
        throw std::runtime_error("znn: causal attention pending v1.1");
    }
    const Tensor* ins[3] = {&q, &k, &v};
    return OpDispatcher::instance().run(&g_attention_operator,
                                        Span<const Tensor*>(ins, 3), nullptr);
}

} // namespace znn
