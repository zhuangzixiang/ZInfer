// attention: scaled_dot_product_attention(q,k,v: [B,H,L,D]);
// 前向 = CPU 朴素 / CUDA 两遍内核; 反向 = 算子组合(自动走当前后端)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"
#include <cmath>
#include <vector>

namespace znn {

extern const Operator g_attention_operator;

struct AttentionOptions {
    bool causal;               // true: 屏蔽未来位置(i < j 的注意力置 -inf)
};

static void attention_infer(Span<const TensorMeta*> in, const void*,
                            std::vector<Shape>& outs) {
    for (size_t i = 1; i < in.size(); ++i) {
        if (!(*in[i]->shape == *in[0]->shape)) {
            throw std::runtime_error("znn: attention expects same-shape q/k/v");
        }
    }
    outs.push_back(*in[0]->shape);
}

// ---- CPU 参考内核(朴素) ----
// C = scale · A × op(B): at=true 转 A, bt=true 转 B(attention 需要 × K^T)
static void gemm(const float* a, const float* b, float* c,
                 int64_t M, int64_t K, int64_t N, float scale, bool at, bool bt) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int64_t k = 0; k < K; ++k) {
                acc += (at ? a[k * M + m] : a[m * K + k])
                     * (bt ? b[n * K + k] : b[k * N + n]);
            }
            c[m * N + n] = acc * scale;
        }
    }
}

static void attention_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                          const void* options) {
    const AttentionOptions* oc = static_cast<const AttentionOptions*>(options);
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
            gemm(qp, kp, s.data(), L, D, L, scale, false, true);   // S = Q @ K^T
            if (oc->causal) {
                for (int64_t i = 0; i < L; ++i)
                    for (int64_t j = i + 1; j < L; ++j)
                        s[i * L + j] = -3.4e38f;
            }
            for (int64_t i = 0; i < L; ++i) {                    // 行 softmax
                float m = s[i * L];
                for (int64_t j = 1; j < L; ++j) m = m < s[i * L + j] ? s[i * L + j] : m;
                float z = 0.f;
                for (int64_t j = 0; j < L; ++j) {
                    p[i * L + j] = expf(s[i * L + j] - m);
                    z += p[i * L + j];
                }
                for (int64_t j = 0; j < L; ++j) p[i * L + j] /= z;
            }
            gemm(p.data(), vp, op, L, L, D, 1.f, false, false);     // O = P @ V
        }
    }
}

// ---- 反向: 全算子组合(matmul/softmax/mul/sub/sum), 自动走当前后端 ----
static void attention_grad(BackwardContext& c) {
    const Tensor& q = *c.saved_inputs[0];
    const Tensor& k = *c.saved_inputs[1];
    const Tensor& v = *c.saved_inputs[2];
    const Tensor& g = *c.grad_outputs[0];
    const int64_t B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    const Device dev = q.device();
    const int64_t BH = B * H;
    Tensor dQ = Tensor::zeros(q.shape(), DType::F32, dev);
    Tensor dK = Tensor::zeros(k.shape(), DType::F32, dev);
    Tensor dV = Tensor::zeros(v.shape(), DType::F32, dev);
    Tensor sc_t = Tensor::full(Shape{}, scale, DType::F32, dev);
    for (int64_t bh = 0; bh < BH; ++bh) {
        const int64_t b = bh / H, h = bh % H;
        Tensor qh = q.slice(0, b, b+1).slice(1, h, h+1).reshape(Shape{L, D});
        Tensor kh = k.slice(0, b, b+1).slice(1, h, h+1).reshape(Shape{L, D});
        Tensor vh = v.slice(0, b, b+1).slice(1, h, h+1).reshape(Shape{L, D});
        Tensor gh = g.slice(0, b, b+1).slice(1, h, h+1).reshape(Shape{L, D});
        Tensor S = Tensor::mul(Tensor::matmul(qh, kh.transpose(0, 1)), sc_t);
        Tensor P = Activator::softmax(S);
        Tensor dVh = Tensor::matmul(P.transpose(0, 1), gh);      // dV = P^T @ dO
        Tensor dPh = Tensor::matmul(gh, vh.transpose(0, 1));     // dP = dO @ V^T
        Tensor dot = Tensor::sum(Tensor::mul(dPh, P), 1, true);  // softmax 反传
        Tensor dSh = Tensor::mul(P, Tensor::sub(dPh, dot));
        Tensor dQh = Tensor::mul(Tensor::matmul(dSh, kh), sc_t);     // dQ = dS @ K·scale
        Tensor dKh = Tensor::mul(Tensor::matmul(dSh.transpose(0, 1), qh), sc_t);
        dQ.slice(0, b, b+1).slice(1, h, h+1).reshape(Shape{L, D}) = dQh;
        dK.slice(0, b, b+1).slice(1, h, h+1).reshape(Shape{L, D}) = dKh;
        dV.slice(0, b, b+1).slice(1, h, h+1).reshape(Shape{L, D}) = dVh;
    }
    *c.grad_inputs[0] = dQ;
    *c.grad_inputs[1] = dK;
    *c.grad_inputs[2] = dV;
}

const Operator g_attention_operator = {
    "attention",
    attention_infer,
    attention_grad,
    attention_ref,
    {nullptr, ZNN_CUDA_K(attention_cuda), nullptr, nullptr},
    sizeof(AttentionOptions),
};

Tensor Tensor::attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal) {
    const AttentionOptions o{causal};
    const Tensor* ins[3] = {&q, &k, &v};
    return OpDispatcher::instance().run(&g_attention_operator,
                                        Span<const Tensor*>(ins, 3), &o);
}

} // namespace znn
