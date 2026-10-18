// linear: y = x @ w.T + b —— 完整算子(融合内核 + 独立梯度);
// 不再组合 matmul+add(省一次中间分配和一次内核启动)
#include "shared.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_linear_operator;

struct LinearOptions {
    bool has_bias;
};

static void linear_infer(Span<const TensorMeta*> in, const void*,
                         std::vector<Shape>& outs) {
    const Shape& x = *in[0]->shape;                       // [N, in]
    const Shape& w = *in[1]->shape;                       // [out, in]
    if (x.rank != 2 || w.rank != 2 || x[1] != w[1]) {
        throw std::runtime_error("znn: linear expects x[N,in] w[out,in]");
    }
    outs.push_back(Shape{x[0], w[0]});
}

// ---- CPU 参考内核(朴素; Eigen 融合版填 kernels[CPU]) ----
void linear_naive(KernelContext&, Span<const Tensor*> in,
                  Span<Tensor*> out, const void* options) {
    const bool has_bias = static_cast<const LinearOptions*>(options)->has_bias;
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    const Tensor* b = has_bias ? in[2] : nullptr;
    Tensor& y = *out[0];
    const int64_t N = x.size(0), IN = x.size(1), OUT = w.size(0);
    const float* xp = x.data<float>();
    const float* wp = w.data<float>();
    const float* bp = b ? b->data<float>() : nullptr;
    float* yp = y.data<float>();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t o = 0; o < OUT; ++o) {
            float acc = 0.f;
            for (int64_t k = 0; k < IN; ++k) {
                acc += xp[n * IN + k] * wp[o * IN + k];   // w[o, k] (w 不转置)
            }
            yp[n * OUT + o] = acc + (bp ? bp[o] : 0.f);
        }
    }
}

static void linear_ref(KernelContext& c, Span<const Tensor*> in,
                       Span<Tensor*> out, const void* o) {
    linear_naive(c, in, out, o);
}

// ---- 梯度: dx = g @ w, dw = g.T @ x, db = sum(g, dim0) ----
static void linear_grad(BackwardContext& c) {
    const LinearOptions* o = static_cast<const LinearOptions*>(c.options);
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& w = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    const int64_t IN = x.size(1);
    // dx = g @ w  [N,OUT] @ [OUT,IN] → [N,IN]
    *c.grad_inputs[0] = Tensor::matmul(g, w);
    // dw = g.T @ x  [OUT,N] @ [N,IN] → [OUT,IN]
    *c.grad_inputs[1] = Tensor::matmul(g.transpose(0, 1), x);
    if (o->has_bias) {
        // db = sum(g, dim=0)  [N,OUT] → [OUT]
        Tensor g2 = g;                                     // 反向期间视图安全
        *c.grad_inputs[2] = Tensor::sum(
            g2.reshape(Shape{g.size(0), g.size(1)}), 0);
    }
}

KernelFunction linear_cpu_eigen();   // linear_eigen.cpp(融合 GEMM+bias epilogue)

const Operator g_linear_operator = {
    "linear",
    linear_infer,
    linear_grad,
    linear_ref,
    {linear_cpu_eigen(), cuda_backend::linear_cuda, nullptr, vulkan_backend::linear_vulkan},
    sizeof(LinearOptions),
};

Tensor Tensor::linear(const Tensor& x, const Tensor& w, const Tensor* b) {
    const LinearOptions o{b != nullptr};
    const Tensor* ins[3] = {&x, &w, b};
    return OpDispatcher::instance().run(&g_linear_operator,
                                        Span<const Tensor*>(ins, b ? 3 : 2), &o);
}

} // namespace znn
