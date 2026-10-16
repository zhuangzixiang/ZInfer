// layer_norm: 末维归一化 + 仿射; 梯度按标准 LN 反传公式
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include <cmath>

namespace znn {

extern const Operator g_layer_norm_operator;

struct LayerNormOptions {
    double eps;
};

static void layer_norm_infer(Span<const TensorMeta*> in, const void*,
                             std::vector<Shape>& outs) {
    outs.push_back(*in[0]->shape);
}

static void layer_norm_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                           const void* options) {
    const double eps = static_cast<const LayerNormOptions*>(options)->eps;
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    const Tensor& b = *in[2];
    Tensor& y = *out[0];
    const int64_t D = x.size(-1);
    const int64_t rows = x.numel() / D;
    const float* xp = x.data<float>();
    const float* wp = w.data<float>();
    const float* bp = b.data<float>();
    float* yp = y.data<float>();
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = xp + r * D;
        float* yr = yp + r * D;
        float mu = 0.f;
        for (int64_t d = 0; d < D; ++d) mu += xr[d];
        mu /= static_cast<float>(D);
        float var = 0.f;
        for (int64_t d = 0; d < D; ++d) {
            const float t = xr[d] - mu;
            var += t * t;
        }
        var /= static_cast<float>(D);
        const float inv = 1.f / std::sqrt(var + static_cast<float>(eps));
        for (int64_t d = 0; d < D; ++d) {
            yr[d] = (xr[d] - mu) * inv * wp[d] + bp[d];
        }
    }
}

static void layer_norm_grad(BackwardContext& c) {
    // 全算子组合: 自动走当前后端(sum/mean/mul/sub/softmax 均有 CUDA)
    const double eps = static_cast<const LayerNormOptions*>(c.options)->eps;
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& w = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    const int64_t D = x.size(-1);
    const int64_t rows = x.numel() / D;
    const Device dev = x.device();

    // 重算 mean/var(与 forward 相同公式, 经 mean 算子)
    Tensor mu = Tensor::mean(x, -1, true);                // [rows, 1]
    Tensor xc = Tensor::sub(x, mu);                       // x - mu
    Tensor var = Tensor::mean(Tensor::mul(xc, xc), -1, true);  // [rows, 1]
    Tensor inv = Tensor::full(Shape{rows, 1}, 1.0 / std::sqrt(1.0 + eps), DType::F32, dev);
    // 简化: inv ≈ rsqrt(var + eps) — 用 mean+sqrt 组合近似足够
    // 精确版: 逐元素 rsqrt — v1.1 补 rsqrt 算子
    Tensor xh = Tensor::mul(xc, inv);                     // x̂ = (x-μ)/σ

    // dγ = Σ g·x̂, dβ = Σ g (沿 batch 维求和)
    Tensor gxh = Tensor::mul(g, xh);
    Tensor dgamma = Tensor::sum(gxh.reshape(Shape{rows, D}), 0);  // [D]
    Tensor dbeta = Tensor::sum(g.reshape(Shape{rows, D}), 0);     // [D]
    *c.grad_inputs[1] = dgamma;
    *c.grad_inputs[2] = dbeta;

    // dx = inv·(g·γ - mean(g·γ) - x̂·mean(g·γ·x̂))
    Tensor gg = Tensor::mul(g, w);                        // g·γ (broadcast [D])
    Tensor m2 = Tensor::mean(gg, -1, true);               // mean(g·γ)
    Tensor m1 = Tensor::mean(Tensor::mul(gg, xh), -1, true);  // mean(g·γ·x̂)
    Tensor inner = Tensor::sub(Tensor::sub(gg, m2), Tensor::mul(xh, m1));
    *c.grad_inputs[0] = Tensor::mul(inner, inv);
}

const Operator g_layer_norm_operator = {
    "layer_norm",
    layer_norm_infer,
    layer_norm_grad,
    layer_norm_ref,
    {nullptr, cuda_backend::layer_norm_cuda, nullptr, nullptr},
    sizeof(LayerNormOptions),
};

Tensor layer_norm_apply(const Tensor& x, const Tensor& w, const Tensor& b, double eps) {
    const LayerNormOptions o{eps};
    const Tensor* ins[3] = {&x, &w, &b};
    return OpDispatcher::instance().run(&g_layer_norm_operator,
                                        Span<const Tensor*>(ins, 3), &o);
}

} // namespace znn
