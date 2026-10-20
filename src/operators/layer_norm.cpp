// layer_norm: 末维归一化 + 仿射; 梯度按标准 LN 反传公式
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"
#include <cmath>

namespace znn {

extern const Operator g_layer_norm_operator;

static void layer_norm_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void* options);


struct LayerNormOptions {
    double eps;
};

static void layer_norm_cpu(KernelContext& c, Span<const Tensor*> in,
                           Span<Tensor*> out, const void* o) {
    const LayerNormOptions* a = static_cast<const LayerNormOptions*>(o);
    if (!onednn::layer_norm(*in[0], *in[1], *in[2], *out[0], a->eps)) {
        layer_norm_ref(c, in, out, o);
    }
}



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
    // 全算子组合(自动走当前后端)。统一按 2D [rows, D] 推导(任意秩输入),
    // inv = 1/sqrt(var+eps) 精确公式(经 mean/mul/sqrt/div 组合, 免近似)
    const double eps = static_cast<const LayerNormOptions*>(c.options)->eps;
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& w = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    const int64_t D = x.size(-1);
    const int64_t rows = x.numel() / D;
    const Device dev = x.device();
    Tensor x2 = x.reshape(Shape{rows, D});
    Tensor g2 = g.reshape(Shape{rows, D});

    Tensor mu = Tensor::mean(x2, -1, true);                     // [rows,1]
    Tensor xc = Tensor::sub(x2, mu);
    Tensor var = Tensor::mean(Tensor::mul(xc, xc), -1, true);   // [rows,1]
    Tensor inv = Tensor::div(Tensor::full(Shape{rows, 1}, 1.0, DType::F32, dev),
                             Tensor::sqrt(Tensor::add(
                                 var, Tensor::full(Shape{rows, 1}, eps,
                                                   DType::F32, dev))));
    Tensor xh = Tensor::mul(xc, inv);                           // x̂ = (x-μ)/σ

    Tensor dgamma = Tensor::sum(Tensor::mul(g2, xh), 0);        // dγ = Σ g·x̂
    Tensor dbeta = Tensor::sum(g2, 0);                          // dβ = Σ g
    *c.grad_inputs[1] = dgamma;
    *c.grad_inputs[2] = dbeta;

    // dx = inv·(g·γ - mean(g·γ) - x̂·mean(g·γ·x̂))
    Tensor gg = Tensor::mul(g2, w);                             // broadcast [D]
    Tensor m2 = Tensor::mean(gg, -1, true);
    Tensor m1 = Tensor::mean(Tensor::mul(gg, xh), -1, true);
    Tensor inner = Tensor::sub(Tensor::sub(gg, m2), Tensor::mul(xh, m1));
    *c.grad_inputs[0] = Tensor::mul(inner, inv).reshape(x.shape());
}

const Operator g_layer_norm_operator = {
    "layer_norm",
    layer_norm_infer,
    layer_norm_grad,
    layer_norm_ref,
    {layer_norm_cpu, ZNN_CUDA_K(layer_norm_cuda), ZNN_METAL_K(layer_norm_metal), ZNN_VULKAN_K(layer_norm_vulkan)},
    sizeof(LayerNormOptions),
};

Tensor layer_norm_apply(const Tensor& x, const Tensor& w, const Tensor& b, double eps) {
    const LayerNormOptions o{eps};
    const Tensor* ins[3] = {&x, &w, &b};
    return OpDispatcher::instance().run(&g_layer_norm_operator,
                                        Span<const Tensor*>(ins, 3), &o);
}

} // namespace znn
