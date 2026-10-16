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
    const double eps = static_cast<const LayerNormOptions*>(c.options)->eps;
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& w = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    Tensor& dx = *c.grad_inputs[0];
    Tensor& dw = *c.grad_inputs[1];
    Tensor& db = *c.grad_inputs[2];
    const int64_t D = x.size(-1);
    const int64_t rows = x.numel() / D;
    const float* xp = x.data<float>();
    const float* wp = w.data<float>();
    const float* gp = g.data<float>();
    float* dxp = dx.data<float>();
    float* dwp = dw.data<float>();
    float* dbp = db.data<float>();
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = xp + r * D;
        const float* gr = gp + r * D;
        float* dr = dxp + r * D;
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
        float m1 = 0.f;                                    // mean(g·γ·x̂)
        float m2 = 0.f;                                    // mean(g·γ)
        for (int64_t d = 0; d < D; ++d) {
            const float xh = (xr[d] - mu) * inv;
            const float gy = gr[d] * wp[d];
            m1 += gy * xh;
            m2 += gy;
            dwp[d] += gr[d] * xh;
            dbp[d] += gr[d];
        }
        m1 /= static_cast<float>(D);
        m2 /= static_cast<float>(D);
        for (int64_t d = 0; d < D; ++d) {
            const float xh = (xr[d] - mu) * inv;
            dr[d] = (gr[d] * wp[d] - m2 - xh * m1) * inv;
        }
    }
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
