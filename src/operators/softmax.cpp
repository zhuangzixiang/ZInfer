// softmax: 沿 dim 归一化指数(v1 仅支持末维); 梯度 s·(g - Σ g·s) 逐行
#include "shared.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_softmax_operator;

static void softmax_infer(Span<const TensorMeta*> in, const void*, std::vector<Shape>& outs) {
    outs.push_back(*in[0]->shape);
}

static void softmax_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const int64_t n = x.size(-1);                          // 末维
    const int64_t rows = x.numel() / n;
    const float* xp = x.data<float>();
    float* yp = y.data<float>();
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = xp + r * n;
        float* yr = yp + r * n;
        float m = xr[0];
        for (int64_t i = 1; i < n; ++i) m = m < xr[i] ? xr[i] : m;   // 数值稳定
        float s = 0.f;
        for (int64_t i = 0; i < n; ++i) {
            yr[i] = expf(xr[i] - m);
            s += yr[i];
        }
        for (int64_t i = 0; i < n; ++i) yr[i] /= s;
    }
}

static void softmax_grad(BackwardContext& c) {
    const Tensor& g = *c.grad_outputs[0];
    const Tensor& y = *c.saved_outputs[0];
    Tensor& gi = *c.grad_inputs[0];
    const int64_t n = y.size(-1);
    const int64_t rows = y.numel() / n;
    const float* gp = g.data<float>();
    const float* yp = y.data<float>();
    float* ip = gi.data<float>();
    for (int64_t r = 0; r < rows; ++r) {
        const float* gr = gp + r * n;
        const float* yr = yp + r * n;
        float* ir = ip + r * n;
        float dot = 0.f;
        for (int64_t i = 0; i < n; ++i) dot += gr[i] * yr[i];
        for (int64_t i = 0; i < n; ++i) ir[i] = yr[i] * (gr[i] - dot);
    }
}

const Operator g_softmax_operator = {
    "softmax",
    softmax_infer,
    softmax_grad,
    softmax_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Activator::softmax(const Tensor& x, int64_t dim) {
    if (dim != -1) {
        throw std::runtime_error("znn: softmax supports last dim only in v1");
    }
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_softmax_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
