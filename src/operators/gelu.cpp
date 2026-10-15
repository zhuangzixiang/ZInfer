// gelu: GELU 激活(tanh 近似)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include <cmath>

namespace znn {

extern const Operator g_gelu_operator;

static float gelu_fwd(float x) {
    const float k = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(k));
}

static float gelu_deriv(float x) {
    const float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    const float t = std::tanh(inner);
    const float dt = (1.0f - t * t) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * dt;
}

static void gelu_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel(in, out, gelu_fwd);
}

static void gelu_grad(BackwardContext& c) {
    unary_grad(c, [](float x, float) { return gelu_deriv(x); });
}

const Operator g_gelu_operator = {
    "gelu",
    infer_same,
    gelu_grad,
    gelu_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Activator::gelu(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_gelu_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
