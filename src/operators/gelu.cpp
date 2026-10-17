// gelu: GELU 激活(tanh 近似)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include <cmath>

namespace znn {

extern const Operator g_gelu_operator;

static float gelu_fwd(float x) {
    const float k = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(k));
}

static void gelu_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, gelu_fwd);
}

static void gelu_grad(BackwardContext& c) {
    unary_grad(c, UnaryBwd::GELU);
}

const Operator g_gelu_operator = {
    "gelu",
    infer_same,
    gelu_grad,
    gelu_ref,
    {nullptr, cuda_backend::gelu_cuda, nullptr, nullptr},
};

Tensor Activator::gelu(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_gelu_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
