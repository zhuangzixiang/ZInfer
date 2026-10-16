// sqrt: 逐元素一元运算()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"

namespace znn {

extern const Operator g_sqrt_operator;

static void sqrt_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return sqrtf(x); });
}

static void sqrt_grad(BackwardContext& c) {
    unary_grad(c, [](float x, float y) { return 0.5f / y; });
}

const Operator g_sqrt_operator = {
    "sqrt",
    infer_same,
    sqrt_grad,
    sqrt_ref,
    {nullptr, cuda_backend::sqrt_cuda, nullptr, nullptr},
};

Tensor Tensor::sqrt(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_sqrt_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
