// exp: 逐元素一元运算()
#include "shared.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_exp_operator;

static void exp_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel(in, out, [](float x) { return expf(x); });
}

static void exp_grad(BackwardContext& c) {
    unary_grad(c, [](float x, float y) { return y; });
}

const Operator g_exp_operator = {
    "exp",
    infer_same,
    exp_grad,
    exp_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::exp(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_exp_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
