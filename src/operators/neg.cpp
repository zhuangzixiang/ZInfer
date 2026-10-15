// neg: 逐元素一元运算()
#include "shared.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_neg_operator;

static void neg_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel(in, out, [](float x) { return -x; });
}

static void neg_grad(BackwardContext& c) {
    unary_grad(c, [](float x, float y) { return -1.0f; });
}

const Operator g_neg_operator = {
    "neg",
    infer_same,
    neg_grad,
    neg_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::neg(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_neg_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
