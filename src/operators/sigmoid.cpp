// sigmoid: 激活函数()
#include "shared.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_sigmoid_operator;

static void sigmoid_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel(in, out, [](float x) { return 1.0f / (1.0f + expf(-x)); });
}

static void sigmoid_grad(BackwardContext& c) {
    unary_grad(c, [](float x, float y) { return y * (1.0f - y); });
}

const Operator g_sigmoid_operator = {
    "sigmoid",
    infer_same,
    sigmoid_grad,
    sigmoid_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Activator::sigmoid(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_sigmoid_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
