// relu: 激活函数()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"

namespace znn {

extern const Operator g_relu_operator;

static void relu_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return x > 0 ? x : 0; });
}

static void relu_grad(BackwardContext& c) {
    unary_grad(c, [](float x, float y) { return x > 0 ? 1.0f : 0.0f; });
}

const Operator g_relu_operator = {
    "relu",
    infer_same,
    relu_grad,
    relu_ref,
    {nullptr, cuda_backend::relu_cuda, nullptr, nullptr},
};

Tensor Activator::relu(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_relu_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
