// tanh: 激活函数()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"

namespace znn {

extern const Operator g_tanh_operator;

static void tanh_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel(in, out, [](float x) { return tanhf(x); });
}

static void tanh_grad(BackwardContext& c) {
    unary_grad(c, [](float x, float y) { return 1.0f - y * y; });
}

const Operator g_tanh_operator = {
    "tanh",
    infer_same,
    tanh_grad,
    tanh_ref,
    {nullptr, cuda_backend::tanh_cuda, nullptr, nullptr},
};

Tensor Activator::tanh(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_tanh_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
