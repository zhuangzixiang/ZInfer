// silu: 激活函数()
#include "shared.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_silu_operator;

static void silu_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel(in, out, [](float x) { return x / (1.0f + expf(-x)); });
}

static void silu_grad(BackwardContext& c) {
    unary_grad(c, [](float x, float y) { return (1.0f + expf(-x) != 0.f) ? (x / (1.0f + expf(-x)) + 1.0f / (1.0f + expf(-x)) * (1.0f - x / (1.0f + expf(-x)))) : 0.0f; });
}

const Operator g_silu_operator = {
    "silu",
    infer_same,
    silu_grad,
    silu_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Activator::silu(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_silu_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
