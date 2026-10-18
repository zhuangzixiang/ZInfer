// sqrt: 逐元素一元运算()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"

namespace znn {

extern const Operator g_sqrt_operator;

static void sqrt_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void sqrt_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::eltwise(in, out, (int)dnnl_eltwise_sqrt, 0.f, 0.f)) {
        sqrt_ref(c, in, out, o);   // 缺 DLL 回落(Eigen 快速路径)
    }
}


static void sqrt_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return sqrtf(x); });
}

static void sqrt_grad(BackwardContext& c) {
    unary_grad(c, UnaryBwd::Sqrt);
}

const Operator g_sqrt_operator = {
    "sqrt",
    infer_same,
    sqrt_grad,
    sqrt_ref,
    {sqrt_cpu, nullptr, nullptr, vulkan_backend::sqrt_vulkan},
};

Tensor Tensor::sqrt(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_sqrt_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
