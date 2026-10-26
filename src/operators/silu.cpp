// silu: 激活函数()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_silu_operator;

static void silu_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void silu_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::eltwise(in, out, (int)dnnl_eltwise_swish, 1.f, 0.f)) {
        silu_ref(c, in, out, o);   // 缺 DLL 回落(Eigen 快速路径)
    }
}


static void silu_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return x / (1.0f + expf(-x)); });
}

static void silu_grad(BackwardContext& c) {
    unary_grad(c, UnaryBwd::Silu);
}

const Operator g_silu_operator = {
    "silu",
    infer_same,
    silu_grad,
    silu_ref,
    {silu_cpu, ZNN_CUDA_K(silu_cuda), ZNN_METAL_K(silu_metal), ZNN_VULKAN_K(silu_vulkan)},
};

Tensor Activator::silu(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_silu_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
