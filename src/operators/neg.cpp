// neg: 逐元素一元运算()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_neg_operator;

static void neg_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void neg_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::eltwise(in, out, (int)dnnl_eltwise_linear, -1.f, 0.f)) {
        neg_ref(c, in, out, o);   // 缺 DLL 回落(Eigen 快速路径)
    }
}


static void neg_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return -x; });
}

static void neg_grad(BackwardContext& c) {
    unary_grad(c, UnaryBwd::Neg);
}

const Operator g_neg_operator = {
    "neg",
    infer_same,
    neg_grad,
    neg_ref,
    {neg_cpu, ZNN_CUDA_K(neg_cuda), ZNN_METAL_K(neg_metal), ZNN_VULKAN_K(neg_vulkan)},
};

Tensor Tensor::neg(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_neg_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
