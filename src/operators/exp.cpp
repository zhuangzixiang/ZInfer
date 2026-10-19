// exp: 逐元素一元运算()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_exp_operator;

static void exp_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void exp_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::eltwise(in, out, (int)dnnl_eltwise_exp, 0.f, 0.f)) {
        exp_ref(c, in, out, o);   // 缺 DLL 回落(Eigen 快速路径)
    }
}


static void exp_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return expf(x); });
}

static void exp_grad(BackwardContext& c) {
    unary_grad(c, UnaryBwd::Exp);
}

const Operator g_exp_operator = {
    "exp",
    infer_same,
    exp_grad,
    exp_ref,
    {exp_cpu, nullptr, ZNN_METAL_K(exp_metal), ZNN_VULKAN_K(exp_vulkan)},
};

Tensor Tensor::exp(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_exp_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
