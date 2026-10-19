// relu: 激活函数()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_relu_operator;

static void relu_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void relu_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::eltwise(in, out, (int)dnnl_eltwise_relu, 0.f, 0.f)) {
        relu_ref(c, in, out, o);   // 缺 DLL 回落(Eigen 快速路径)
    }
}


static void relu_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return x > 0 ? x : 0; });
}

static void relu_grad(BackwardContext& c) {
    unary_grad(c, UnaryBwd::ReLU);
}

const Operator g_relu_operator = {
    "relu",
    infer_same,
    relu_grad,
    relu_ref,
    {relu_cpu, nullptr, ZNN_METAL_K(relu_metal), ZNN_VULKAN_K(relu_vulkan)},
};

Tensor Activator::relu(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_relu_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
