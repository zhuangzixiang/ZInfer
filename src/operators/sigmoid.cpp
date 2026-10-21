// sigmoid: 激活函数()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_sigmoid_operator;

static void sigmoid_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void sigmoid_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::eltwise(in, out, (int)dnnl_eltwise_logistic, 0.f, 0.f)) {
        sigmoid_ref(c, in, out, o);   // 缺 DLL 回落(Eigen 快速路径)
    }
}


static void sigmoid_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return 1.0f / (1.0f + expf(-x)); });
}

static void sigmoid_grad(BackwardContext& c) {
    unary_grad(c, UnaryBwd::Sigmoid);
}

const Operator g_sigmoid_operator = {
    "sigmoid",
    infer_same,
    sigmoid_grad,
    sigmoid_ref,
    {sigmoid_cpu, ZNN_CUDA_K(sigmoid_cuda), ZNN_METAL_K(sigmoid_metal), ZNN_VULKAN_K(sigmoid_vulkan)},
};

Tensor Activator::sigmoid(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_sigmoid_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
