// tanh: 激活函数()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"

namespace znn {

extern const Operator g_tanh_operator;

static void tanh_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void tanh_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::eltwise(in, out, (int)dnnl_eltwise_tanh, 0.f, 0.f)) {
        tanh_ref(c, in, out, o);   // 缺 DLL 回落(Eigen 快速路径)
    }
}


static void tanh_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return tanhf(x); });
}

static void tanh_grad(BackwardContext& c) {
    unary_grad(c, UnaryBwd::Tanh);
}

const Operator g_tanh_operator = {
    "tanh",
    infer_same,
    tanh_grad,
    tanh_ref,
    {tanh_cpu, nullptr, nullptr, vulkan_backend::tanh_vulkan},
};

Tensor Activator::tanh(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_tanh_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
