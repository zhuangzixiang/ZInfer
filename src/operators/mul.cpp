// mul: 逐元素二元运算(输入间自动广播)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"

namespace znn {

extern const Operator g_mul_operator;

static void mul_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void mul_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::binary(in, out, (int)dnnl_binary_mul)) {
        mul_ref(c, in, out, o);
    }
}


static void mul_grad(BackwardContext& c) {
    const Tensor& g = *c.grad_outputs[0];
    Tensor ga = Tensor::mul(g, *c.saved_inputs[1]);
    Tensor gb = Tensor::mul(g, *c.saved_inputs[0]);
    *c.grad_inputs[0] = BackwardContext::reduce_to(ga, c.saved_inputs[0]->shape());
    *c.grad_inputs[1] = BackwardContext::reduce_to(gb, c.saved_inputs[1]->shape());
}

static void mul_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    binary_kernel_fast(in, out, [](float a, float b) { return a * b; });
}

const Operator g_mul_operator = {
    "mul",
    infer_broadcast,
    mul_grad,
    mul_ref,
    {mul_cpu, nullptr, nullptr, vulkan_backend::mul_vulkan},
};

Tensor Tensor::mul(const Tensor& a, const Tensor& b) {
    const Tensor* ins[2] = {&a, &b};
    return OpDispatcher::instance().run(&g_mul_operator, Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
