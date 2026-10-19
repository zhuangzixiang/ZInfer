// div: 逐元素除法(广播); 梯度 ga=g/b, gb=-g·a/b²(经算子组合表达)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_div_operator;

static void div_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void div_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::binary(in, out, (int)dnnl_binary_div)) {
        div_ref(c, in, out, o);
    }
}


static void div_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    binary_kernel_fast(in, out, [](float a, float b) { return a / b; });
}

static void div_grad(BackwardContext& c) {
    const Tensor& g = *c.grad_outputs[0];
    const Tensor& a = *c.saved_inputs[0];
    const Tensor& b = *c.saved_inputs[1];
    Tensor ga = Tensor::div(g, b);
    Tensor gb = Tensor::div(Tensor::neg(Tensor::mul(g, a)), Tensor::mul(b, b));
    *c.grad_inputs[0] = BackwardContext::reduce_to(ga, a.shape());
    *c.grad_inputs[1] = BackwardContext::reduce_to(gb, b.shape());
}

const Operator g_div_operator = {
    "div",
    infer_broadcast,
    div_grad,
    div_ref,
    {div_cpu, nullptr, ZNN_METAL_K(div_metal), ZNN_VULKAN_K(div_vulkan)},
};

Tensor Tensor::div(const Tensor& a, const Tensor& b) {
    const Tensor* ins[2] = {&a, &b};
    return OpDispatcher::instance().run(&g_div_operator, Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
