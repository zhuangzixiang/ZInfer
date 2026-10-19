// sub: 逐元素二元运算(输入间自动广播)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_sub_operator;

static void sub_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void sub_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::binary(in, out, (int)dnnl_binary_sub)) {
        sub_ref(c, in, out, o);
    }
}


static void sub_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    binary_kernel_fast(in, out, [](float a, float b) { return a - b; });
}

const Operator g_sub_operator = {
    "sub",
    infer_broadcast,
    binary_grad,
    sub_ref,
    {sub_cpu, nullptr, nullptr, ZNN_VULKAN_K(sub_vulkan)},
};

Tensor Tensor::sub(const Tensor& a, const Tensor& b) {
    const Tensor* ins[2] = {&a, &b};
    return OpDispatcher::instance().run(&g_sub_operator, Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
