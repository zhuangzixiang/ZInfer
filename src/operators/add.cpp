// add: 逐元素二元运算(输入间自动广播)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_add_operator;

static void add_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void add_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::binary(in, out, (int)dnnl_binary_add)) {
        add_ref(c, in, out, o);
    }
}


static void add_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    binary_kernel_fast(in, out, [](float a, float b) { return a + b; });
}

const Operator g_add_operator = {
    "add",
    infer_broadcast,
    binary_grad,
    add_ref,
    {add_cpu, nullptr, ZNN_METAL_K(add_metal), ZNN_VULKAN_K(add_vulkan)},
};

Tensor Tensor::add(const Tensor& a, const Tensor& b) {
    const Tensor* ins[2] = {&a, &b};
    return OpDispatcher::instance().run(&g_add_operator, Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
