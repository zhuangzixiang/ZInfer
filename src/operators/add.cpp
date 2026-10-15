// add: 逐元素二元运算(输入间自动广播)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"

namespace znn {

extern const Operator g_add_operator;

static void add_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    binary_kernel(in, out, [](float a, float b) { return a + b; });
}

const Operator g_add_operator = {
    "add",
    infer_broadcast,
    binary_grad,
    add_ref,
    {nullptr, cuda_backend::add_cuda, nullptr, nullptr},
};

Tensor Tensor::add(const Tensor& a, const Tensor& b) {
    const Tensor* ins[2] = {&a, &b};
    return OpDispatcher::instance().run(&g_add_operator, Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
