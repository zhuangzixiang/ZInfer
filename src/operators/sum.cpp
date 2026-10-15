// sum: 沿 dim 求和(v1 仅支持末维, 与 softmax 同款限制); 梯度 = 广播回填
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include <cstring>

namespace znn {

extern const Operator g_sum_operator;

static void sum_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void* options) {
    const bool keepdim = static_cast<const ReduceAttrs*>(options)->keepdim;
    std::memset(out[0]->data<float>(), 0, static_cast<size_t>(out[0]->numel()) * sizeof(float));
    sum_into(*in[0], in[0]->dim_count() - 1, keepdim, *out[0]);
}

static void sum_grad(BackwardContext& c) {
    const Tensor& g = *c.grad_outputs[0];
    const Tensor& x = *c.saved_inputs[0];
    sum_grad_into(g, x.dim_count() - 1, x.dim_count() == g.dim_count(), x.shape(), *c.grad_inputs[0]);
}

const Operator g_sum_operator = {
    "sum",
    infer_reduce,
    sum_grad,
    sum_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::sum(const Tensor& x, int64_t dim, bool keepdim) {
    if (dim != -1) {
        throw std::runtime_error("znn: sum supports last dim only in v1");
    }
    const ReduceAttrs attrs{dim, keepdim};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_sum_operator, Span<const Tensor*>(ins, 1), &attrs);
}

} // namespace znn
