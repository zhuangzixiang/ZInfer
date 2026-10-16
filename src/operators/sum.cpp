// sum: 沿任意 dim 求和(ReduceAttrs 选项经节点副本传入梯度规则)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include <cstring>

namespace znn {

extern const Operator g_sum_operator;

static void sum_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void* options) {
    const ReduceAttrs* a = static_cast<const ReduceAttrs*>(options);
    const Shape& s = in[0]->shape();
    const int64_t dim = a->dim < 0 ? a->dim + s.rank : a->dim;
    std::memset(out[0]->data<float>(), 0, static_cast<size_t>(out[0]->numel()) * sizeof(float));
    sum_into(*in[0], dim, a->keepdim, *out[0]);
}

static void sum_grad(BackwardContext& c) {
    const ReduceAttrs* a = static_cast<const ReduceAttrs*>(c.options);
    const Tensor& g = *c.grad_outputs[0];
    const Tensor& x = *c.saved_inputs[0];
    const int64_t dim = a->dim < 0 ? a->dim + x.dim_count() : a->dim;
    sum_grad_into(g, dim, a->keepdim, x.shape(), *c.grad_inputs[0]);
}

const Operator g_sum_operator = {
    "sum",
    infer_reduce,
    sum_grad,
    sum_ref,
    {nullptr, cuda_backend::sum_cuda, nullptr, nullptr},
    sizeof(ReduceAttrs),
};

Tensor Tensor::sum(const Tensor& x, int64_t dim, bool keepdim) {
    const ReduceAttrs attrs{dim, keepdim};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_sum_operator, Span<const Tensor*>(ins, 1), &attrs);
}

} // namespace znn
