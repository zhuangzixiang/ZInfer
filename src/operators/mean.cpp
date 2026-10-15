// mean: 沿 dim 平均(v1 仅末维) = sum/n; 梯度 = 广播回填/n
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include <cstring>

namespace znn {

extern const Operator g_mean_operator;

static void mean_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void* options) {
    const bool keepdim = static_cast<const ReduceAttrs*>(options)->keepdim;
    std::memset(out[0]->data<float>(), 0, static_cast<size_t>(out[0]->numel()) * sizeof(float));
    const Tensor& x = *in[0];
    sum_into(x, x.dim_count() - 1, keepdim, *out[0]);
    const float inv = 1.f / static_cast<float>(x.size(-1));
    float* p = out[0]->data<float>();
    for (int64_t i = 0; i < out[0]->numel(); ++i) p[i] *= inv;
}

static void mean_grad(BackwardContext& c) {
    const Tensor& g = *c.grad_outputs[0];
    const Tensor& x = *c.saved_inputs[0];
    const float inv = 1.f / static_cast<float>(x.size(-1));
    sum_grad_into(g, x.dim_count() - 1, x.dim_count() == g.dim_count(), x.shape(), *c.grad_inputs[0]);
    float* p = c.grad_inputs[0]->data<float>();
    for (int64_t i = 0; i < c.grad_inputs[0]->numel(); ++i) p[i] *= inv;
}

const Operator g_mean_operator = {
    "mean",
    infer_reduce,
    mean_grad,
    mean_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::mean(const Tensor& x, int64_t dim, bool keepdim) {
    if (dim != -1) {
        throw std::runtime_error("znn: mean supports last dim only in v1");
    }
    const ReduceAttrs attrs{dim, keepdim};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_mean_operator, Span<const Tensor*>(ins, 1), &attrs);
}

} // namespace znn
