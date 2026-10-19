// mean: 沿任意 dim 平均 = sum / n; 梯度 = 广播回填 / n
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"
#include <cstring>

namespace znn {

extern const Operator g_mean_operator;

static void mean_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void* options);

static void mean_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                         const void* o) {
    const ReduceAttrs* a = static_cast<const ReduceAttrs*>(o);
    if (!onednn::reduce(in, out, a->dim, /*mean=*/true)) {
        mean_ref(c, in, out, o);
    }
}


static void mean_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void* options) {
    const ReduceAttrs* a = static_cast<const ReduceAttrs*>(options);
    const Shape& s = in[0]->shape();
    const int64_t dim = a->dim < 0 ? a->dim + s.rank : a->dim;
    std::memset(out[0]->data<float>(), 0, static_cast<size_t>(out[0]->numel()) * sizeof(float));
    const Tensor& x = *in[0];
    sum_into(x, dim, a->keepdim, *out[0]);
    const float inv = 1.f / static_cast<float>(s[dim]);
    float* p = out[0]->data<float>();
    for (int64_t i = 0; i < out[0]->numel(); ++i) p[i] *= inv;
}

static void mean_grad(BackwardContext& c) {
    const ReduceAttrs* a = static_cast<const ReduceAttrs*>(c.options);
    const Tensor& g = *c.grad_outputs[0];
    const Tensor& x = *c.saved_inputs[0];
    const int64_t dim = a->dim < 0 ? a->dim + x.dim_count() : a->dim;
    if (g.device() != Device::CPU) {          // 原生: 广播步长 + 1/n 缩放
        const float inv = 1.f / static_cast<float>(x.size(dim));
        long long dims[8], eff[8];
        const int64_t* gs = g.strides();
        const int rank = (int)x.dim_count();
        for (int d = 0; d < rank; ++d) {
            dims[d] = x.size(d);
            const int64_t gd = (d == dim && !a->keepdim) ? -1
                                     : (a->keepdim ? d : (d > dim ? d - 1 : d));
            eff[d] = (gd < 0 || g.size(gd) == 1) ? 0 : gs[gd];
        }
        grad_bcast_dispatch(g.device(), g.data<float>(),
                            c.grad_inputs[0]->data<float>(),
                            c.grad_inputs[0]->numel(), rank, dims, eff, inv);
        return;
    }
    const float inv = 1.f / static_cast<float>(x.size(dim));
    sum_grad_into(g, dim, a->keepdim, x.shape(), *c.grad_inputs[0]);
    float* p = c.grad_inputs[0]->data<float>();
    for (int64_t i = 0; i < c.grad_inputs[0]->numel(); ++i) p[i] *= inv;
}

const Operator g_mean_operator = {
    "mean",
    infer_reduce,
    mean_grad,
    mean_ref,
    {mean_cpu, ZNN_CUDA_K(mean_cuda), ZNN_METAL_K(mean_metal), ZNN_VULKAN_K(mean_vulkan)},
    sizeof(ReduceAttrs),
};

Tensor Tensor::mean(const Tensor& x, int64_t dim, bool keepdim) {
    const ReduceAttrs attrs{dim, keepdim};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_mean_operator, Span<const Tensor*>(ins, 1), &attrs);
}

} // namespace znn
