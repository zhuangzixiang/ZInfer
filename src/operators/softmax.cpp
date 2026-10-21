// softmax: 沿任意 dim 归一化指数(数值稳定); 梯度 s·(g - Σ g·s) 逐行
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_softmax_operator;

static void softmax_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void* options);


struct SoftmaxOptions {
    int64_t dim;
};

static void softmax_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                           const void* o) {
    const int64_t dim = static_cast<const SoftmaxOptions*>(o)->dim;
    if (!onednn::softmax(in, out, dim)) {
        softmax_ref(c, in, out, o);
    }
}



static void softmax_infer(Span<const TensorMeta*> in, const void*, ShapeList& outs) {
    outs.push_back(*in[0]->shape);
}

// 沿 dim 的逐行 softmax: outer 遍历除 dim 外的全部下标。
// 输入按其步长寻址(可为任意视图), 输出按自身(连续)步长写 —— 此前
// 直接拿输入步长寻址输出, 仅连续输入时碰巧正确(存量 bug)。
static void softmax_rows(const float* xp, float* yp, const Shape& shape,
                         const int64_t* xstr, int64_t dim) {
    const int64_t n = shape[dim];
    const int64_t sd = xstr[dim];
    int64_t outer = 1, os[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t acc = 1;
    for (int64_t d = shape.rank - 1; d >= 0; --d) {         // 输出行主序步长
        os[d] = acc;
        acc *= shape[d];
        if (d != dim) outer *= shape[d];
    }
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t o = 0; o < outer; ++o) {
        int64_t bx = 0, by = 0;
        for (int64_t d = 0; d < shape.rank; ++d) {
            if (d != dim) { bx += idx[d] * xstr[d]; by += idx[d] * os[d]; }
        }
        float m = xp[bx];
        for (int64_t k = 1; k < n; ++k) m = m < xp[bx + k * sd] ? xp[bx + k * sd] : m;
        float s = 0.f;
        for (int64_t k = 0; k < n; ++k) {
            yp[by + k * os[dim]] = expf(xp[bx + k * sd] - m);
            s += yp[by + k * os[dim]];
        }
        for (int64_t k = 0; k < n; ++k) yp[by + k * os[dim]] /= s;
        for (int64_t d = shape.rank - 1; d >= 0; --d) {     // 推进 outer 下标(跳过 dim)
            if (d == dim) continue;
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}

static void softmax_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* options) {
    const SoftmaxOptions* o = static_cast<const SoftmaxOptions*>(options);
    const Shape& s = in[0]->shape();
    const int64_t dim = o->dim < 0 ? o->dim + s.rank : o->dim;
    softmax_rows(in[0]->data<float>(), out[0]->data<float>(), s, in[0]->strides(), dim);
}

static void softmax_grad(BackwardContext& c) {
    // 组合实现: dx = s * (g - sum(g*s, dim, keepdim))
    // 全部经算子(mul/sum/sub), 自动走当前后端
    const SoftmaxOptions* o = static_cast<const SoftmaxOptions*>(c.options);
    const Tensor& g = *c.grad_outputs[0];
    const Tensor& y = *c.saved_outputs[0];
    Tensor gs = Tensor::mul(g, y);
    Tensor dot = Tensor::sum(gs, o->dim, true);           // keepdim 广播安全
    *c.grad_inputs[0] = Tensor::mul(y, Tensor::sub(g, dot));
}

const Operator g_softmax_operator = {
    "softmax",
    softmax_infer,
    softmax_grad,
    softmax_ref,
    {softmax_cpu, ZNN_CUDA_K(softmax_cuda), ZNN_METAL_K(softmax_metal), ZNN_VULKAN_K(softmax_vulkan)},
    sizeof(SoftmaxOptions),
};

Tensor Activator::softmax(const Tensor& x, int64_t dim) {
    const SoftmaxOptions o{dim};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_softmax_operator, Span<const Tensor*>(ins, 1), &o);
}

} // namespace znn
