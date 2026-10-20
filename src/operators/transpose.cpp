// transpose: 训练态转置走本算子(材料化拷贝 + 逆梯度 = 再转置);
// 纯推理态 Tensor::transpose 直接返回零拷贝视图, 不经此文件。
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_transpose_operator;

struct TransposeOptions {
    int64_t d0, d1;
};

static int64_t resolve(int64_t dim, int64_t rank) {
    return dim < 0 ? dim + rank : dim;
}

static void transpose_infer(Span<const TensorMeta*> in, const void* options,
                            std::vector<Shape>& outs) {
    const TransposeOptions* o = static_cast<const TransposeOptions*>(options);
    Shape s = *in[0]->shape;
    const int64_t a = resolve(o->d0, s.rank);
    const int64_t b = resolve(o->d1, s.rank);
    const int64_t t = s.dims[a];
    s.dims[a] = s.dims[b];
    s.dims[b] = t;
    outs.push_back(s);
}

static void transpose_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                          const void* options) {
    const TransposeOptions* o = static_cast<const TransposeOptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const int64_t r = y.dim_count();
    const int64_t a = resolve(o->d0, r);
    const int64_t b = resolve(o->d1, r);
    const float* xp = x.data<float>();
    float* yp = y.data<float>();
    const int64_t* xs = x.strides();
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < y.numel(); ++i) {
        int64_t so = 0;
        for (int64_t d = 0; d < r; ++d) {
            const int64_t sd = d == a ? b : (d == b ? a : d);   // 目标维 ← 源维互换
            so += idx[d] * xs[sd];
        }
        yp[i] = xp[so];
        for (int64_t d = r - 1; d >= 0; --d) {
            if (++idx[d] < y.size(d)) break;
            idx[d] = 0;
        }
    }
}

static void transpose_grad(BackwardContext& c) {
    const TransposeOptions* o = static_cast<const TransposeOptions*>(c.options);
    *c.grad_inputs[0] = c.grad_outputs[0]->transpose(o->d0, o->d1);
}

const Operator g_transpose_operator = {
    "transpose",
    transpose_infer,
    transpose_grad,
    transpose_ref,
    {nullptr, ZNN_CUDA_K(transpose_cuda), ZNN_METAL_K(transpose_metal), ZNN_VULKAN_K(transpose_vulkan)},
    sizeof(TransposeOptions),
};

Tensor transpose_for_autograd(const Tensor& x, int64_t d0, int64_t d1) {
    const TransposeOptions o{d0, d1};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_transpose_operator,
                                        Span<const Tensor*>(ins, 1), &o);
}

} // namespace znn
