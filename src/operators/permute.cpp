// permute: 训练态维度重排走本算子(材料化 + 逆排列梯度);
// 纯推理态 Tensor::permute 直接返回零拷贝视图
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"

namespace znn {

extern const Operator g_permute_operator;

struct PermuteOptions {
    int64_t order[8];
    int64_t rank;
};

static void permute_infer(Span<const TensorMeta*> in, const void* options,
                          std::vector<Shape>& outs) {
    const PermuteOptions* o = static_cast<const PermuteOptions*>(options);
    const Shape& s = *in[0]->shape;
    Shape out;
    out.rank = s.rank;
    for (int64_t d = 0; d < s.rank; ++d) {
        out.dims[d] = s[o->order[d]];
    }
    outs.push_back(out);
}

static void permute_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* options) {
    const PermuteOptions* o = static_cast<const PermuteOptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const float* xp = x.data<float>();
    float* yp = y.data<float>();
    const int64_t r = y.dim_count();
    int64_t yidx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < y.numel(); ++i) {
        int64_t xo = 0;
        for (int64_t d = 0; d < r; ++d) {
            xo += yidx[d] * x.strides()[o->order[d]];
        }
        yp[i] = xp[xo];
        for (int64_t d = r - 1; d >= 0; --d) {
            if (++yidx[d] < y.size(d)) break;
            yidx[d] = 0;
        }
    }
}

static void permute_grad(BackwardContext& c) {
    const PermuteOptions* o = static_cast<const PermuteOptions*>(c.options);
    // 逆排列: dx[order[d]] = dy[d]
    int64_t inv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t d = 0; d < o->rank; ++d) {
        inv[o->order[d]] = d;
    }
    *c.grad_inputs[0] = c.grad_outputs[0]->permute({inv[0], inv[1], inv[2], inv[3],
                                                    inv[4], inv[5], inv[6], inv[7]});
    // 截断到实际秩(permute 收 initializer_list, 上面补零)
}

const Operator g_permute_operator = {
    "permute",
    permute_infer,
    permute_grad,
    permute_ref,
    {nullptr, cuda_backend::permute_cuda, nullptr, nullptr},
    sizeof(PermuteOptions),
};

Tensor permute_for_autograd(const Tensor& x, std::initializer_list<int64_t> dims) {
    PermuteOptions o;
    o.rank = 0;
    for (int64_t d : dims) {
        o.order[o.rank++] = d;
    }
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_permute_operator,
                                        Span<const Tensor*>(ins, 1), &o);
}

} // namespace znn
