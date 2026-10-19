// permute: 训练态维度重排走本算子(材料化 + 逆排列梯度);
// 纯推理态 Tensor::permute 直接返回零拷贝视图
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

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
    // 逆排列: 梯度直接按逆排列做材料化拷贝
    int64_t inv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t d = 0; d < o->rank; ++d) {
        inv[o->order[d]] = d;
    }
    const Tensor& g = *c.grad_outputs[0];
    Tensor& dx = *c.grad_inputs[0];
    const int64_t* gs = g.strides();
    const Shape& ds = dx.shape();
    if (g.device() == Device::CUDA) {       // 原生: 逆置换步长收集
        long long dims[8], eff[8];
        for (int64_t d = 0; d < ds.rank; ++d) {
            dims[d] = ds.dims[d];
            eff[d] = gs[o->order[d]];       // idx[p] 的步长 = g 第 order[p] 维步长
        }
        ZNN_CUDA_K(grad_bcast_cuda)(g.data<float>(), dx.data<float>(),
                                      dx.numel(), (int)ds.rank, dims, eff, 1.f);
        return;
    }
    const float* gp = g.data<float>();
    float* dp = dx.data<float>();
    // 遍历输出(=输入形状)多下标, 按逆排列算梯度源偏移
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < dx.numel(); ++i) {
        int64_t go = 0;
        for (int64_t d = 0; d < ds.rank; ++d) {
            go += idx[inv[d]] * gs[d];
        }
        dp[i] = gp[go];
        for (int64_t d = ds.rank - 1; d >= 0; --d) {
            if (++idx[d] < ds.dims[d]) break;
            idx[d] = 0;
        }
    }
}

const Operator g_permute_operator = {
    "permute",
    permute_infer,
    permute_grad,
    permute_ref,
    {nullptr, ZNN_CUDA_K(permute_cuda), nullptr, nullptr},
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
