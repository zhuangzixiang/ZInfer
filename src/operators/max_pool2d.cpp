// max_pool2d: NCHW 最大池化; 梯度经"重找窗口最大位置"回填(无需保存掩码)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_max_pool2d_operator;

static void max_pool2d_ref(KernelContext&, Span<const Tensor*> in,
                           Span<Tensor*> out, const void* options);

static void max_pool2d_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                             const void* o) {
    const PoolCallOptions* a = static_cast<const PoolCallOptions*>(o);
    if (!onednn::pool2d(*in[0], *out[0], a->kernel, a->pool, /*max_pool=*/true)) {
        max_pool2d_ref(c, in, out, o);
    }
}


static void max_pool2d_infer(Span<const TensorMeta*> in, const void* options,
                             std::vector<Shape>& outs) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    const Shape& x = *in[0]->shape;                       // {N,C,H,W}
    if (x.rank != 4) {
        throw std::runtime_error("znn: max_pool2d expects NCHW");
    }
    const int64_t ho = (x[2] + 2 * o->pool.pad[0] - o->kernel[0]) / o->pool.stride[0] + 1;
    const int64_t wo = (x[3] + 2 * o->pool.pad[1] - o->kernel[1]) / o->pool.stride[1] + 1;
    outs.push_back(Shape{x[0], x[1], ho, wo});
}

static void max_pool2d_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                           const void* options) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const int64_t C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t HO = y.size(2), WO = y.size(3);
    const float* xp = x.data<float>();
    float* yp = y.data<float>();
    for (int64_t n = 0; n < x.size(0); ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* xc = xp + ((n * C + c) * H) * W;
            for (int64_t ho = 0; ho < HO; ++ho) {
                for (int64_t wo = 0; wo < WO; ++wo) {
                    float m = -3.4e38f;
                    for (int64_t ki = 0; ki < o->kernel[0]; ++ki) {
                        const int64_t i = ho * o->pool.stride[0] + ki - o->pool.pad[0];
                        if (i < 0 || i >= H) continue;
                        for (int64_t kj = 0; kj < o->kernel[1]; ++kj) {
                            const int64_t j = wo * o->pool.stride[1] + kj - o->pool.pad[1];
                            if (j < 0 || j >= W) continue;
                            const float v = xc[i * W + j];
                            m = m < v ? v : m;
                        }
                    }
                    yp[((n * C + c) * HO + ho) * WO + wo] = m;
                }
            }
        }
    }
}

static void max_pool2d_grad(BackwardContext& c) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(c.options);
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& g = *c.grad_outputs[0];
    Tensor& dx = *c.grad_inputs[0];
    const int64_t C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t HO = g.size(2), WO = g.size(3);
    if (x.device() == Device::CUDA) {       // 原生: 重找窗口最大后原子散射
        ZNN_CUDA_CALL(ZNN_CUDA_K(max_pool_grad_cuda)(
            cuda_ptr(x), cuda_ptr(g), cuda_ptr(dx),
            x.size(0) * C * HO * WO, (int)H, (int)W, (int)HO, (int)WO,
            (int)o->kernel[0], (int)o->kernel[1],
            (int)o->pool.stride[0], (int)o->pool.stride[1],
            (int)o->pool.pad[0], (int)o->pool.pad[1]));
        return;
    }
    const float* xp = x.data<float>();
    const float* gp = g.data<float>();
    float* dp = dx.data<float>();
    for (int64_t n = 0; n < x.size(0); ++n) {
        for (int64_t ch = 0; ch < C; ++ch) {
            const float* xc = xp + ((n * C + ch) * H) * W;
            for (int64_t ho = 0; ho < HO; ++ho) {
                for (int64_t wo = 0; wo < WO; ++wo) {
                    int64_t bi = -1, bj = -1;              // 重找窗口最大位置
                    float m = -3.4e38f;
                    for (int64_t ki = 0; ki < o->kernel[0]; ++ki) {
                        const int64_t i = ho * o->pool.stride[0] + ki - o->pool.pad[0];
                        if (i < 0 || i >= H) continue;
                        for (int64_t kj = 0; kj < o->kernel[1]; ++kj) {
                            const int64_t j = wo * o->pool.stride[1] + kj - o->pool.pad[1];
                            if (j < 0 || j >= W) continue;
                            if (xc[i * W + j] > m) {
                                m = xc[i * W + j];
                                bi = i;
                                bj = j;
                            }
                        }
                    }
                    if (bi >= 0) {
                        dp[((n * C + ch) * H + bi) * W + bj]
                            += gp[((n * C + ch) * HO + ho) * WO + wo];
                    }
                }
            }
        }
    }
}

const Operator g_max_pool2d_operator = {
    "max_pool2d",
    max_pool2d_infer,
    max_pool2d_grad,
    max_pool2d_ref,
    {max_pool2d_cpu, nullptr, nullptr},
    sizeof(PoolCallOptions),
};

Tensor Tensor::max_pool2d(const Tensor& x, Shape kernel, const Pool2dOptions& options) {
    const PoolCallOptions o{options, kernel};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_max_pool2d_operator,
                                        Span<const Tensor*>(ins, 1), &o);
}

} // namespace znn
