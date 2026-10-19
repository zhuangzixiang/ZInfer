// avg_pool2d: NCHW 平均池化; 梯度均摊到窗口内有效元素
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_avg_pool2d_operator;

static void avg_pool2d_ref(KernelContext&, Span<const Tensor*> in,
                           Span<Tensor*> out, const void* options);

static void avg_pool2d_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                             const void* o) {
    const PoolCallOptions* a = static_cast<const PoolCallOptions*>(o);
    if (!onednn::pool2d(*in[0], *out[0], a->kernel, a->pool, /*max_pool=*/false)) {
        avg_pool2d_ref(c, in, out, o);
    }
}


static void avg_pool2d_infer(Span<const TensorMeta*> in, const void* options,
                             std::vector<Shape>& outs) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    const Shape& x = *in[0]->shape;
    if (x.rank != 4) {
        throw std::runtime_error("znn: avg_pool2d expects NCHW");
    }
    const int64_t ho = (x[2] + 2 * o->pool.pad[0] - o->kernel[0]) / o->pool.stride[0] + 1;
    const int64_t wo = (x[3] + 2 * o->pool.pad[1] - o->kernel[1]) / o->pool.stride[1] + 1;
    outs.push_back(Shape{x[0], x[1], ho, wo});
}

static void avg_pool2d_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
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
                    float acc = 0.f;
                    int64_t cnt = 0;
                    for (int64_t ki = 0; ki < o->kernel[0]; ++ki) {
                        const int64_t i = ho * o->pool.stride[0] + ki - o->pool.pad[0];
                        if (i < 0 || i >= H) continue;
                        for (int64_t kj = 0; kj < o->kernel[1]; ++kj) {
                            const int64_t j = wo * o->pool.stride[1] + kj - o->pool.pad[1];
                            if (j < 0 || j >= W) continue;
                            acc += xc[i * W + j];
                            ++cnt;
                        }
                    }
                    yp[((n * C + c) * HO + ho) * WO + wo] = acc / static_cast<float>(cnt);
                }
            }
        }
    }
}

static void avg_pool2d_grad(BackwardContext& c) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(c.options);
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& g = *c.grad_outputs[0];
    Tensor& dx = *c.grad_inputs[0];
    const int64_t C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t HO = g.size(2), WO = g.size(3);
    if (x.device() == Device::CUDA) {       // 原生: 均摊梯度原子散射
        ZNN_CUDA_CALL(ZNN_CUDA_K(avg_pool_grad_cuda)(
            g.data<float>(), dx.data<float>(),
            x.size(0) * C * HO * WO, (int)H, (int)W, (int)HO, (int)WO,
            (int)o->kernel[0], (int)o->kernel[1],
            (int)o->pool.stride[0], (int)o->pool.stride[1],
            (int)o->pool.pad[0], (int)o->pool.pad[1]));
        return;
    }
    const float* gp = g.data<float>();
    float* dp = dx.data<float>();
    for (int64_t n = 0; n < x.size(0); ++n) {
        for (int64_t ch = 0; ch < C; ++ch) {
            for (int64_t ho = 0; ho < HO; ++ho) {
                for (int64_t wo = 0; wo < WO; ++wo) {
                    int64_t cnt = 0;
                    for (int64_t ki = 0; ki < o->kernel[0]; ++ki) {
                        const int64_t i = ho * o->pool.stride[0] + ki - o->pool.pad[0];
                        if (i < 0 || i >= H) continue;
                        for (int64_t kj = 0; kj < o->kernel[1]; ++kj) {
                            const int64_t j = wo * o->pool.stride[1] + kj - o->pool.pad[1];
                            if (j < 0 || j >= W) continue;
                            ++cnt;
                        }
                    }
                    const float share =
                        gp[((n * C + ch) * HO + ho) * WO + wo] / static_cast<float>(cnt);
                    for (int64_t ki = 0; ki < o->kernel[0]; ++ki) {
                        const int64_t i = ho * o->pool.stride[0] + ki - o->pool.pad[0];
                        if (i < 0 || i >= H) continue;
                        for (int64_t kj = 0; kj < o->kernel[1]; ++kj) {
                            const int64_t j = wo * o->pool.stride[1] + kj - o->pool.pad[1];
                            if (j < 0 || j >= W) continue;
                            dp[((n * C + ch) * H + i) * W + j] += share;
                        }
                    }
                }
            }
        }
    }
}

const Operator g_avg_pool2d_operator = {
    "avg_pool2d",
    avg_pool2d_infer,
    avg_pool2d_grad,
    avg_pool2d_ref,
    {avg_pool2d_cpu, nullptr, nullptr},
    sizeof(PoolCallOptions),
};

Tensor Tensor::avg_pool2d(const Tensor& x, Shape kernel, const Pool2dOptions& options) {
    const PoolCallOptions o{options, kernel};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_avg_pool2d_operator,
                                        Span<const Tensor*>(ins, 1), &o);
}

} // namespace znn
