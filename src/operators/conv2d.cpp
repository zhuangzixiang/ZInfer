// conv2d: 2D 卷积(NCHW, v1 groups=1) 直接朴素实现;
// 梯度 dX/dW 按标准卷积反传公式直接累加
#include "shared.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_conv2d_operator;

static void conv2d_infer(Span<const TensorMeta*> in, const void* options,
                         std::vector<Shape>& outs) {
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(options);
    const Shape& x = *in[0]->shape;                       // {N,C,H,W}
    const Shape& w = *in[1]->shape;                       // {O,C,KH,KW}
    if (x.rank != 4 || w.rank != 4 || x[1] != w[1] || o->groups != 1) {
        throw std::runtime_error("znn: conv2d expects NCHW x [O,C,KH,KW], groups=1");
    }
    const int64_t ho = (x[2] + 2 * o->pad[0] - o->dilation[0] * (w[2] - 1) - 1) / o->stride[0] + 1;
    const int64_t wo = (x[3] + 2 * o->pad[1] - o->dilation[1] * (w[3] - 1) - 1) / o->stride[1] + 1;
    outs.push_back(Shape{x[0], w[0], ho, wo});
}

static void conv2d_ref(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                       const void* options) {
    conv2d_naive(c, in, out, options);
}

void conv2d_naive(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void* options) {
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(options);
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    Tensor& y = *out[0];
    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t O = w.size(0), KH = w.size(2), KW = w.size(3);
    const int64_t HO = y.size(2), WO = y.size(3);
    const float* xp = x.data<float>();
    const float* wp = w.data<float>();
    float* yp = y.data<float>();
    const int64_t* xs = x.strides();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < O; ++oc) {
            for (int64_t ho = 0; ho < HO; ++ho) {
                for (int64_t wo = 0; wo < WO; ++wo) {
                    float acc = 0.f;
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t ki = 0; ki < KH; ++ki) {
                            const int64_t i = ho * o->stride[0] + ki * o->dilation[0] - o->pad[0];
                            if (i < 0 || i >= H) continue;
                            for (int64_t kj = 0; kj < KW; ++kj) {
                                const int64_t j = wo * o->stride[1] + kj * o->dilation[1] - o->pad[1];
                                if (j < 0 || j >= W) continue;
                                acc += xp[n * xs[0] + c * xs[1] + i * xs[2] + j * xs[3]]
                                     * wp[((oc * C + c) * KH + ki) * KW + kj];
                            }
                        }
                    }
                    yp[((n * O + oc) * HO + ho) * WO + wo] = acc;
                }
            }
        }
    }
}

static void conv2d_grad(BackwardContext& c) {
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(c.options);
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& w = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    Tensor& dx = *c.grad_inputs[0];
    Tensor& dw = *c.grad_inputs[1];
    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t O = w.size(0), KH = w.size(2), KW = w.size(3);
    const int64_t HO = g.size(2), WO = g.size(3);
    const float* xp = x.data<float>();
    const float* wp = w.data<float>();
    const float* gp = g.data<float>();
    float* dxp = dx.data<float>();
    float* dwp = dw.data<float>();
    const int64_t* xs = x.strides();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t oc = 0; oc < O; ++oc) {
            for (int64_t ho = 0; ho < HO; ++ho) {
                for (int64_t wo = 0; wo < WO; ++wo) {
                    const float dy = gp[((n * O + oc) * HO + ho) * WO + wo];
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t ki = 0; ki < KH; ++ki) {
                            const int64_t i = ho * o->stride[0] + ki * o->dilation[0] - o->pad[0];
                            if (i < 0 || i >= H) continue;
                            for (int64_t kj = 0; kj < KW; ++kj) {
                                const int64_t j = wo * o->stride[1] + kj * o->dilation[1] - o->pad[1];
                                if (j < 0 || j >= W) continue;
                                const float xv = xp[n * xs[0] + c * xs[1] + i * xs[2] + j * xs[3]];
                                dwp[((oc * C + c) * KH + ki) * KW + kj] += dy * xv;
                                dxp[((n * C + c) * H + i) * W + j] += dy * wp[((oc * C + c) * KH + ki) * KW + kj];
                            }
                        }
                    }
                }
            }
        }
    }
}

KernelFunction conv2d_cpu_kernel();   // conv2d_eigen.cpp: im2col + Eigen GEMM

const Operator g_conv2d_operator = {
    "conv2d",
    conv2d_infer,
    conv2d_grad,
    conv2d_ref,
    {conv2d_cpu_kernel()},       // kernels[CPU] = Eigen(非连续输入内部回退)
    sizeof(Conv2dOptions),
};

Tensor Tensor::conv2d(const Tensor& x, const Tensor& w, const Conv2dOptions& options) {
    const Tensor* ins[2] = {&x, &w};
    return OpDispatcher::instance().run(&g_conv2d_operator,
                                        Span<const Tensor*>(ins, 2), &options);
}

} // namespace znn
