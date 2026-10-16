// batch_norm2d: 通道归一化(NCHW); 训练态用批统计并就地更新 running 均值/方差,
// 推理态用 running 统计; 梯度按标准 BN 反传公式(训练态)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include <cmath>

namespace znn {

extern const Operator g_batch_norm2d_operator;

struct BatchNormOptions {
    double eps;
    double momentum;
    bool   training;
};

static void batch_norm2d_infer(Span<const TensorMeta*> in, const void*,
                               std::vector<Shape>& outs) {
    outs.push_back(*in[0]->shape);
}

static void batch_norm2d_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                             const void* options) {
    const BatchNormOptions* o = static_cast<const BatchNormOptions*>(options);
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    const Tensor& b = *in[2];
    const Tensor& run_mean = *in[3];   // data() const 返回可写指针, 支持就地更新
    const Tensor& run_var = *in[4];
    Tensor& y = *out[0];
    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t M = N * H * W;                          // 每通道元素数
    const float* xp = x.data<float>();
    const float* wp = w.data<float>();
    const float* bp = b.data<float>();
    float* mp = run_mean.data<float>();                   // 经共享存储就地更新
    float* vp = run_var.data<float>();
    float* yp = y.data<float>();
    for (int64_t c = 0; c < C; ++c) {
        float mu = 0.f;
        float var = 0.f;
        if (o->training) {
            for (int64_t n = 0; n < N; ++n) {
                const float* xc = xp + ((n * C + c) * H) * W;
                for (int64_t i = 0; i < H * W; ++i) mu += xc[i];
            }
            mu /= static_cast<float>(M);
            for (int64_t n = 0; n < N; ++n) {
                const float* xc = xp + ((n * C + c) * H) * W;
                for (int64_t i = 0; i < H * W; ++i) {
                    const float t = xc[i] - mu;
                    var += t * t;
                }
            }
            var /= static_cast<float>(M);
            const float mom = static_cast<float>(o->momentum);
            mp[c] = (1.f - mom) * mp[c] + mom * mu;
            vp[c] = (1.f - mom) * vp[c] + mom * var * static_cast<float>(M) / static_cast<float>(M - 1 > 0 ? M - 1 : 1);
        } else {
            mu = mp[c];
            var = vp[c];
        }
        const float inv = 1.f / std::sqrt(var + static_cast<float>(o->eps));
        for (int64_t n = 0; n < N; ++n) {
            const float* xc = xp + ((n * C + c) * H) * W;
            float* yc = yp + ((n * C + c) * H) * W;
            for (int64_t i = 0; i < H * W; ++i) {
                yc[i] = (xc[i] - mu) * inv * wp[c] + bp[c];
            }
        }
    }
}

static void batch_norm2d_grad(BackwardContext& c) {
    const BatchNormOptions* o = static_cast<const BatchNormOptions*>(c.options);
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& w = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    Tensor& dx = *c.grad_inputs[0];
    Tensor& dw = *c.grad_inputs[1];
    Tensor& db = *c.grad_inputs[2];
    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t M = N * H * W;
    const float* xp = x.data<float>();
    const float* wp = w.data<float>();
    const float* gp = g.data<float>();
    float* dxp = dx.data<float>();
    float* dwp = dw.data<float>();
    float* dbp = db.data<float>();
    for (int64_t ch = 0; ch < C; ++ch) {
        float mu = 0.f;
        float var = 0.f;
        for (int64_t n = 0; n < N; ++n) {
            const float* xc = xp + ((n * C + ch) * H) * W;
            for (int64_t i = 0; i < H * W; ++i) mu += xc[i];
        }
        mu /= static_cast<float>(M);
        for (int64_t n = 0; n < N; ++n) {
            const float* xc = xp + ((n * C + ch) * H) * W;
            for (int64_t i = 0; i < H * W; ++i) {
                const float t = xc[i] - mu;
                var += t * t;
            }
        }
        var /= static_cast<float>(M);
        const float inv = 1.f / std::sqrt(var + static_cast<float>(o->eps));
        float sg = 0.f;                                    // Σ g
        float sgx = 0.f;                                   // Σ g·x̂
        for (int64_t n = 0; n < N; ++n) {
            const float* xc = xp + ((n * C + ch) * H) * W;
            const float* gc = gp + ((n * C + ch) * H) * W;
            for (int64_t i = 0; i < H * W; ++i) {
                const float xh = (xc[i] - mu) * inv;
                sg += gc[i];
                sgx += gc[i] * xh;
                dwp[ch] += gc[i] * xh;
                dbp[ch] += gc[i];
            }
        }
        const float coef = wp[ch] * inv / static_cast<float>(M);
        for (int64_t n = 0; n < N; ++n) {
            const float* xc = xp + ((n * C + ch) * H) * W;
            const float* gc = gp + ((n * C + ch) * H) * W;
            float* dc = dxp + ((n * C + ch) * H) * W;
            for (int64_t i = 0; i < H * W; ++i) {
                const float xh = (xc[i] - mu) * inv;
                dc[i] = coef * (static_cast<float>(M) * gc[i] - sg - xh * sgx);
            }
        }
    }
}

const Operator g_batch_norm2d_operator = {
    "batch_norm2d",
    batch_norm2d_infer,
    batch_norm2d_grad,
    batch_norm2d_ref,
    {nullptr, cuda_backend::batch_norm2d_cuda, nullptr, nullptr},
    sizeof(BatchNormOptions),
};

Tensor batch_norm2d_apply(const Tensor& x, const Tensor& w, const Tensor& b,
                          Tensor& running_mean, Tensor& running_var,
                          double eps, double momentum, bool training) {
    const BatchNormOptions o{eps, momentum, training};
    const Tensor* ins[5] = {&x, &w, &b, &running_mean, &running_var};
    return OpDispatcher::instance().run(&g_batch_norm2d_operator,
                                        Span<const Tensor*>(ins, 5), &o);
}

} // namespace znn
