// batch_norm2d: 通道归一化(NCHW); 训练态用批统计并就地更新 running 均值/方差,
// 推理态用 running 统计; 梯度按标准 BN 反传公式(训练态)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"
#include <cmath>

namespace znn {

extern const Operator g_batch_norm2d_operator;

static void batch_norm2d_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void* options);

static void batch_norm2d_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                                const void* o) {
    const float* opts = static_cast<const float*>(o);   // {eps, momentum, training}
    // 训练态统计更新走组合路径(ref), 推理统计量走 oneDNN
    if (opts[2] == 0.f
        && !onednn::batch_norm(*in[0], *in[1], *in[2], *in[3], *in[4],
                               *out[0], opts[0])) {
        batch_norm2d_ref(c, in, out, o);
        return;
    }
    if (opts[2] != 0.f) {
        batch_norm2d_ref(c, in, out, o);
    }
}


struct BatchNormOptions {
    double eps;
    double momentum;
    bool   training;
};

static void batch_norm2d_infer(Span<const TensorMeta*> in, const void*,
                               ShapeList& outs) {
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
    // 全算子组合: 经 sum/mul/sub/mean/div/sqrt 自动走当前后端
    const BatchNormOptions* o = static_cast<const BatchNormOptions*>(c.options);
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& w = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    const Device dev = x.device();
    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t M = N * H * W;
    Tensor x2 = x.reshape(Shape{M, C});                    // 反向期间纯视图
    Tensor mu = Tensor::mean(x2, 0);                       // [C]
    Tensor xc = Tensor::sub(x2, mu);
    Tensor var = Tensor::mean(Tensor::mul(xc, xc), 0);
    Tensor eps_t = Tensor::full(Shape{C}, o->eps, DType::F32, dev);
    Tensor vp = Tensor::add(var, eps_t);
    Tensor inv = Tensor::div(Tensor::ones(Shape{C}, DType::F32, dev), Tensor::sqrt(vp));
    Tensor xh = Tensor::mul(xc, inv);                      // x̂ [M, C]
    Tensor g2 = g.reshape(Shape{M, C});
    *c.grad_inputs[1] = Tensor::sum(Tensor::mul(g2, xh), 0);   // dγ [C]
    *c.grad_inputs[2] = Tensor::sum(g2, 0);                    // dβ [C]
    Tensor sg = Tensor::sum(g2, 0);                        // Σg
    Tensor sgx = Tensor::sum(Tensor::mul(g2, xh), 0);      // Σ(g·x̂)
    Tensor M_t = Tensor::full(Shape{C}, static_cast<double>(M), DType::F32, dev);
    Tensor t1 = Tensor::mul(M_t, g2);                      // M·g
    Tensor t2 = Tensor::sub(t1, sg);                       // M·g − Σg
    Tensor t3 = Tensor::sub(t2, Tensor::mul(xh, sgx));    // − x̂·Σ(g·x̂)
    Tensor coef = Tensor::div(Tensor::mul(w, inv), M_t);  // γ·inv/M
    Tensor dx2 = Tensor::mul(t3, coef);                    // [M, C]
    *c.grad_inputs[0] = dx2.reshape(x.shape());
}

const Operator g_batch_norm2d_operator = {
    "batch_norm2d",
    batch_norm2d_infer,
    batch_norm2d_grad,
    batch_norm2d_ref,
    {batch_norm2d_cpu, ZNN_CUDA_K(batch_norm2d_cuda), ZNN_METAL_K(batch_norm2d_metal), ZNN_VULKAN_K(batch_norm2d_vulkan)},
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
