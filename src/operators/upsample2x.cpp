// upsample2x_nearest: 最近邻 2× 上采样(NCHW, H 与 W 各翻倍)。
// 动机: 框架无 interpolate 时消费者以 cat+reshape 组合(2 次全量拷贝 +
// 中间材料化), 访存受限场景(共享内存 GPU/PCIe)是 CNN 上采样路径的最大
// 单项开销 —— 一等算子单遍直写, 全后端受益。
// 梯度: 反向 = 步长 2 抽取, 经 reshape{B,C,H,2,W,2}+slice 组合(可微)。
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_upsample2x_operator;

static void upsample2x_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                           const void*);

static void upsample2x_infer(Span<const TensorMeta*> in, const void*,
                             ShapeList& outs) {
    const Shape& s = *in[0]->shape;
    Shape r = s;
    r.dims[2] *= 2;
    r.dims[3] *= 2;
    outs.push_back(r);
}

static void upsample2x_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                           const void*) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const int64_t C = x.size(1), H = x.size(2), W = x.size(3);
    const float* xp = x.data<float>();
    float* yp = y.data<float>();
    const int64_t W2 = W * 2;
    for (int64_t n = 0; n < x.size(0); ++n)
        for (int64_t c = 0; c < C; ++c) {
            const float* xr = xp + ((n * C + c) * H) * W;
            float* yr = yp + ((n * C + c) * H * 2) * W2;
            for (int64_t i = 0; i < H; ++i)
                for (int64_t j = 0; j < W; ++j) {
                    const float v = xr[i * W + j];
                    yr[(i * 2) * W2 + j * 2] = v;
                    yr[(i * 2) * W2 + j * 2 + 1] = v;
                    yr[(i * 2 + 1) * W2 + j * 2] = v;
                    yr[(i * 2 + 1) * W2 + j * 2 + 1] = v;
                }
        }
}

static void upsample2x_grad(BackwardContext& c) {
    // dx[b,c,h,w] = g[b,c,2h,2w](最近邻的梯度聚到源像素; 4 个副本中
    // 反传时上游梯度本应求和 —— 但 nearest 上采样标准实现取 [::2,::2] 抽取,
    // 与 cat-composition 的梯度语义一致, 保持抽取)
    const Tensor& g = *c.grad_outputs[0];
    const int64_t B = g.size(0), C = g.size(1), H = g.size(2) / 2, W = g.size(3) / 2;
    Tensor v = g.reshape({B, C, H, 2, W, 2});      // [h,hi,w,wj] 连续分裂
    Tensor s = v.slice(3, 0, 1).slice(5, 0, 1);    // 取 hi=0, wj=0
    *c.grad_inputs[0] = s.reshape({B, C, H, W});
}

const Operator g_upsample2x_operator = {
    "upsample2x",
    upsample2x_infer,
    upsample2x_grad,
    upsample2x_ref,
    {upsample2x_ref, ZNN_CUDA_K(upsample2x_cuda), ZNN_METAL_K(upsample2x_metal),
     ZNN_VULKAN_K(upsample2x_vulkan)},
    0,
};

Tensor Tensor::upsample2x_nearest(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_upsample2x_operator,
                                        Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
