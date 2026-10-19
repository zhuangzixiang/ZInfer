// ============================================================================
// grad_scaler.cpp — FP16 混合精度训练的动态损失缩放
// (动机与用法见 znn.hpp 的 GradScaler 注释)
// ============================================================================
#include "znn_internal.hpp"
#include "operators/shared.hpp"
#include "backend_slots.hpp"
#include <cmath>

namespace znn {

GradScaler::GradScaler(float init_scale, int growth_interval)
    : scale_(init_scale), growth_interval_(growth_interval) {}

Tensor GradScaler::scale(const Tensor& loss) const {
    // 建图内 ×s: 反传路径自动把全部梯度放大 s 倍(mul 的梯度即 ×s)
    Tensor s = Tensor::full(Shape{}, scale_, loss.dtype(), loss.device());
    return Tensor::mul(loss, s);
}

bool GradScaler::unscale(Optimizer& opt) {
    const float inv = 1.f / scale_;
    bool finite = true;
    for (Tensor& p : opt.params_) {
        Tensor& g = p.grad();               // 无梯度参数惰性分配零张量(有限, 无害)
        const int64_t n = g.numel();
        if (g.device() == Device::CUDA) {
            // 原生内核: 梯度 ×(1/s) + 发现 inf/nan 置标志
            Tensor bad = Tensor::zeros(Shape{1}, DType::F32, Device::CUDA);
            ZNN_CUDA_CALL(ZNN_CUDA_K(grad_unscale_cuda)(cuda_ptr(g), cuda_ptr(bad),
                                            n, inv));
            finite = finite && bad.to_device(Device::CPU).data<float>()[0] == 0.f;
        } else {
            float* gd = g.data<float>();
            for (int64_t k = 0; k < n; ++k) {
                gd[k] *= inv;
                finite = finite && std::isfinite(gd[k]);
            }
        }
    }
    last_finite_ = finite;
    return finite;
}

void GradScaler::update() {
    if (!last_finite_) {
        scale_ *= 0.5f;                     // 梯度溢出: 收缩; 下限 1(只放大不缩小)
        if (scale_ < 1.f) scale_ = 1.f;
        good_steps_ = 0;
        return;
    }
    if (++good_steps_ >= growth_interval_) {
        scale_ *= 2.f;                      // 连续健康: 渐增; 上限 2^22 防无界
        if (scale_ > 4194304.f) scale_ = 4194304.f;
        good_steps_ = 0;
    }
}

} // namespace znn
