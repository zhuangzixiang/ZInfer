// Conv2d: 2D 卷积层(NCHW); kaiming 风格初始化; groups>1 时权重为
// [O, C/groups, KH, KW](conv2d 算子原生支持分组, 工厂此前漏建分组形状)
#include "layers/layers.hpp"
#include "znn_internal.hpp"
#include <cmath>

namespace znn {

Conv2dLayer::Conv2dLayer(int64_t in_ch, int64_t out_ch, Shape kernel,
                         const Conv2dOptions& options, bool bias)
    : w_(Tensor::randn({out_ch, in_ch / options.groups, kernel[0], kernel[1]})),
      options_(options), has_bias_(bias) {
    const float fan_in = static_cast<float>(in_ch / options.groups
                                            * kernel[0] * kernel[1]);
    const float scale = 1.f / std::sqrt(fan_in);
    float* wp = w_.data<float>();
    for (int64_t i = 0; i < w_.numel(); ++i) wp[i] *= scale;
    if (has_bias_) {
        b_ = Tensor::zeros({1, out_ch, 1, 1});    // 4D 通道广播形态
        b_.requires_grad(true);
    }
    w_.requires_grad(true);
}

Tensor Conv2dLayer::forward(const Tensor& x) {
    migrate_to_device(w_, x.device());
    migrate_to_device(b_, x.device());
    Tensor y = Tensor::conv2d(x, w_, options_);
    return has_bias_ ? Tensor::add(y, b_) : y;    // b 按 [1,C,1,1] 广播
}

std::vector<Tensor> Conv2dLayer::params() const {
    return has_bias_ ? std::vector<Tensor>{w_, b_} : std::vector<Tensor>{w_};
}

} // namespace znn
