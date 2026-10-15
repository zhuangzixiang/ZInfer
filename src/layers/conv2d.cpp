// Conv2d: 2D 卷积层(NCHW); kaiming 风格初始化
#include "layers/layers.hpp"
#include "znn_internal.hpp"
#include <cmath>

namespace znn {

Conv2dLayer::Conv2dLayer(int64_t in_ch, int64_t out_ch, Shape kernel,
                         const Conv2dOptions& options, bool bias)
    : w_(Tensor::randn({out_ch, in_ch, kernel[0], kernel[1]})),
      options_(options), has_bias_(bias) {
    const float scale = 1.f / std::sqrt(static_cast<float>(in_ch * kernel[0] * kernel[1]));
    float* wp = w_.data<float>();
    for (int64_t i = 0; i < w_.numel(); ++i) wp[i] *= scale;
    if (has_bias_) {
        b_ = Tensor::zeros({out_ch});
        b_.requires_grad(true);
    }
    w_.requires_grad(true);
}

Tensor Conv2dLayer::forward(const Tensor& x) {
    Tensor y = Tensor::conv2d(x, w_, options_);
    return has_bias_ ? Tensor::add(y, b_) : y;            // b 按通道广播
}

std::vector<Tensor> Conv2dLayer::params() const {
    return has_bias_ ? std::vector<Tensor>{w_, b_} : std::vector<Tensor>{w_};
}

} // namespace znn
