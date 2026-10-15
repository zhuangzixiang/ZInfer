// Dropout: 逆置失活 y = x·mask/(1-p); mask 为常量张量, 梯度经 mul 算子
// 自动回传(g·mask/(1-p)); 推理态直通
#include "layers/layers.hpp"
#include "znn_internal.hpp"
#include <cmath>
#include <random>

namespace znn {

DropoutLayer::DropoutLayer(double rate, unsigned seed)
    : rate_(rate), rng_(seed ? seed : 12345u) {
}

Tensor DropoutLayer::forward(const Tensor& x) {
    if (!training_ || rate_ <= 0.0) {
        return x;                                          // 推理/eval 直通
    }
    Tensor mask = Tensor::empty(x.shape());
    float* mp = mask.data<float>();
    std::bernoulli_distribution keep(1.0 - rate_);
    const float scale = 1.f / static_cast<float>(1.0 - rate_);
    for (int64_t i = 0; i < mask.numel(); ++i) {
        mp[i] = keep(rng_) ? scale : 0.f;                  // 逆置失活缩放
    }
    return Tensor::mul(x, mask);                           // mask 无梯度, x 梯度自动回传
}

} // namespace znn
