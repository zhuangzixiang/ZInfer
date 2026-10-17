// Linear: y = x @ w.T + b; kaiming 风格初始化
#include "layers/layers.hpp"
#include "znn_internal.hpp"

namespace znn {

LinearLayer::LinearLayer(int64_t in, int64_t out, bool bias)
    : w_(Tensor::randn({out, in}, /*seed=*/0)), has_bias_(bias) {
    const float scale = 1.f / sqrtf(static_cast<float>(in));   // 方差 ≈ 1/in
    float* wp = w_.data<float>();
    for (int64_t i = 0; i < w_.numel(); ++i) wp[i] *= scale;
    if (has_bias_) {
        b_ = Tensor::zeros({out});
    }
    w_.requires_grad(true);
    b_.requires_grad(true);
}

Tensor LinearLayer::forward(const Tensor& x) {
    migrate_to_device(w_, x.device());
    migrate_to_device(b_, x.device());
    return Tensor::linear(x, w_, has_bias_ ? &b_ : nullptr);
}

std::vector<Tensor> LinearLayer::params() const {
    return has_bias_ ? std::vector<Tensor>{w_, b_} : std::vector<Tensor>{w_};
}

} // namespace znn
