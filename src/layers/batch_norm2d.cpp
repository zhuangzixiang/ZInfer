// BatchNorm2d: 批归一化(train 态用批统计并更新 running, eval 态用 running)
#include "layers/layers.hpp"
#include "znn_internal.hpp"

namespace znn {

Tensor batch_norm2d_apply(const Tensor& x, const Tensor& w, const Tensor& b,
                          Tensor& running_mean, Tensor& running_var,
                          double eps, double momentum, bool training);

BatchNorm2dLayer::BatchNorm2dLayer(int64_t channels, double eps, double momentum)
    : w_(Tensor::ones({channels})), b_(Tensor::zeros({channels})),
      mean_(Tensor::zeros({channels})), var_(Tensor::ones({channels})),
      eps_(eps), momentum_(momentum) {
    w_.requires_grad(true);
    b_.requires_grad(true);
}

Tensor BatchNorm2dLayer::forward(const Tensor& x) {
    migrate_to_device(w_, x.device());
    migrate_to_device(b_, x.device());
    migrate_to_device(mean_, x.device());
    migrate_to_device(var_, x.device());
    return batch_norm2d_apply(x, w_, b_, mean_, var_, eps_, momentum_, training_);
}

std::vector<Tensor> BatchNorm2dLayer::params() const {
    return {w_, b_};
}

} // namespace znn
