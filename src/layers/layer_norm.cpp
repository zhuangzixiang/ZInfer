// LayerNorm: 层归一化 + 仿射
#include "layers/layers.hpp"
#include "znn_internal.hpp"

namespace znn {

Tensor layer_norm_apply(const Tensor& x, const Tensor& w, const Tensor& b, double eps,
                         int64_t dim = -1);

LayerNormLayer::LayerNormLayer(int64_t normalized_dim, double eps, int64_t dim)
    : w_(Tensor::ones({normalized_dim})), b_(Tensor::zeros({normalized_dim})),
      eps_(eps), dim_(dim) {
    w_.requires_grad(true);
    b_.requires_grad(true);
}

Tensor LayerNormLayer::forward(const Tensor& x) {
    migrate_to_device(w_, x.device());
    migrate_to_device(b_, x.device());
    return layer_norm_apply(x, w_, b_, eps_, dim_);
}

std::vector<Tensor> LayerNormLayer::params() const {
    return {w_, b_};
}

} // namespace znn
