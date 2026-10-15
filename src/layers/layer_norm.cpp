// LayerNorm: 层归一化 + 仿射
#include "layers/layers.hpp"
#include "znn_ext.hpp"
#include "znn_internal.hpp"

namespace znn {

LayerNormLayer::LayerNormLayer(int64_t, double eps)
    : eps_(eps) {
    // TODO: w_/b_ 按维度分配
}

Tensor LayerNormLayer::forward(const Tensor& x) {
    // TODO: 均值方差归一化 + 仿射
    ZNN_TODO();
}

std::vector<Tensor*> LayerNormLayer::params() const {
    return {mut(w_), mut(b_)};
}

} // namespace znn
