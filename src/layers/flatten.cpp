// Flatten: 维度展平(训练态经 reshape 算子材料化, 梯度自动回传)
#include "layers/layers.hpp"
#include "znn_internal.hpp"

namespace znn {

FlattenLayer::FlattenLayer(int64_t start_dim)
    : start_dim_(start_dim) {
}

Tensor FlattenLayer::forward(const Tensor& x) {
    return x.flatten(start_dim_);
}

} // namespace znn
