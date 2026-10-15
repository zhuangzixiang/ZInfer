// Flatten: 维度展平
#include "layers/layers.hpp"
#include "znn_ext.hpp"
#include "znn_internal.hpp"

namespace znn {

FlattenLayer::FlattenLayer(int64_t start_dim)
    : start_dim_(start_dim) {
}

Tensor FlattenLayer::forward(const Tensor& x) {
    // TODO: x.flatten(start_dim_)
    ZNN_TODO();
}

} // namespace znn
