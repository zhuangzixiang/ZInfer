// Pool2d: MaxPool2d/AvgPool2d 共用层(NCHW)
#include "layers/layers.hpp"
#include "op_def.hpp"
#include "znn_internal.hpp"
#include "znn_internal.hpp"

namespace znn {

Pool2dLayer::Pool2dLayer(LayerType type, Shape kernel, const Pool2dOptions& options)
    : type_(type), kernel_(kernel), options_(options) {
}

Tensor Pool2dLayer::forward(const Tensor& x) {
    // TODO: 按类型分发 Tensor::max_pool2d / avg_pool2d
    ZNN_TODO();
}

} // namespace znn
