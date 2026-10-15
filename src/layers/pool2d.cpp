// Pool2d: MaxPool2d/AvgPool2d 共用层(NCHW)
#include "layers/layers.hpp"
#include "znn_internal.hpp"

namespace znn {

Pool2dLayer::Pool2dLayer(LayerType type, Shape kernel, const Pool2dOptions& options)
    : type_(type), kernel_(kernel), options_(options) {
}

Tensor Pool2dLayer::forward(const Tensor& x) {
    if (type_ == LayerType::MaxPool2d) {
        return Tensor::max_pool2d(x, kernel_, options_);
    }
    return Tensor::avg_pool2d(x, kernel_, options_);
}

} // namespace znn
