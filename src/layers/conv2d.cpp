// Conv2d: 2D 卷积层(NCHW)
#include "layers/layers.hpp"
#include "znn_ext.hpp"
#include "znn_internal.hpp"

namespace znn {

Conv2dLayer::Conv2dLayer(int64_t, int64_t, Shape kernel,
                         const Conv2dOptions& options, bool bias)
    : options_(options), has_bias_(bias) {
    // TODO: w_/b_ 按形状分配并初始化(kernel 参数用于校验)
}

Tensor Conv2dLayer::forward(const Tensor& x) {
    // TODO: Tensor::conv2d(x, w_, options_)
    ZNN_TODO();
}

std::vector<Tensor*> Conv2dLayer::params() const {
    return has_bias_ ? std::vector<Tensor*>{mut(w_), mut(b_)}
                     : std::vector<Tensor*>{mut(w_)};
}

} // namespace znn
