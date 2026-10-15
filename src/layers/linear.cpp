// Linear: y = x @ w.T + b
#include "layers/layers.hpp"
#include "op_def.hpp"
#include "znn_internal.hpp"
#include "znn_internal.hpp"

namespace znn {

LinearLayer::LinearLayer(int64_t, int64_t, bool bias)
    : has_bias_(bias) {
    // TODO: w_/b_ 按形状分配并初始化
}

Tensor LinearLayer::forward(const Tensor& x) {
    // TODO: Tensor::linear(x, w_, has_bias_ ? &b_ : nullptr)
    ZNN_TODO();
}

std::vector<Tensor*> LinearLayer::params() const {
    return has_bias_ ? std::vector<Tensor*>{mut(w_), mut(b_)}
                     : std::vector<Tensor*>{mut(w_)};
}

} // namespace znn
