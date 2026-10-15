// Activation: ReLU/GELU/Sigmoid/Tanh 共用层(无参数)
#include "layers/layers.hpp"
#include "znn_ext.hpp"
#include "znn_internal.hpp"

namespace znn {

ActivationLayer::ActivationLayer(LayerType type)
    : type_(type) {
}

Tensor ActivationLayer::forward(const Tensor& x) {
    // TODO: 按类型分发 Activator:: 静态方法
    ZNN_TODO();
}

} // namespace znn
