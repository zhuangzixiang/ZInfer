// Activation: ReLU/GELU/Sigmoid/Tanh 共用层(无参数)
#include "layers/layers.hpp"
#include "znn_internal.hpp"

namespace znn {

ActivationLayer::ActivationLayer(LayerType type)
    : type_(type) {
}

Tensor ActivationLayer::forward(const Tensor& x) {
    switch (type_) {
        case LayerType::ReLU:    return Activator::relu(x);
        case LayerType::GELU:    return Activator::gelu(x);
        case LayerType::Sigmoid: return Activator::sigmoid(x);
        case LayerType::Tanh:    return Activator::tanh(x);
        default: throw std::runtime_error("znn: activation layer: bad type");
    }
}

} // namespace znn
