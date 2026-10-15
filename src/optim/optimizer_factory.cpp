// Optimizer 工厂: layers 重载自动收集各层参数
#include "optim.hpp"
#include "znn_internal.hpp"
#include <stdexcept>

namespace znn {

Optimizer* Optimizer::create(OptimType type, const std::vector<Layer*>& layers,
                             const OptimParam& param) {
    std::vector<Tensor> params;
    for (Layer* layer : layers) {
        for (Tensor& t : layer->params()) {
            params.push_back(t);
        }
    }
    return create(type, std::move(params), param);
}

Optimizer* Optimizer::create(OptimType type, std::vector<Tensor> params,
                             const OptimParam& param) {
    switch (type) {
        case OptimType::Sgd:   return new SgdImpl(std::move(params), param);
        case OptimType::Adam:  return new AdamImpl(std::move(params), param);
        case OptimType::AdamW: return new AdamWImpl(std::move(params), param);
        default: throw std::runtime_error("znn: unknown optimizer type");
    }
}

void Optimizer::destroy(Optimizer* opt) {
    delete opt;
}

} // namespace znn
