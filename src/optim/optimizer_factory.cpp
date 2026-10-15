// Optimizer 工厂: 按 OptimType 分发; layers 重载自动收集各层参数
#include "optim/optim.hpp"
#include "znn_ext.hpp"

namespace znn {

Optimizer* Optimizer::create(OptimType type, const std::vector<Layer*>& layers,
                             const OptimParam& param) {
    // TODO: 收集各层 params() 后委托给下方 create
    ZNN_TODO();
}

Optimizer* Optimizer::create(OptimType type, std::vector<Tensor*> params,
                             const OptimParam& param) {
    // TODO: switch(type) 分发到 SgdImpl/AdamImpl/AdamWImpl
    ZNN_TODO();
}

void Optimizer::destroy(Optimizer* opt) {
    delete opt;
}

} // namespace znn
