// AdamW: 与 Adam 仅解耦衰减语义(衰减直接作用于参数, 不经动量/梯度)
#include "optim.hpp"

namespace znn {

void AdamWImpl::step() {
    decoupled_ = true;
    AdamImpl::step();
}

} // namespace znn
