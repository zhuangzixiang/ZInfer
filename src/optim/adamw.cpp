// AdamW: 复用 AdamImpl 动量逻辑, 仅 weight_decay 为解耦衰减
#include "optim/optim.hpp"
#include "znn_ext.hpp"

namespace znn {

void AdamWImpl::step() {
    // TODO: AdamImpl 动量更新 + 解耦衰减(衰减不经过动量)
    ZNN_TODO();
}

} // namespace znn
