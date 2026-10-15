// Adam: 动量更新 + 偏差校正
#include "optim/optim.hpp"
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

AdamImpl::AdamImpl(std::vector<Tensor> params, const OptimParam& param)
    : params_(std::move(params)), param_(param) {
    // TODO: 按参数形状分配 moments_(m/v 置零)
}

void AdamImpl::step() {
    ZNN_TODO();
}

void AdamImpl::clear() {
    // TODO: 各参数梯度清零
    ZNN_TODO();
}

} // namespace znn
