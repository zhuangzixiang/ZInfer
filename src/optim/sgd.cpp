// SGD: p -= lr * (grad + momentum 项 + weight_decay 项)
#include "optim/optim.hpp"
#include "znn_ext.hpp"

namespace znn {

SgdImpl::SgdImpl(std::vector<Tensor*> params, const OptimParam& param)
    : params_(std::move(params)), param_(param) {
    // TODO: momentum > 0 时分配速度缓冲
}

void SgdImpl::step() {
    ZNN_TODO();
}

void SgdImpl::clear() {
    // TODO: 各参数梯度清零
    ZNN_TODO();
}

} // namespace znn
