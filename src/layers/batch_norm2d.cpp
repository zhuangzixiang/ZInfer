// BatchNorm2d: 批归一化(train 态更新 running 统计, eval 态使用统计)
#include "layers/layers.hpp"
#include "znn_ext.hpp"
#include "znn_internal.hpp"

namespace znn {

BatchNorm2dLayer::BatchNorm2dLayer(int64_t, double eps, double momentum)
    : eps_(eps), momentum_(momentum) {
    // TODO: w_/b_/mean_/var_ 按通道数分配
}

Tensor BatchNorm2dLayer::forward(const Tensor& x) {
    // TODO: 归一化 + 仿射; 训练态按 momentum 更新 running 统计
    ZNN_TODO();
}

std::vector<Tensor*> BatchNorm2dLayer::params() const {
    return {mut(w_), mut(b_)};
}

} // namespace znn
