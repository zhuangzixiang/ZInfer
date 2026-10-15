// Dropout: 训练态按 rate 掩码缩放, 推理态直通
#include "layers/layers.hpp"
#include "op_def.hpp"
#include "znn_internal.hpp"
#include "znn_internal.hpp"

namespace znn {

DropoutLayer::DropoutLayer(double rate, unsigned seed)
    : rate_(rate), seed_(seed) {
}

Tensor DropoutLayer::forward(const Tensor& x) {
    ZNN_TODO();
}

} // namespace znn
