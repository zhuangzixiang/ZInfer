// max_pool2d: 池化(NCHW)
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_max_pool2d_operator;
const Operator g_max_pool2d_operator = {
    "max_pool2d",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补(仅 max 需记录掩码)
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::max_pool2d(const Tensor& x, Shape kernel, const Pool2dOptions& options) {
    ZNN_TODO();
}

} // namespace znn
