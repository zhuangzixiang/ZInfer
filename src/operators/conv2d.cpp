// conv2d: 2D 卷积(NCHW; w 形状 [out, in/groups, kh, kw])
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_conv2d_operator;
const Operator g_conv2d_operator = {
    "conv2d",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::conv2d(const Tensor& x, const Tensor& w, const Conv2dOptions& options) {
    ZNN_TODO();
}

} // namespace znn
