// avg_pool2d: 池化(NCHW)
#include "op_def.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const OpDef g_avg_pool2d_def;
const OpDef g_avg_pool2d_def = {
    "avg_pool2d",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补(仅 max 需记录掩码)
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::avg_pool2d(const Tensor& x, Shape kernel, const Pool2dOptions& options) {
    ZNN_TODO();
}

} // namespace znn
