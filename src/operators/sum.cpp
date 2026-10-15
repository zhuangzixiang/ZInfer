// sum: 沿 dim 规约(keepdim 控制保留维度)
#include "znn_ext.hpp"

namespace znn {

extern const OpDef g_sum_def;
const OpDef g_sum_def = {
    "sum",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::sum(const Tensor& x, int64_t dim, bool keepdim) {
    ZNN_TODO();
}

} // namespace znn
