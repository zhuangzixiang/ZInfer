// mean: 沿 dim 规约(keepdim 控制保留维度)
#include "op_def.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const OpDef g_mean_def;
const OpDef g_mean_def = {
    "mean",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::mean(const Tensor& x, int64_t dim, bool keepdim) {
    ZNN_TODO();
}

} // namespace znn
