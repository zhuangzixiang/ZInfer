// arange: 等差序列 [0, n)
#include "znn_ext.hpp"

namespace znn {

extern const OpDef g_arange_def;
const OpDef g_arange_def = {
    "arange",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 无需求导
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::arange(int64_t n, DType dtype, Device device) {
    ZNN_TODO();
}

} // namespace znn
