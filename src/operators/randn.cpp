// randn: 标准正态随机张量(seed 为 0 时用全局随机源)
#include "op_def.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const OpDef g_randn_def;
const OpDef g_randn_def = {
    "randn",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 无需求导
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::randn(Shape shape, unsigned seed, DType dtype, Device device) {
    ZNN_TODO();
}

} // namespace znn
