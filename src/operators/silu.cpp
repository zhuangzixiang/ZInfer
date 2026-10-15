// silu: 激活函数
#include "op_def.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const OpDef g_silu_def;
const OpDef g_silu_def = {
    "silu",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Activator::silu(const Tensor& x) {
    ZNN_TODO();
}

} // namespace znn
