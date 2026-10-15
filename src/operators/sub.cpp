// sub: 逐元素二元运算(输入间自动广播)
#include "op_def.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const OpDef g_sub_def;
const OpDef g_sub_def = {
    "sub",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::sub(const Tensor& a, const Tensor& b) {
    ZNN_TODO();
}

} // namespace znn
