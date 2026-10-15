// mul: 逐元素二元运算(输入间自动广播)
#include "znn_ext.hpp"

namespace znn {

extern const OpDef g_mul_def;
const OpDef g_mul_def = {
    "mul",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::mul(const Tensor& a, const Tensor& b) {
    ZNN_TODO();
}

} // namespace znn
