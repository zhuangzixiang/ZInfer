// mul: 逐元素二元运算(输入间自动广播)
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_mul_operator;
const Operator g_mul_operator = {
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
