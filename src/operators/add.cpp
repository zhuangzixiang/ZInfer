// add: 逐元素二元运算(输入间自动广播)
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_add_operator;
const Operator g_add_operator = {
    "add",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::add(const Tensor& a, const Tensor& b) {
    ZNN_TODO();
}

} // namespace znn
