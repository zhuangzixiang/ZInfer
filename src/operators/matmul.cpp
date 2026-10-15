// matmul: 2D 矩阵乘, 高维按广播批处理
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_matmul_operator;
const Operator g_matmul_operator = {
    "matmul",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    ZNN_TODO();
}

} // namespace znn
