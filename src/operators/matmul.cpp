// matmul: 2D 矩阵乘, 高维按广播批处理
#include "znn_ext.hpp"

namespace znn {

extern const OpDef g_matmul_def;
const OpDef g_matmul_def = {
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
