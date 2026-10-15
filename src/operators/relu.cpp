// relu: 激活函数
#include "znn_ext.hpp"

namespace znn {

extern const OpDef g_relu_def;
const OpDef g_relu_def = {
    "relu",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Activator::relu(const Tensor& x) {
    ZNN_TODO();
}

} // namespace znn
