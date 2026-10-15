// mse: 均方误差(回归标准损失), 返回标量
#include "znn_ext.hpp"

namespace znn {

extern const OpDef g_mse_def;
const OpDef g_mse_def = {
    "mse",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Loss::mse(const Tensor& pred, const Tensor& target) {
    ZNN_TODO();
}

} // namespace znn
