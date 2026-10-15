// mse: 均方误差(回归标准损失), 返回标量
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_mse_operator;
const Operator g_mse_operator = {
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
