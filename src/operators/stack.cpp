// stack: 沿新维度 dim 堆叠(所有输入同形)
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_stack_operator;
const Operator g_stack_operator = {
    "stack",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::stack(const std::vector<Tensor>& tensors, int64_t dim) {
    ZNN_TODO();
}

} // namespace znn
