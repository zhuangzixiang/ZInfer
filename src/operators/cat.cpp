// cat: 沿 dim 拼接(其余维必须同形)
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_cat_operator;
const Operator g_cat_operator = {
    "cat",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补(按输入区间切片回传)
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::cat(const std::vector<Tensor>& tensors, int64_t dim) {
    ZNN_TODO();
}

Tensor Tensor::cat(const Tensor& a, const Tensor& b, int64_t dim) {
    return cat({a, b}, dim);
}

} // namespace znn
