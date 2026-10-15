// linear: x @ w.T + b —— v1 前端经 matmul/add 组合实现(梯度由组件算子
// 自动获得); OpDef 保留给未来 Eigen/cuBLASLt 融合内核
#include "shared.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_linear_operator;

const Operator g_linear_operator = {
    "linear",
    nullptr,   // 前端组合实现, 不经派发
    nullptr,
    nullptr,
    {},        // 融合内核槽位(未来)
};

Tensor Tensor::linear(const Tensor& x, const Tensor& w, const Tensor* b) {
    Tensor y = Tensor::matmul(x, w.transpose(0, 1));
    return b ? Tensor::add(y, *b) : y;
}

} // namespace znn
