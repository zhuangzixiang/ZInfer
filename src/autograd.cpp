// autograd.cpp — 自动求导引擎实现(骨架阶段为占位)
#include "autograd.hpp"
#include "znn_internal.hpp"

namespace znn {

AutogradEngine& AutogradEngine::instance() {
    static AutogradEngine engine;
    return engine;
}

GraphNode* AutogradEngine::record(const OpDef* op,
                                  Span<const Tensor*> inputs,
                                  Span<const Tensor*> outputs) {
    // TODO: 节点池分配 → 保存输入输出 → 链接上下游消费计数
    ZNN_TODO();
}

void AutogradEngine::backward(const Tensor& loss) {
    // TODO: 逆拓扑遍历, pending 归零触发, 梯度缓冲合并, 广播规约
    ZNN_TODO();
}

Tensor BackwardContext::reduce_to(const Tensor& g, const Shape& target_shape) {
    // TODO: 按广播语义对 g 求和规约回 target_shape
    ZNN_TODO();
}

} // namespace znn
