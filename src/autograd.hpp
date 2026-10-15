// ============================================================================
// autograd.hpp — 自动求导引擎
//
// 职责: 前向时为参与训练的算子记录计算图(tape), 反向时按拓扑序执行各
// 算子的 grad 规则, 把梯度写进各参数的 .grad()。
// 实现范围: 节点池分配 / 图所有权 / 梯度缓冲合并(等价 PyTorch InputBuffer)
// / 广播梯度规约(BackwardContext::reduce_to) / NoGrad 感知。
//
// 设计: tape 由引擎自持——张量本体不携带任何图信息(无 grad_fn 字段),
// 引擎经公共接口工作(requires_grad() 判断 / grad() 可写引用),
// 反向路由靠"张量实体→生产节点"映射表, 因此 Tensor 无需 friend 授权。
//
// 链路位置: OpDispatcher::run 在内核执行后调用 record 建图;
// Tensor::backward() 调用 backward 执行反向。
// ============================================================================
#pragma once
#include "operator.hpp"
#include <unordered_map>
#include <vector>

namespace znn {

// 前向建图节点: 创建序即拓扑序(动态图天然性质, 反向免排序)
struct GraphNode {
    const Operator* op = nullptr;            // 反向规则入口
    std::vector<Tensor> saved_inputs;        // 前向输入(默认全存)
    std::vector<Tensor> saved_outputs;       // 前向输出(默认全存)
    std::vector<GraphNode*> consumers;       // 消费本节点输出的下游节点
    int pending = 0;                         // 尚未到齐的上游梯度数
};

class AutogradEngine {
public:
    static AutogradEngine& instance();

    // 前向: 有 grad 规则且任一输入 requires_grad 时由 OpDispatcher 调用
    GraphNode* record(const Operator* op,
                      Span<const Tensor*> inputs,
                      Span<const Tensor*> outputs);
    // 反向: 从标量损失出发, 逆拓扑序填充各参数 .grad()
    void backward(const Tensor& loss);

private:
    AutogradEngine() = default;

    std::vector<GraphNode*> tape_;                    // 建图序 = 拓扑序
    std::unordered_map<const void*, GraphNode*> producer_;  // 张量实体 → 生产节点
};

} // namespace znn
