// ============================================================================
// autograd.hpp — 可微计算图(TensorTape)
//
// 职责: 前向时为参与训练的算子记录 TensorTape —— "本张量作为算子输出
// 被生产"这一事件, 挂在输出张量的 impl 上(叶子张量无 Tape)。inputs 持
// 句柄 → 天然链接上游, 图随张量引用计数存亡 —— 无全局 tape。
// 反向: 从损失张量出发 DFS 收集本图(后序=拓扑序), 逆序执行各算子
// grad 规则, 把梯度累加进各输入 .grad()。
// 链路位置: OpDispatcher::run 在内核执行后调用 record_tape 建图;
// Tensor::backward() 调用 run_backward 执行反向。
// ============================================================================
#pragma once
#include "operator.hpp"
#include <memory>
#include <vector>

namespace znn {

// 归属单个张量的计算记录: 算子 + 输入句柄 + 参数副本
struct TensorTape {
    const Operator*     op = nullptr;    // 反向规则入口(叶子为 null)
    std::vector<Tensor> inputs;          // 前向输入(句柄共享 → 链接上游 Tape)
    std::vector<char>   options;         // 前向选项副本(op->options_size>0 时)

    // 挂载/读取(znn.hpp 中 friend 授权访问 Tensor 私有 impl_)
    static void        attach(Tensor& t, std::unique_ptr<TensorTape> node);
    static TensorTape* of(const Tensor& t);
};

// 建图: 有 grad 规则且任一输入 requires_grad 时由 OpDispatcher 调用;
// options 为调用方栈上指针, 按 op->options_size 深拷贝进 Tape
void record_tape(Tensor& out, const Operator* op, Span<const Tensor*> inputs,
                 const void* options = nullptr);

// 反向: 从标量损失出发, 逆拓扑序填充各参数 .grad()
// (图随张量存亡 —— 张量在图在, 天然支持二次 backward, 梯度继续累加)
void run_backward(const Tensor& loss);

} // namespace znn
