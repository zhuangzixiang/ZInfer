// op_dispatcher.cpp — 派发器实现(索引构建为结构性实现, 派发链为占位)
#include "op_dispatcher.hpp"
#include "znn_internal.hpp"

namespace znn {

OpDispatcher& OpDispatcher::instance() {
    static OpDispatcher dispatcher;
    return dispatcher;
}

// 遍历注册名单(operator_list.cpp)构建只读索引
OpDispatcher::OpDispatcher() {
    for (const OpDef* const* it = g_znn_op_list; *it != nullptr; ++it) {
        table_.emplace((*it)->name, *it);
    }
}

const OpDef* OpDispatcher::find(const char* name) const {
    auto it = table_.find(name);
    return it == table_.end() ? nullptr : it->second;
}

Tensor OpDispatcher::run(const OpDef* op, Span<const Tensor*> inputs, const void* options) {
    // TODO: 形状推理 → 输出分配 → 内核选择(空槽兜底链) → 执行 → autograd 建图
    ZNN_TODO();
}

} // namespace znn
