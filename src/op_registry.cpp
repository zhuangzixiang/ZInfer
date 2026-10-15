// op_registry.cpp — 注册表实现(查找为结构性实现, 派发链为占位)
#include "op_registry.hpp"
#include "znn_internal.hpp"
#include <unordered_map>

namespace znn {

OpRegistry& OpRegistry::instance() {
    static OpRegistry registry;
    return registry;
}

// 从注册数据(operator_list)构建只读查找表
OpRegistry::OpRegistry() {
    for (const OpDef* const* it = g_znn_op_list; *it != nullptr; ++it) {
        table_.emplace((*it)->name, *it);
    }
}

const OpDef* OpRegistry::find(const char* name) const {
    auto it = table_.find(name);
    return it == table_.end() ? nullptr : it->second;
}

Tensor OpRegistry::run(const OpDef* op, Span<const Tensor*> inputs, const void* attrs) {
    // TODO: 形状推理 → 输出分配 → 内核选择(空槽兜底链) → autograd 建图
    ZNN_TODO();
}

} // namespace znn
