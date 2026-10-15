// ============================================================================
// op_registry.hpp — 算子注册表与派发
//
// 职责: 注册"机制"——启动时读取注册数据(operators/operator_list.cpp 的
// g_znn_op_list)建立只读查找表; run() 完成一次算子调用的完整链路:
//   形状推理 → 分配输出 → 按输入设备选内核(空槽走兜底链) → (训练时)建图
// 与 operator_list 的关系: 数据在那里, 机制在这里。
//
// 链路位置: 算子前端(operators/*.cpp)唯一调用入口。
// ============================================================================
#pragma once
#include "op_def.hpp"
#include <string>
#include <unordered_map>

namespace znn {

class OpRegistry {
public:
    static OpRegistry& instance();

    const OpDef* find(const char* name) const;      // 调试/测试用
    // 完整派发: 形状推理 → 输出分配 → 内核选择(空槽走兜底链) → 建图
    Tensor run(const OpDef* op, Span<const Tensor*> inputs, const void* attrs);

private:
    OpRegistry();
    std::unordered_map<std::string, const OpDef*> table_;
};

} // namespace znn
