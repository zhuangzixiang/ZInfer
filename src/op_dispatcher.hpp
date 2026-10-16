// ============================================================================
// op_dispatcher.hpp — 算子派发器
//
// 先澄清"注册"发生在哪——注册是编译/链接期的静态动作, 共三步:
//   1. operators/<name>.cpp 定义 Operator 常量(算子出生)
//   2. operator_list.cpp 把其地址登记进 g_operator_list[](= 注册)
//   3. CMake 把源文件纳入构建
// 运行期没有任何注册动作; 本类首次使用时只做一次"建索引"(遍历名单
// 建 name→Operator 哈希表), 之后只读。
//
// 职责:
//   run()   派发执行一次算子调用(主职): 形状推理 → 分配输出 → 按输入
//           设备选内核(空槽走兜底链) → 执行 → (训练时)建图。
//           前端直接持 Operator 常量指针, 不经过任何查找。
//   find()  按名字查 Operator: 调试/测试断言; 以及未来的模型文件加载
//           (ONNX/GGUF 存的是字符串算子名, 加载时经此找回 Operator)。
//
// 链路位置: 29 个算子前端(operators/*.cpp)的唯一后端入口。
// ============================================================================
#pragma once
#include "operator.hpp"
#include <string>
#include <unordered_map>

namespace znn {

class OpDispatcher {
public:
    static OpDispatcher& instance();

    // 精度配置(Runtime::set_precision 委托到此处; 成员变量, 非全局)
    void set_precision(Precision p);
    Precision precision() const;

    // 按名字查算子(nullptr = 未登记); 字符串驱动场景用
    const Operator* find(const char* name) const;

    // 派发执行一次算子调用(完整链路见文件头)
    Tensor run(const Operator* op, Span<const Tensor*> inputs, const void* options);

private:
    OpDispatcher();     // 遍历 g_operator_list 建 name→Operator 索引
    std::unordered_map<std::string, const Operator*> table_;
    Precision precision_ = Precision::AUTO;
};

} // namespace znn
