// ============================================================================
// op_def.hpp — 算子契约 (算子作者唯一需要阅读的内部头)
//
// 一个算子由三类函数 + 一份注册项构成:
//   ShapeInferenceFunction  形状推导: 只看形状不算数据, 在分配输出前运行
//   KernelFunction          计算内核: 真正做数学的代码, 每个后端一份
//   GradientFunction        反向规则: 反向传播时怎么求对每个输入的梯度
//   OpDef                   注册项: 把三者与各后端内核槽位绑在一起
//
// 调用链路(算子在链路中的位置):
//   Tensor::add(a,b) → OpRegistry::run → infer → 分配输出 → kernels[设备] → (训练时)record
//   loss.backward()  → AutogradEngine  → grad 逐节点逆序执行
//
// 新增算子三步(详见 docs/DESIGN.md §5.1):
//   1. operators/<name>.cpp 实现 OpDef 与前端函数
//   2. operators/operator_list.cpp 的 g_znn_op_list 数组加一行
//   3. CMake 的 ZNN_OPERATORS 清单加一行
// ============================================================================
#pragma once
#include "znn.hpp"
#include <cstddef>
#include <vector>

namespace znn {

// ---------------------------------------------------------------- Span 轻量视图
// "指针 + 长度"的视图: 内核收"一组张量"时零拷贝零分配地传参。
// 不拥有内存(指向调用方的栈数组), 用完即弃; C++20 std::span 的 C++17 替身。

template <class T>
struct Span {
    T*     p = nullptr;
    size_t n = 0;
    T* begin() const { return p; }
    T* end()   const { return p + n; }
    T& operator[](size_t i) const { return p[i]; }
    size_t size() const { return n; }
};

// --------------------------------------------- 三类函数各自的调用上下文

// 张量"名片": 只有形状/类型/设备, 无数据指针。
// 专供形状推导——推导不该碰数据, 结构上就杜绝。
struct TensorMeta {
    const Shape* shape;
    DType        dtype;
    Device       device;
};

// 计算内核的执行上下文
struct KernelContext {
    Device device;              // 本次在哪个后端执行(委托型内核可多后端复用)
    void*  stream = nullptr;    // 预留: GPU 里程碑填异步流, v1 恒为 nullptr
};

// 反向规则的执行上下文: 框架备好一切, 规则只负责填充 grad_inputs
struct BackwardContext {
    Span<const Tensor* const*> saved_inputs;   // 前向输入(建图时保存, 共享存储不拷贝)
    Span<const Tensor* const*> saved_outputs;  // 前向输出
    Span<const Tensor* const*> grad_outputs;   // 上游传来的"对本算子输出的梯度"
    Span<Tensor* const*>       grad_inputs;    // 待写入:"对各输入的梯度"(框架已分配好)

    // 前向发生过广播时(如 [B,C]+[C]), 把梯度沿广播维求和规约回输入形状
    static Tensor reduce_to(const Tensor& grad, const Shape& target_shape);
};

// --------------------------------------------- 三类算子作者函数

// 形状推导: 输入名片 + 选项 → 填写每个输出的 shape。
// 时机: OpRegistry::run 在分配输出缓冲之前调用——不知道输出大小就无法分配。
// 示例: matmul 校验 a[-1]==b[0], 推出输出 shape {a[0], b[1]}。
using ShapeInferenceFunction = void (*)(Span<const TensorMeta*> inputs,
                                        const void* options,
                                        std::vector<Shape>& outputs);

// 计算内核: 真正做数学的函数。
// inputs 只读; outputs 已按 infer 的结果分配好, 只管写;
// options 指向本算子的强类型选项 struct(如 Conv2dOptions), 无选项算子为 nullptr。
// 时机: run 分配输出之后经函数指针调用; 每个后端一份, 按输入设备选择。
using KernelFunction = void (*)(KernelContext& context,
                                Span<const Tensor*> inputs,
                                Span<Tensor*>       outputs,
                                const void*         options);

// 反向规则: 拿前向保存的张量与上游梯度, 算出对每个输入的梯度并写入 grad_inputs。
// 时机: loss.backward() 时由 AutogradEngine 按逆拓扑序逐节点调用。
// 示例: relu 的规则 = grad_out * (前向输入 > 0)。
using GradientFunction = void (*)(BackwardContext& context);

// --------------------------------------------- OpDef 注册项

// 一个算子的完整注册项: 语义与各后端内核集中一处(单注册表模型)。
struct OpDef {
    // 算子名, 全局唯一。OpRegistry 以它建哈希键(find 调试用);
    // 正式调用链不用名字查找——前端直接持有本结构体的常量指针, 零开销。
    const char* name;

    // 形状推导函数, 必填。作用与时机见 ShapeInferenceFunction 注释。
    ShapeInferenceFunction infer;

    // 反向规则函数。nullptr = 仅推理算子(如 randn 无需求导), 参与训练时报错。
    GradientFunction grad;

    // CPU 朴素实现(参考内核), 必填。两个用途:
    //   1) 对拍基准——所有后端内核与它比对数值正确性(测试体系的地基)
    //   2) 兜底——某后端没有快速内核时保证功能可用
    KernelFunction ref;

    // 各后端快速内核槽位, 数组下标 = Device 枚举值(CPU=0/CUDA=1/Metal=2/Vulkan=3)。
    // kernels[CPU] 由 Eigen 版填充; 空槽走兜底链: ref / 透明拷回 CPU 执行+warning。
    KernelFunction kernels[(int)Device::kDeviceCount];
};

// --------------------------------------------- 注册数据
// operators/operator_list.cpp 直接以普通数组定义(无宏);
// OpRegistry 启动时读取建表; 以 nullptr 结尾。
extern const OpDef* g_znn_op_list[];

} // namespace znn
