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
//   Tensor::add(a,b) → OpDispatcher::run → infer → 分配输出 → kernels[设备] → (训练时)record
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

namespace znn
{
/*** 函数指针需要参数固定, 可以使Tensor** p, int n 两个分离参数或者vector, span直接绑定了指针和数量, 同时不存在vector的"拥有"参数的含义 ***/
template <class T>
struct Span
{
public:
    T* begin() const { return p; }
    T* end()   const { return p + n; }
    T& operator[](size_t i) const { return p[i]; }
    size_t size() const { return n; }

private:
    T* p = nullptr;
    size_t n = 0;
};

// --------------------------------------------- 三类函数各自的调用上下文

// 张量"名片": 只有形状/类型/设备, 无数据指针。
// 专供形状推导——推导不该碰数据, 结构上就杜绝。
struct TensorMeta
{
    const Shape* shape;
    DType        dtype;
    Device       device;
};

// 计算内核的执行上下文
struct KernelContext
{
    Device device;              // 本次在哪个后端执行(委托型内核可多后端复用)
    void*  stream = nullptr;    // 预留: GPU 里程碑填异步流, v1 恒为 nullptr
};

// 反向规则的执行上下文: 框架备好一切, 规则只负责填充 grad_inputs
struct BackwardContext
{
    Span<const Tensor* const*> saved_inputs;   // 前向输入(建图时保存, 共享存储不拷贝)
    Span<const Tensor* const*> saved_outputs;  // 前向输出
    Span<const Tensor* const*> grad_outputs;   // 上游传来的"对本算子输出的梯度"
    Span<Tensor* const*>       grad_inputs;    // 待写入:"对各输入的梯度"(框架已分配好)

    // 前向发生过广播时(如 [B,C]+[C]), 把梯度沿广播维求和规约回输入形状
    static Tensor reduce_to(const Tensor& grad, const Shape& target_shape);
};

// 形状推导: 推导每个输出的shape(不碰具体数据)!OpDispatcher::run 在分配输出缓冲之前调用
using ShapeInferenceFunction = void (*)(Span<const TensorMeta*> inputs, const void* options, std::vector<Shape>& outputs);
// 计算内核: 真正做数学的函数! run 分配输出之后经函数指针调用
using KernelFunction = void (*)(KernelContext& context, Span<const Tensor*> inputs, Span<Tensor*> outputs, const void* options);
// 反向规则: 求出“对每个输入的梯度”! loss.backward() 时按逆拓扑序逐节点调用
using GradientFunction = void (*)(BackwardContext& context);

// 一个算子的完整注册项: 一个算子 = "1 个形状推导 + N 份内核(每后端一份) + 可选 1 个反向规则"
struct OpDef
{
    const char* name;  // 算子名(全局唯一)
    ShapeInferenceFunction infer;  // 形状推导函数, 必填
    GradientFunction grad;  // 反向规则函数(nullptr: 仅推理, 不求导)

    KernelFunction ref;  // CPU 朴素实现(某后端没有快速内核时保证功能可用)
    KernelFunction kernels[(int)Device::kDeviceCount];  // 具体的后端快速内核(kernels[CPU] 由 Eigen 版填充)
};

// --------------------------------------------- 注册数据
// operators/operator_list.cpp 直接以普通数组定义(无宏);
// OpDispatcher 首次使用时读取建索引; 以 nullptr 结尾。
extern const OpDef* g_znn_op_list[];

} // namespace znn
