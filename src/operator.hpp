// ============================================================================
// operator.hpp — 算子契约 (算子作者唯一需要阅读的内部头)
//
// 一个算子由三类函数 + 一份注册项构成:
//   ShapeInferenceFunction  形状推导: 只看形状不算数据, 在分配输出前运行
//   KernelFunction          计算内核: 真正做数学的代码, 每个后端一份
//   GradientFunction        反向规则: 反向传播时怎么求对每个输入的梯度
//   Operator                   注册项: 把三者与各后端内核槽位绑在一起
//
// 调用链路(算子在链路中的位置):
//   Tensor::add(a,b) → OpDispatcher::run → infer → 分配输出 → kernels[设备] → (训练时)record
//   loss.backward()  → AutogradEngine  → grad 逐节点逆序执行
//
// 新增算子三步(详见 docs/DESIGN.md §5.1):
//   1. operators/<name>.cpp 实现 Operator 与前端函数
//   2. operators/operator_list.cpp 的 g_operator_list 数组加一行
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
    Span() = default;
    Span(T* ptr, size_t count) : p(ptr), n(count) {}

    T* begin() const { return p; }
    T* end()   const { return p + n; }
    T& operator[](size_t i) const { return p[i]; }
    size_t size() const { return n; }

private:
    T* p = nullptr;
    size_t n = 0;
};

struct TensorMeta
{
    const Shape* shape;
    DType        dtype;
    Device       device;
};

// 计算内核的执行上下文
// Precision 定义在 znn.hpp 公共头(用户 API 需要), 此处直接使用
struct KernelContext
{
    Device device;               // 本次在哪个后端执行
    void*  stream = nullptr;     // 预留: GPU 里程碑填异步流
    Precision precision = Precision::AUTO;  // GEMM 精度(由派发器填充)
};

// 反向规则的执行上下文: 框架备好一切, 规则只负责填充 grad_inputs
struct BackwardContext
{
    Span<const Tensor*> saved_inputs;   // 前向输入(建图时保存, 共享存储不拷贝)
    Span<const Tensor*> saved_outputs;  // 前向输出
    Span<const Tensor*> grad_outputs;   // 上游传来的"对本算子输出的梯度"
    Span<Tensor*>       grad_inputs;    // 待写入:"对各输入的梯度"(框架已分配好)
    const void*         options;        // 前向调用选项副本(引擎保存, 见 Operator::options_size)

    // 前向发生过广播时(如 [B,C]+[C]), 把梯度沿广播维求和规约回输入形状
    static Tensor reduce_to(const Tensor& grad, const Shape& target_shape);
};

// 形状推导: 推导每个输出的shape(不碰具体数据)!OpDispatcher::run 在分配输出缓冲之前调用
using ShapeInferenceFunction = void (*)(Span<const TensorMeta*> inputs, const void* options, std::vector<Shape>& outputs);
// 计算内核: 真正做数学的函数! run 分配输出之后经函数指针调用;
// 数据几何经 Tensor 公共访问器获取(data<T>/strides/shape/device),
// inputs 只读约定, outputs 已分配只管写
using KernelFunction = void (*)(KernelContext& context, Span<const Tensor*> inputs, Span<Tensor*> outputs, const void* options);
// 反向规则: 求出“对每个输入的梯度”! loss.backward() 时按逆拓扑序逐节点调用
using GradientFunction = void (*)(BackwardContext& context);

// 一个算子的完整注册项: 一个算子 = "1 个形状推导 + N 份内核(每后端一份) + 可选 1 个反向规则"
struct Operator
{
    const char* name;  // 算子名(全局唯一)
    ShapeInferenceFunction infer;  // 形状推导函数, 必填
    GradientFunction grad;  // 反向规则函数(nullptr: 仅推理, 不求导)

    KernelFunction ref;  // CPU 朴素实现(某后端没有快速内核时保证功能可用)
    KernelFunction kernels[(int)Device::kDeviceCount];  // 具体的后端快速内核(kernels[CPU] 由 Eigen 版填充)
    // 选项结构体字节数: >0 时引擎把调用方栈上的 options 拷贝进图节点,
    // 反向时经 BackwardContext::options 交还(梯度规则可读前向选项)
    size_t options_size = 0;
};

extern const Operator* g_operator_list[];

} // namespace znn
