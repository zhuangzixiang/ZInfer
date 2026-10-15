// ============================================================================
// op_def.hpp — 算子契约 (算子作者唯一需要阅读的内部头)
//
// 本文件定义"一个算子是什么"与"如何注册":
//   OpDef                  算子注册项: 名字/形状推导/反向规则/各后端内核槽位
//   KernelFn InferFn GradFn 三类函数指针(后端无关的算子作者接口)
//   KernelCtx BackwardCtx TensorMeta 内核与反向规则的调用上下文
//   ZNN_OP_* 宏           注册到 operator_list 的方式
//
// 算子在调用链路中的位置:
//   Tensor::add(a,b) → OpRegistry::run → 按 OpDef.kernels[设备] 执行内核
//   loss.backward()  → AutogradEngine  → 按 OpDef.grad 执行反向规则
//
// 新增算子三步(详见 docs/DESIGN.md §5.1):
//   1. operators/<name>.cpp 实现 OpDef 与前端
//   2. operators/operator_list.cpp: extern 与 ZNN_OP 各加一行
//   3. CMake 的 ZNN_OPERATORS 清单加一行
// ============================================================================
#pragma once
#include "znn.hpp"
#include <cstddef>
#include <vector>

namespace znn {

// ---------------------------------------------------------------- 轻量视图
// 算子接口专用, 不进入用户 API。

template <class T>
struct Span {
    T*     p = nullptr;
    size_t n = 0;
    T* begin() const { return p; }
    T* end()   const { return p + n; }
    T& operator[](size_t i) const { return p[i]; }
    size_t size() const { return n; }
};

// ---------------------------------------------------------------- 调用上下文

// 调用内核时各输入的元信息(不含数据指针)
struct TensorMeta {
    const Shape* shape;
    DType        dtype;
    Device       device;
};

// 内核执行上下文
struct KernelCtx {
    Device device;
    void*  stream = nullptr;   // 预留, v1 恒为 nullptr
};

// 反向传播上下文: 框架已分配好 grad_in, 规则只负责填充
struct BackwardCtx {
    Span<const Tensor* const*> saved_in;    // 前向输入(默认全部保存)
    Span<const Tensor* const*> saved_out;   // 前向输出(默认全部保存)
    Span<const Tensor* const*> grad_out;    // 各输出收到的梯度
    Span<Tensor* const*>       grad_in;     // 待写入: 各输入的梯度

    // 工具: 前向发生过广播时, 把梯度规约回输入形状
    static Tensor reduce_to(const Tensor& g, const Shape& target_shape);
};

// ---------------------------------------------------------------- 算子接口

// attrs 指向该算子的强类型选项 struct(如 Conv2dOptions), 无选项算子为 nullptr
using KernelFn = void (*)(KernelCtx& ctx,
                          Span<const Tensor*> ins,
                          Span<Tensor*>       outs,
                          const void*         attrs);

// 形状推导: 填写每个输出的 shape(dtype/device 由框架按输入推导)
using InferFn = void (*)(Span<const TensorMeta*> ins,
                         const void* attrs,
                         std::vector<Shape>& outs);

// 反向规则: nullptr = 该算子仅支持推理, 参与训练时报错
using GradFn = void (*)(BackwardCtx& ctx);

// ---------------------------------------------------------------- OpDef
// 一个算子的完整注册项 —— 语义与各后端内核集中一处(单注册表模型)。

struct OpDef {
    const char* name;                       // 全局唯一
    InferFn     infer;
    GradFn      grad;
    KernelFn    ref;                        // CPU 朴素实现 = 所有后端的对拍基准
    KernelFn    kernels[(int)Device::kDeviceCount];
};

// ---------------------------------------------------------------- 注册
// operators/operator_list.cpp (≈ ffmpeg codec_list.c), 整个宏块置于 namespace znn 内:

#define ZNN_OP_LIST_BEGIN \
    const OpDef* g_znn_op_list[] = {
#define ZNN_OP(def)  &def,
#define ZNN_OP_LIST_END \
    nullptr };

// 注册表(OpRegistry)启动时读取; 以 nullptr 结尾
extern const OpDef* g_znn_op_list[];

} // namespace znn
