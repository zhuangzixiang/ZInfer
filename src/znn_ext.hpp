// ============================================================================
// znn_ext.hpp — ZInfer 扩展接口 (v0.7)
//
// 受众: 算子/后端作者(库内开发), 不随库发布, 不在 include/ 中。
// 库的使用者只需要 include/znn.hpp。
//
// 新增算子三步 (零改内核, 详见 docs/DESIGN.md §5.1):
//   1. operators/<name>.cpp: 定义 OpDef(infer 必填/grad 可空/ref 必填/
//      kernels 按后端填, 空槽走兜底链) + 前端函数
//   2. operators/operator_list.cpp: extern 声明 + ZNN_OP 列表各加一行
//   3. CMake 源清单加一行
// ============================================================================
#pragma once
#include "znn.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>

namespace znn {

// 未实现占位: 骨架阶段统一抛出, 待逐项填充真实逻辑
#define ZNN_TODO() throw std::runtime_error(std::string("znn: TODO ") + __func__)

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

// ---------------------------------------------------------------- 算子上下文

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

// ---------------------------------------------------------------- 函数指针

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

// 注册表(OpRegistry, 见 znn_internal.hpp)启动时读取; 以 nullptr 结尾
extern const OpDef* g_znn_op_list[];

} // namespace znn
