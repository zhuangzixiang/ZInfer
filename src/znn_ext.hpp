// ============================================================================
// znn_ext.hpp — ZInfer 扩展接口 (v0.2 讨论稿)
//
// 受众: 算子/后端作者(库内开发), 不随库发布, 不在 include/ 中。
// 库的使用者只需要 include/znn.hpp。
//
// 新增算子四步 (零改 core, 详见 docs/DESIGN.md §5.1):
//   1. 实现一个 OpDef: infer 必填(形状推导), grad 可空(=仅推理),
//      ref 必填(CPU 朴素实现, 所有后端的 对拍基准),
//      kernels[] 按后端填, 空槽走兜底链 [D9: 透明拷回 CPU 执行+warning]
//   2. 内核物理上可分属 .cpp/.cu/.mm/.comp 翻译单元, 注册项 extern 引用
//   3. OpDef 加入 operator_list (ZNN_OP 宏, 或 CMake 扫描自动生成)
//   4. 构建脚本加入源文件
// ============================================================================
#pragma once
#include "znn.hpp"          // include/ 在头文件搜索路径上
#include <cstddef>

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
    void*  stream = nullptr;   // [D8] 预留, v1 恒为 nullptr
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
    KernelFn    ref;                        // CPU 朴素实现 = 对拍基准
    KernelFn    kernels[(int)Device::kDeviceCount];
};

// ---------------------------------------------------------------- 注册
// src/reg/operator_list.cpp (≈ ffmpeg codec_list.c):
//
//   ZNN_OP_LIST_BEGIN
//       ZNN_OP(g_relu_def)
//       ZNN_OP(g_matmul_def)
//   ZNN_OP_LIST_END

#define ZNN_OP_LIST_BEGIN \
    const znn::OpDef* const g_znn_op_list[] = {
#define ZNN_OP(def)  &def,
#define ZNN_OP_LIST_END \
    nullptr };

// 测试/调试用: 按名字查注册项; nullptr = 未注册
const OpDef* find_op(const char* name);

} // namespace znn
