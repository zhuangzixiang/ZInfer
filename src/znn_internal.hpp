// ============================================================================
// znn_internal.hpp — 公共头 znn.hpp 的实现支撑
//
// 职责: Storage(引用计数设备缓冲) / Tensor::Impl(张量实体, 即 Tensor::Impl)
// / mut 辅助 / 全库占位宏。Tensor 公共方法与 Runtime 的实现均在
// znn_internal.cpp。
// 本文件不含任何"引擎": 算子契约在 op_def.hpp, 注册机制在 op_registry,
// 求导引擎在 autograd, 层/优化器在各自目录。
// ============================================================================
#pragma once
#include "znn.hpp"
#include <atomic>
#include <stdexcept>
#include <string>

namespace znn {

// 未实现占位: 骨架阶段统一抛出, 待逐项填充真实逻辑
#define ZNN_TODO() throw std::runtime_error(std::string("znn: TODO ") + __func__)

// ============================================================ 设备内存

// 引用计数设备缓冲; Tensor::Impl 以 offset+stride 视图化共享同一 Storage
class Storage {
public:
    static Storage* create(size_t bytes, Device device);   // 唯一构造入口
    Storage* retain();
    void     release();            // 引用归零时释放内存并自毁

    void*   data()   const { return data_; }
    size_t  bytes()  const { return bytes_; }
    Device  device() const { return device_; }

private:
    Storage(size_t bytes, Device device);
    ~Storage();

    std::atomic<int64_t> refcount_;
    void*   data_;
    size_t  bytes_;
    Device  device_;
};

// ============================================================ Tensor 内部

struct GraphNode;                 // 定义在 autograd.hpp

// 公共 Tensor 的唯一实体: 存储视图 + 求导元数据
struct Tensor::Impl {
    Storage*   storage = nullptr;     // null = 空句柄
    int64_t    offset  = 0;           // 元素级偏移(视图)
    Shape      shape;
    int64_t    strides[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    DType      dtype  = DType::F32;
    Device     device = Device::CPU;

    bool       requires_grad = false;
    Impl*      grad    = nullptr;     // 惰性分配, 与本 Impl 同形状
    GraphNode* grad_fn = nullptr;     // 不拥有; 由 AutogradEngine 节点池管理

    static Impl* create(const Shape& shape, DType dtype, Device device);
    Impl*        retain();
    void         release();

    bool   is_contiguous() const;
    size_t nbytes() const;            // numel × 元素大小
};

// 层对象逻辑上可变(参数被 Optimizer 更新), const 方法内取可变参数指针
Tensor* mut(const Tensor& t);

} // namespace znn
