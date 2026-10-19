// ============================================================================
// znn_internal.hpp — 公共头 znn.hpp 的实现支撑
//
// 职责: Storage(引用计数设备缓冲) / Tensor::Impl(张量实体) / 全库占位宏。
// Tensor 公共方法与 Runtime 的实现均在 znn_internal.cpp。
// 本文件不含任何"引擎": 算子契约在 operator.hpp, 派发器在 op_dispatcher,
// 求导引擎在 autograd, 层/优化器在各自目录。
// ============================================================================
#pragma once
#include "znn.hpp"
#include <atomic>
#include "backends/cuda/cuda_backend.hpp"
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

// 公共 Tensor 的唯一实体: 存储视图。
// 不含任何计算图信息——tape 归 AutogradEngine 所有, 引擎经公共接口工作。
struct Tensor::Impl {
    std::atomic<int64_t> refcount = 1;
    Storage*   storage = nullptr;     // null = 空句柄
    int64_t    offset  = 0;           // 元素级偏移(视图)
    Shape      shape;
    int64_t    strides[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    DType      dtype  = DType::F32;
    Device     device = Device::CPU;

    bool       requires_grad = false;
    Tensor     grad;                  // 惰性分配, grad() 首次访问时创建

    static Impl* create(const Shape& shape, DType dtype, Device device);
    ~Impl();
    static void copy_elements(const Impl* src, Impl* dst);   // dst 连续, src 任意视图
    Impl*        retain();
    void         release();           // 归零时释放 grad/Storage 与自身

    // 共享存储派生视图(新实体, 不影响源张量)
    static Impl* make_view(const Impl* src, const Shape& shape,
                           const int64_t (&strides)[8], int64_t offset);

    bool   is_contiguous() const;
    size_t nbytes() const;            // numel × 元素大小
};

namespace impl {
// 视图(src 可为任意 strided) → dst(连续)的逐元素紧凑拷贝
void copy_view_to(const Tensor& src, Tensor& dst);
}

// ---- 内部共用小工具 ----
size_t dtype_size(DType dtype);       // 元素字节数
void   check_cpu(Device device);      // GPU 里程碑前统一报错

} // namespace znn
