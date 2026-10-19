// ============================================================================
// backend_slots.hpp — 后端内核槽位宏(注册表唯一正确写法)
//
// 编入的后端展开为真实内核指针; 未编入的后端展开为 nullptr,
// 派发器透明回落 CPU(ref 内核)并告警 —— 无 stub 实现, 无占位。
// 源文件禁止直接写 cuda_backend::/vulkan_backend::/metal_backend:: 内核名,
// 一律经 ZNN_*_K()/ZNN_*_CALL() 宏。
// ============================================================================
#pragma once

#include "operator.hpp"
#include <stdexcept>
#include <string>

#if defined(ZNN_CUDA)
  #include "backends/cuda/cuda_backend.hpp"
  #define ZNN_CUDA_K(name)      cuda_backend::name
  #define ZNN_CUDA_CALL(expr)   expr
  #define ZNN_CUDA_CALLV(t, e)  e
  #define ZNN_CUDA_AVAIL()      cuda_backend::available()
#else
  // CALL/CALLV 未编入时丢弃实参表达式(expr 不进入展开, 其中的
  // K(...) 调用形态不会留下非法代码), 换成抛错的占位
  #define ZNN_CUDA_K(name)      nullptr
  #define ZNN_CUDA_CALL(expr)   znn_backend_unavailable<void>("CUDA")
  #define ZNN_CUDA_CALLV(t, e)  znn_backend_unavailable<t>("CUDA")
  #define ZNN_CUDA_AVAIL()      false
#endif

#if defined(ZNN_METAL)
  #include "backends/metal/metal_backend.hpp"
  #define ZNN_METAL_K(name)     metal_backend::name
#else
  #define ZNN_METAL_K(name)     nullptr
  #define ZNN_METAL_AVAIL()     false
#endif
#if defined(ZNN_METAL)
  #define ZNN_METAL_AVAIL()     metal_backend::available()
#else
  #define ZNN_METAL_AVAIL()     false
#endif

#if defined(ZNN_VULKAN)
  #include "backends/vulkan/vulkan_backend.hpp"
  #define ZNN_VULKAN_K(name)    vulkan_backend::name
  #define ZNN_VULKAN_CALL(expr) expr
  #define ZNN_VULKAN_CALLV(t, e) e
  #define ZNN_VULKAN_AVAIL()    vulkan_backend::available()
#else
  #define ZNN_VULKAN_K(name)    nullptr
  #define ZNN_VULKAN_CALL(expr) znn_backend_unavailable<void>("Vulkan")
  #define ZNN_VULKAN_CALLV(t, e) znn_backend_unavailable<t>("Vulkan")
  #define ZNN_VULKAN_AVAIL()    false
#endif

namespace znn {
// 未编入后端的直连调用路径(如梯度内核): 抛错而非静默(设备检查保证
// 未编入时理论不可达, 此为防呆)。返回值模板化 —— 匹配任意调用形态
template <class T>
inline T znn_backend_unavailable(const char* which) {
    throw std::runtime_error(std::string("znn: ") + which
                             + " backend not built");
}
} // namespace znn
