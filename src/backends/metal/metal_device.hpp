// ============================================================================
// metal_device.hpp — Metal 设备层跨编译单元声明(仅 .mm 内部包含)
//
// metal_kernels.mm 经 device:: 派发; 本头含 MTL 类型, 纯 C++ 编译单元
// (经 backend_slots.hpp 间接包含的 metal_backend.hpp)不得包含本文件。
// 实现: metal_device.mm
// ============================================================================
#pragma once
#include <cstdint>

#import <Metal/Metal.h>

namespace znn {
namespace metal_backend {
namespace device {

id<MTLBuffer> buf_of(void* p);                        // Storage 句柄 → MTLBuffer
void* mapped_of(void* p);                            // 句柄 → 共享内存映射指针
id<MTLComputePipelineState> pipeline(const char* msl, // MSL 源码运行时编译
                                     const char* name);
// 句柄+字节偏移成对(视图); 偏移在 setBuffer 折叠, 句柄永不算术
struct MBuf { id<MTLBuffer> buf; uint64_t off; };

// grid=总线程数, tg=每线程组线程数(dispatchThreads 语义); 绑定 0/1/2=a/b/y,
// push 常量走 buffer(3); config 可追加设置(无则传 nil)
void dispatch(id<MTLComputePipelineState> p, MBuf a, MBuf b, MBuf y,
              const void* push, size_t push_size,
              MTLSize grid, MTLSize tg,
              void (^config)(id<MTLComputeCommandEncoder>));
// N 缓冲派发(多输入/优化器内核): 缓冲绑 0..n-1, push 走 buffer(8)
void dispatch_bufs(id<MTLComputePipelineState> p, const MBuf* bufs,
                   int nbuf, const void* push, size_t push_size,
                   MTLSize grid, MTLSize tg);
// 带 threadgroup 内存(归约内核用): 分配 tg0/tg1 两块
void dispatch_bufs_tg(id<MTLComputePipelineState> p, const MBuf* bufs,
                      int nbuf, const void* push, size_t push_size,
                      MTLSize grid, MTLSize tg,
                      NSUInteger tg0_bytes, NSUInteger tg1_bytes);

} // namespace device
} // namespace metal_backend
} // namespace znn
