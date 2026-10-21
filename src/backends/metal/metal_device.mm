// ============================================================================
// metal_device.mm — Metal 设备与内存基础设施(Apple 平台, Objective-C++)
//
// 链路: MTLCreateSystemDefaultDevice → 命令队列 → 共享内存缓冲(MTLStorageModeShared,
// CPU/GPU 均可直接访问, 免 staging —— Apple 统一内存架构天然支持)。
// 注: 本文件仅在 APPLE 平台被 CMake 选用(ZNN_METAL=ON)。
// ============================================================================
#include "backends/metal/metal_backend.hpp"
#include "backends/metal/metal_device.hpp"

#if defined(__APPLE__) && defined(ZNN_METAL)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <stdexcept>
#include <string>

namespace znn {
namespace metal_backend {

namespace {

// 进程级设备状态(Metal 对象天然进程级; 与 cuBLAS handle 同生命周期语义)
struct MtlState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    bool ok = false;
};
MtlState* st() {
    static MtlState s;
    return &s;
}

bool init_metal() {
    MtlState* v = st();
    if (v->ok) return true;
    v->device = MTLCreateSystemDefaultDevice();
    if (!v->device) return false;
    v->queue = [v->device newCommandQueue];
    v->ok = v->queue != nil;
    return v->ok;
}

// 缓冲句柄: MTLBuffer + 派生信息
struct MtlBuf {
    id<MTLBuffer> buffer;
    void* mapped;           // 共享内存: CPU 可直接访问
    size_t cap;
};

} // namespace

bool available() {
    static const bool ok = init_metal();
    return ok;
}

void* alloc(size_t bytes) {
    if (!st()->ok || bytes == 0) return nullptr;
    id<MTLBuffer> buf = [st()->device newBufferWithLength:bytes
                                                  options:MTLStorageModeShared];
    if (!buf) return nullptr;
    return new MtlBuf{buf, buf.contents, bytes};
}

void free(void* p) {
    if (!p) return;
    delete static_cast<MtlBuf*>(p);      // ARC 释放 MTLBuffer
}

void copy_host_to_device(void* dst, const void* src, size_t bytes) {
    memcpy(static_cast<MtlBuf*>(dst)->mapped, src, bytes);   // 共享内存直拷
}
void copy_device_to_host(void* dst, const void* src, size_t bytes) {
    memcpy(dst, static_cast<const MtlBuf*>(src)->mapped, bytes);
}

void copy_device_to_host_off(void* dst, const void* src, size_t soff,
                             size_t bytes) {
    memcpy(dst, static_cast<const char*>(static_cast<const MtlBuf*>(src)->mapped)
               + soff, bytes);
}

void copy_device_to_device(void* dst, const void* src, size_t bytes) {
    MtlState* v = st();
    id<MTLCommandBuffer> cb = [v->queue commandBuffer];
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    MtlBuf* d = static_cast<MtlBuf*>(dst);
    MtlBuf* s = static_cast<MtlBuf*>(const_cast<void*>(src));
    [enc copyFromBuffer:s->buffer sourceOffset:0
                 toBuffer:d->buffer destinationOffset:0 size:bytes];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void memset0(void* p, size_t bytes) {
    memset(static_cast<MtlBuf*>(p)->mapped, 0, bytes);       // 共享内存
}

// ---- 计算派发入口(内核文件使用; 声明见 metal_device.hpp) ----
namespace device {

id<MTLBuffer> buf_of(void* p) { return static_cast<MtlBuf*>(p)->buffer; }
void* mapped_of(void* p) { return static_cast<MtlBuf*>(p)->mapped; }

// MSL 源码 → 管线(驱动运行时编译; Metal 官方路径, 无离线工具)
id<MTLComputePipelineState> pipeline(const char* msl, const char* name) {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:msl];
    id<MTLLibrary> lib = [st()->device newLibraryWithSource:src
                                                    options:nil
                                                      error:&err];
    if (!lib) {
        throw std::runtime_error(std::string("znn metal: MSL compile failed: ")
                                 + err.localizedDescription.UTF8String);
    }
    id<MTLFunction> fn = [lib newFunctionWithName:
                                  [NSString stringWithUTF8String:name]];
    if (!fn) {
        throw std::runtime_error("znn metal: kernel function not found");
    }
    NSError* perr = nil;
    id<MTLComputePipelineState> p =
        [st()->device newComputePipelineStateWithFunction:fn error:&perr];
    if (!p) {
        throw std::runtime_error(std::string("znn metal: pipeline failed: ")
                                 + perr.localizedDescription.UTF8String);
    }
    return p;
}

void dispatch(id<MTLComputePipelineState> p, MBuf a, MBuf b, MBuf y,
              const void* push, size_t push_size,
              MTLSize grid, MTLSize tg,
              void (^config)(id<MTLComputeCommandEncoder>)) {
    MtlState* v = st();
    id<MTLCommandBuffer> cb = [v->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:a.buf offset:a.off atIndex:0];   // 视图偏移在绑定折叠
    [enc setBuffer:b.buf offset:b.off atIndex:1];
    [enc setBuffer:y.buf offset:y.off atIndex:2];
    if (push) {
        [enc setBytes:push length:push_size atIndex:3];   // 推送常量=缓冲3
    }
    if (config) config(enc);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void dispatch_bufs(id<MTLComputePipelineState> p, const MBuf* bufs,
                   int nbuf, const void* push, size_t push_size,
                   MTLSize grid, MTLSize tg) {
    MtlState* v = st();
    id<MTLCommandBuffer> cb = [v->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    for (int i = 0; i < nbuf; ++i) {
        [enc setBuffer:bufs[i].buf offset:bufs[i].off atIndex:(NSUInteger)i];
    }
    if (push) {
        [enc setBytes:push length:push_size atIndex:8];   // 推送常量=缓冲8
    }
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void dispatch_bufs_tg(id<MTLComputePipelineState> p, const MBuf* bufs,
                      int nbuf, const void* push, size_t push_size,
                      MTLSize grid, MTLSize tg,
                      NSUInteger tg0_bytes, NSUInteger tg1_bytes) {
    MtlState* v = st();
    id<MTLCommandBuffer> cb = [v->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    for (int i = 0; i < nbuf; ++i) {
        [enc setBuffer:bufs[i].buf offset:bufs[i].off atIndex:(NSUInteger)i];
    }
    if (push) {
        [enc setBytes:push length:push_size atIndex:8];
    }
    if (tg0_bytes > 0) [enc setThreadgroupMemoryLength:tg0_bytes atIndex:0];
    if (tg1_bytes > 0) [enc setThreadgroupMemoryLength:tg1_bytes atIndex:1];
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

} // namespace device

} // namespace metal_backend
} // namespace znn

#endif // __APPLE__ && ZNN_METAL
