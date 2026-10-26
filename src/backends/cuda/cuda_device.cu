// ============================================================================
// cuda_device.cu — CUDA 后端骨架(里程碑 2 起点)
//
// 职责: 设备内存与拷贝(经 cuda_backend 接口暴露) + 第一个示范内核(add)。
// 编译条件: CMake ZNN_CUDA=ON 时进入构建(未开时该目录不参与编译)。
// 扩展路径(DESIGN §5.2): 新算子 = operators/foo.cu 填 kernels[CUDA] 槽位。
// ============================================================================
#include <cuda_runtime.h>
#include <stdexcept>
#include "backends/cuda/cuda_backend.hpp"

namespace znn {
namespace cuda_backend {

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("znn cuda: ") + what + " failed: "
                                 + cudaGetErrorString(err));
    }
}

// 示范内核: 逐元素 add(z = x + y); 后续算子按此模式填槽位
__global__ void add_kernel(const float* x, const float* y, float* z, long long n) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

} // namespace

bool available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return false;
    }
    cudaSetDevice(0);               // 显式初始化 CUDA 上下文(静态库必需)
    cudaFree(nullptr);             // 触发惰性上下文创建
    // 默认内存池释放阈值设为最大: 空闲块留在池内复用, 免去每次
    // 分配的驱动级开销与 cudaFree 的隐式同步(与 kernel 同流, 有序即安全)
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
        unsigned long long keep = ~0ull;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
    }
    return true;
}

void* alloc(size_t bytes) {
    void* p = nullptr;
    check(cudaMallocAsync(&p, bytes, /*stream=*/nullptr), "cudaMallocAsync");
    return p;
}

void free(void* p) {
    if (p) {
        cudaFreeAsync(p, /*stream=*/nullptr);   // 析构路径不抛异常
    }
}

void copy_host_to_device(void* dst, const void* src, size_t bytes) {
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "h2d copy");
}

void copy_device_to_host(void* dst, const void* src, size_t bytes) {
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "d2h copy");
}

void copy_device_to_device(void* dst, const void* src, size_t bytes) {
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "d2d copy");
}

void memset0(void* p, size_t bytes) {
    check(cudaMemsetAsync(p, 0, bytes, nullptr), "memset0");
}
__global__ void fill_f32_k(float* p,size_t n,float value) {size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;if(i<n)p[i]=value;}
void fill_f32(void* p,size_t n,float value) {if(n){fill_f32_k<<<(unsigned)((n+255)/256),256>>>((float*)p,n,value);check(cudaGetLastError(),"fill_f32");}}

} // namespace cuda_backend
} // namespace znn
