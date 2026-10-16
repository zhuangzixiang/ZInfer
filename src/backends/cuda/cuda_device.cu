// ============================================================================
// cuda_device.cu — CUDA 后端骨架(里程碑 2 起点)
//
// 职责: 设备内存与拷贝(经 cuda_backend 接口暴露) + 第一个示范内核(add)。
// 编译条件: CMake ZNN_CUDA=ON 时替代 cuda_stub.cpp 进入构建。
// 扩展路径(DESIGN §5.2): 新算子 = operators/foo.cu 填 kernels[CUDA] 槽位。
// ============================================================================
#include <cuda_runtime.h>
#include <stdexcept>

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
    return true;
}

void* alloc(size_t bytes) {
    void* p = nullptr;
    check(cudaMalloc(&p, bytes), "cudaMalloc");
    return p;
}

void free(void* p) {
    cudaFree(p);                       // 析构路径不抛异常
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
    check(cudaMemset(p, 0, bytes), "memset0");
}

} // namespace cuda_backend
} // namespace znn
