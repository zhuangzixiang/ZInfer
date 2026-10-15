// cuda_stub.cpp — 非CUDA 构建的占位实现(调用即明确报错)
#include "backends/cuda/cuda_backend.hpp"
#include <stdexcept>
#include <string>

namespace znn {
namespace cuda_backend {

static void unavailable() {
    throw std::runtime_error("znn: CUDA backend not built (ZNN_CUDA=OFF)");
}

bool available() { return false; }
void* alloc(size_t) { unavailable(); return nullptr; }
void free(void*) { unavailable(); }
void copy_host_to_device(void*, const void*, size_t) { unavailable(); }
void copy_device_to_host(void*, const void*, size_t) { unavailable(); }

} // namespace cuda_backend
} // namespace znn
