// cuda_stub.cpp — 非CUDA 构建的占位实现(内存操作明确报错, 内核工厂返回空)
#include "backends/cuda/cuda_backend.hpp"
#include "operator.hpp"
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
void memset0(void*, size_t) { unavailable(); }
void copy_host_to_device(void*, const void*, size_t) { unavailable(); }
void copy_device_to_host(void*, const void*, size_t) { unavailable(); }
void copy_device_to_device(void*, const void*, size_t) { unavailable(); }

// 内核工厂: nullptr → 派发器按兜底链处理(拷回 CPU 执行)
#define ZNN_CUDA_KERNEL_NULL(name) \
    void name(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*) { \
        unavailable(); \
    }
ZNN_CUDA_KERNEL_NULL(add_cuda)
ZNN_CUDA_KERNEL_NULL(sub_cuda)
ZNN_CUDA_KERNEL_NULL(mul_cuda)
ZNN_CUDA_KERNEL_NULL(div_cuda)
ZNN_CUDA_KERNEL_NULL(neg_cuda)
ZNN_CUDA_KERNEL_NULL(exp_cuda)
ZNN_CUDA_KERNEL_NULL(log_cuda)
ZNN_CUDA_KERNEL_NULL(sqrt_cuda)
ZNN_CUDA_KERNEL_NULL(relu_cuda)
ZNN_CUDA_KERNEL_NULL(gelu_cuda)
ZNN_CUDA_KERNEL_NULL(sigmoid_cuda)
ZNN_CUDA_KERNEL_NULL(tanh_cuda)
ZNN_CUDA_KERNEL_NULL(silu_cuda)
ZNN_CUDA_KERNEL_NULL(matmul_cuda)
ZNN_CUDA_KERNEL_NULL(softmax_cuda)
ZNN_CUDA_KERNEL_NULL(sum_cuda)
ZNN_CUDA_KERNEL_NULL(mean_cuda)
ZNN_CUDA_KERNEL_NULL(cross_entropy_cuda)
ZNN_CUDA_KERNEL_NULL(mse_cuda)
ZNN_CUDA_KERNEL_NULL(conv2d_cuda)
ZNN_CUDA_KERNEL_NULL(max_pool2d_cuda)
ZNN_CUDA_KERNEL_NULL(avg_pool2d_cuda)

ZNN_CUDA_KERNEL_NULL(embedding_cuda)
ZNN_CUDA_KERNEL_NULL(reshape_cuda)
ZNN_CUDA_KERNEL_NULL(transpose_cuda)
ZNN_CUDA_KERNEL_NULL(permute_cuda)
ZNN_CUDA_KERNEL_NULL(slice_cuda)
ZNN_CUDA_KERNEL_NULL(cat_cuda)
ZNN_CUDA_KERNEL_NULL(stack_cuda)
ZNN_CUDA_KERNEL_NULL(arange_cuda)
ZNN_CUDA_KERNEL_NULL(randn_cuda)
ZNN_CUDA_KERNEL_NULL(layer_norm_cuda)
ZNN_CUDA_KERNEL_NULL(batch_norm2d_cuda)
ZNN_CUDA_KERNEL_NULL(attention_cuda)

void sgd_step_cuda(float*, const float*, float*, long long, float, float, float) { unavailable(); }
void adam_step_cuda(float*, const float*, float*, float*, long long,
                    float, float, float, float, float, float, float) { unavailable(); }
void memset0_cuda(void*, size_t) { unavailable(); }

} // namespace cuda_backend
} // namespace znn
