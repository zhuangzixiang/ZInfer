// ============================================================================
// vulkan_stub.cpp — 非 Vulkan 构建的同名实现(明确报错, 派发器回落 CPU)
// ============================================================================
#include "operator.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include <stdexcept>

namespace znn {
namespace vulkan_backend {

namespace {
void unavailable() {
    throw std::runtime_error("znn: Vulkan backend not built (ZNN_VULKAN=OFF)");
}
}

bool available() { return false; }
void* alloc(size_t) { unavailable(); return nullptr; }
void free(void*) {}
void memset0(void*, size_t) { unavailable(); }
void copy_host_to_device(void*, const void*, size_t) { unavailable(); }
void copy_device_to_host(void*, const void*, size_t) { unavailable(); }
void copy_device_to_device(void*, const void*, size_t) { unavailable(); }

#define VK_STUB(name)                                                        \
    void name(KernelContext&, Span<const Tensor*>, Span<Tensor*>,            \
              const void*) {                                                 \
        unavailable();                                                       \
    }
VK_STUB(add_vulkan) VK_STUB(sub_vulkan) VK_STUB(mul_vulkan) VK_STUB(div_vulkan)
VK_STUB(neg_vulkan) VK_STUB(exp_vulkan) VK_STUB(log_vulkan) VK_STUB(sqrt_vulkan)
VK_STUB(relu_vulkan) VK_STUB(gelu_vulkan) VK_STUB(sigmoid_vulkan)
VK_STUB(tanh_vulkan) VK_STUB(silu_vulkan)
VK_STUB(matmul_vulkan) VK_STUB(linear_vulkan)
VK_STUB(softmax_vulkan) VK_STUB(sum_vulkan) VK_STUB(mean_vulkan)
#undef VK_STUB

} // namespace vulkan_backend
} // namespace znn
