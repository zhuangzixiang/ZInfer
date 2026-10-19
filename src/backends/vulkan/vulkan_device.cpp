// ============================================================================
// vulkan_device.cpp — Vulkan 设备与内存基础设施
//
// 链路: vulkan-1.dll 动态加载(驱动自带) → 独立计算队列 → 主机可见内存
// (v1 简化: HOST_VISIBLE|HOST_COHERENT, 免 staging; 独占内存优化属后续)。
// 管线: SPIR-V 构建期嵌入, 运行时按(着色器, 特化常量)缓存。
// ============================================================================
#include "backends/vulkan/vulkan_backend.hpp"

#if defined(ZNN_VULKAN)

#include "vulkan/vulkan.h"
#include <cstring>
#include <cstdio>
#include <vector>
#include <mutex>
#include <unordered_map>
#ifdef _WIN32
  #include <windows.h>
#endif

namespace znn {
namespace vulkan_backend {

// 管线与缓存类型(device 命名空间 —— 桥与内核共享, 不匿名)
namespace device {
struct Pipeline {
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkDescriptorSetLayout dsl;
    VkDescriptorSet ds;
};
struct PipeKey {
    const uint32_t* code;
    size_t bytes;
    uint32_t c0, c1;
    uint32_t nbind;      // 存储缓冲绑定数(2..8; 0 视为 3, 兼容旧调用)
    bool operator==(const PipeKey& o) const {
        return code == o.code && bytes == o.bytes && c0 == o.c0 && c1 == o.c1
               && nbind == o.nbind;
    }
};
struct PipeKeyHash {
    size_t operator()(const PipeKey& k) const {
        return std::hash<const void*>()(k.code) ^ (k.c0 * 31) ^ (k.c1 << 8)
               ^ (k.nbind << 16);
    }
};
} // namespace device

namespace {

// ---- 动态加载(与 dynlib 同哲学: 缺驱动则全回落) ----
#ifdef _WIN32
  #define VK_DL(name)   reinterpret_cast<void*>(GetProcAddress(hmod_, name))
  HMODULE hmod_ = nullptr;
  void* vk_open_lib() {
    if (!hmod_) hmod_ = LoadLibraryA("vulkan-1.dll");
    return hmod_ ? (void*)1 : nullptr;
  }
#else
  #include <dlfcn.h>
  void* vk_lib_ = nullptr;
  void* vk_open_lib() {
    if (!vk_lib_) vk_lib_ = dlopen("libvulkan.so.1", RTLD_LAZY);
    if (!vk_lib_) vk_lib_ = dlopen("libvulkan.so", RTLD_LAZY);
    return vk_lib_;
  }
  #define VK_DL(name)   dlsym(vk_lib_, name)
#endif

// 实例级函数(经 vkGetInstanceProcAddr 按实例解析)
#define VK_FOREACH_INSTANCE_FN(F)                                          \
    F(vkCreateInstance) F(vkEnumeratePhysicalDevices) F(vkGetPhysicalDeviceProperties) \
    F(vkGetPhysicalDeviceMemoryProperties) F(vkGetPhysicalDeviceQueueFamilyProperties) \
    F(vkCreateDevice) F(vkDestroyDevice) F(vkDestroyInstance) F(vkGetDeviceProcAddr) \
    F(vkEnumerateDeviceExtensionProperties)
// 设备级函数(经 vkGetDeviceProcAddr 解析)
#define VK_FOREACH_DEVICE_FN(F)                                            \
    F(vkAllocateMemory) F(vkFreeMemory) F(vkMapMemory) F(vkUnmapMemory)    \
    F(vkCreateBuffer) F(vkDestroyBuffer) F(vkBindBufferMemory)             \
    F(vkGetBufferMemoryRequirements)                                       \
    F(vkCreateShaderModule) F(vkDestroyShaderModule)                       \
    F(vkCreatePipelineLayout) F(vkDestroyPipelineLayout)                   \
    F(vkCreateComputePipelines) F(vkDestroyPipeline)                       \
    F(vkCreateCommandPool) F(vkDestroyCommandPool)                         \
    F(vkAllocateCommandBuffers) F(vkFreeCommandBuffers)                    \
    F(vkBeginCommandBuffer) F(vkEndCommandBuffer) F(vkResetCommandBuffer)  \
    F(vkCmdBindPipeline) F(vkCmdBindDescriptorSets) F(vkCmdPushConstants)  \
    F(vkCmdDispatch) F(vkCmdCopyBuffer) F(vkCmdFillBuffer)                 \
    F(vkQueueSubmit) F(vkQueueWaitIdle) F(vkGetDeviceQueue)                                    \
    F(vkCreateDescriptorSetLayout) F(vkDestroyDescriptorSetLayout)         \
    F(vkCreateDescriptorPool) F(vkDestroyDescriptorPool)                   \
    F(vkAllocateDescriptorSets) F(vkUpdateDescriptorSets)

#define VK_DECLARE_FN(name) PFN_##name name;
VK_FOREACH_INSTANCE_FN(VK_DECLARE_FN)
VK_FOREACH_DEVICE_FN(VK_DECLARE_FN)
#undef VK_DECLARE_FN

// ---- 全局 Vulkan 状态(进程级单次初始化; static 必要性: Vulkan 句柄
// 天然进程级, 与 cuBLAS handle 同生命周期语义) ----
struct VkState {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pdev = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfamily = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    uint32_t host_mem_idx = UINT32_MAX;    // 系统内存(小分配: CPU 写缓存 + L2 吸收)
    uint32_t local_mem_idx = UINT32_MAX;  // BAR 显存(大分配: GPU 直读 VRAM)
    bool ok = false;
    std::mutex mu;                       // 提交串行化(v1 单队列)
    // 缓冲池: 尺寸桶(2^n)空闲链表 —— 每调用 vkCreateBuffer+AllocateMemory
    // 是毫秒级开销, 复用免分配(与 CUDA 驱动池同动机; 归还而非销毁)
    std::unordered_map<uint64_t, std::vector<void*>> buf_pool;   // key=(类型<<48|桶)
};
VkState* st() {
    static VkState s;
    return &s;
}

bool init_vulkan() {
    VkState* v = st();
    if (v->ok || !vk_open_lib()) return v->ok;
    auto gipa = (PFN_vkGetInstanceProcAddr)VK_DL("vkGetInstanceProcAddr");
    if (!gipa) { std::fprintf(stderr, "[vk] no gipa\n"); return false; }
    // 全局级仅 vkCreateInstance 可直取; 其余在实例创建后经 gipa(inst) 解析
    vkCreateInstance = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!vkCreateInstance) {
        std::fprintf(stderr, "[vk] no createInstance entry\n");
        return false;
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_1;   // 计算所需最低版(推送常量+storage)
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    const VkResult rci = vkCreateInstance(&ci, nullptr, &v->inst);
    if (rci != VK_SUCCESS) return false;
#define VK_LOAD_I2(name)                                                   \
    name = (PFN_##name)gipa(v->inst, #name);                               \
    if (!name) { std::fprintf(stderr, "[vk] inst-fn %s null\n", #name); return false; }
    VK_FOREACH_INSTANCE_FN(VK_LOAD_I2)
#undef VK_LOAD_I2

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(v->inst, &ndev, nullptr);
    if (ndev == 0) return false;
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(v->inst, &ndev, devs.data());
    v->pdev = devs[0];

    // 计算队列族
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(v->pdev, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(v->pdev, &nq, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < nq; ++i) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            v->qfamily = i;
            found = true;
            break;
        }
    }
    if (!found) { std::fprintf(stderr, "[vk] no compute queue\n"); return false; }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = v->qfamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    const VkResult rdv = vkCreateDevice(v->pdev, &dci, nullptr, &v->dev);
    if (rdv != VK_SUCCESS) return false;
#define VK_LOAD_DEV(name)                                                  \
    name = (PFN_##name)vkGetDeviceProcAddr(v->dev, #name);                 \
    if (!name) { std::fprintf(stderr, "[vk] dev-fn %s null\n", #name); return false; }
    VK_FOREACH_DEVICE_FN(VK_LOAD_DEV)
#undef VK_LOAD_DEV
    vkGetDeviceQueue(v->dev, v->qfamily, 0, &v->queue);
    // 内存类型选择: 优先 DEVICE_LOCAL+HOST_VISIBLE(BAR 显存, 680GB/s);
    // 兜底首个 HOST_VISIBLE。此前取首个可见类型(常为系统内存堆) ——
    // GPU 经 PCIe 读取(~25GB/s), 大矩阵 L2 失效后崩塌(存量缺口, 本块修复)
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(v->pdev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
        if ((f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) continue;
        if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            v->local_mem_idx = i;         // BAR 显存(大矩阵: GPU 直读 VRAM)
        } else if (v->host_mem_idx == UINT32_MAX) {
            v->host_mem_idx = i;          // 系统内存(小矩阵实测更快)
        }
    }
    if (v->host_mem_idx == UINT32_MAX) v->host_mem_idx = v->local_mem_idx;
    if (v->host_mem_idx == UINT32_MAX) return false;

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = v->qfamily;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(v->dev, &pci, nullptr, &v->pool) != VK_SUCCESS) return false;

    VkDescriptorPoolSize dpsz[1] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64}};
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 64;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = dpsz;
    if (vkCreateDescriptorPool(v->dev, &dpi, nullptr, &v->dpool) != VK_SUCCESS) {
        return false;
    }
    v->ok = true;
    return true;
}

// ---- 内存分配: Buffer + DeviceMemory 绑定, 句柄内携带两者 ----
struct VkBuf {
    VkBuffer buf;
    VkDeviceMemory mem;
    void* mapped;          // 主机可见: 常驻映射
    size_t cap;
    uint32_t mem_idx;
};

} // namespace

bool available() {
    static const bool ok = init_vulkan();
    return ok;
}

namespace {
size_t bucket_of(size_t bytes) {          // 桶 = ≥256 的下一个 2^n
    size_t b = 256;
    while (b < bytes) b <<= 1;
    return b;
}
}

void* alloc(size_t bytes) {
    VkState* v = st();
    if (!v->ok || bytes == 0) return nullptr;
    // 内存类型: ≥64MB(4096 级矩阵)走 BAR 显存, 其余系统内存
    // (实测: ≤2048 工作集系统内存+L2 更快; 8MB 阈值曾把 2048 推上显存反而 2.6× 慢)
    const uint32_t mem_idx = (bytes >= ((size_t)64 << 20)
                              && v->local_mem_idx != UINT32_MAX)
                                 ? v->local_mem_idx : v->host_mem_idx;
    {   // 池命中(key = 类型<<48|桶; 超大块不池化)
        if (bytes <= (size_t)256 << 20) {
        const uint64_t key = ((uint64_t)mem_idx << 48) | bucket_of(bytes);
        std::lock_guard<std::mutex> lk(v->mu);
        auto it = v->buf_pool.find(key);
        if (it != v->buf_pool.end() && !it->second.empty()) {
            void* p = it->second.back();
            it->second.pop_back();
            return p;
        }
        bytes = key & 0xFFFFFFFFFFFFULL;  // 新建按桶容量(下次可复用)
        }
    }
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
              | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf;
    if (vkCreateBuffer(v->dev, &bci, nullptr, &buf) != VK_SUCCESS) return nullptr;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(v->dev, buf, &mr);
    // 内存类型须同时满足: 主机可见 且 缓冲区需求位掩码
    if (((mr.memoryTypeBits >> mem_idx) & 1) == 0) {   // 类型与缓冲不兼容
        std::fprintf(stderr, "[vk] alloc: no compatible host memory\n");
        vkDestroyBuffer(v->dev, buf, nullptr);
        return nullptr;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mem_idx;
    VkDeviceMemory mem;
    const VkResult ram = vkAllocateMemory(v->dev, &mai, nullptr, &mem);
    if (ram != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] alloc memory st=%d\n", (int)ram);
        vkDestroyBuffer(v->dev, buf, nullptr);
        return nullptr;
    }
    const VkResult rbm = vkBindBufferMemory(v->dev, buf, mem, 0);
    if (rbm != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] bind st=%d\n", (int)rbm);
        vkFreeMemory(v->dev, mem, nullptr);
        vkDestroyBuffer(v->dev, buf, nullptr);
        return nullptr;
    }
    void* mapped = nullptr;
    const VkResult rmp = vkMapMemory(v->dev, mem, 0, bytes, 0, &mapped);
    if (rmp != VK_SUCCESS) {
        std::fprintf(stderr, "[vk] map st=%d\n", (int)rmp);
        vkFreeMemory(v->dev, mem, nullptr);
        vkDestroyBuffer(v->dev, buf, nullptr);
        return nullptr;
    }
    return new VkBuf{buf, mem, mapped, bytes, mem_idx};
}

void free(void* p) {
    VkState* v = st();
    if (!p) return;
    if (v->ok) {
        VkBuf* b = static_cast<VkBuf*>(p);
        if (b->cap <= (size_t)256 << 20) {  // 池内块归还复用; 超大块真销毁
            std::lock_guard<std::mutex> lk(v->mu);
            v->buf_pool[((uint64_t)b->mem_idx << 48) | bucket_of(b->cap)].push_back(p);
            return;
        }
        vkUnmapMemory(v->dev, b->mem);
        vkFreeMemory(v->dev, b->mem, nullptr);
        vkDestroyBuffer(v->dev, b->buf, nullptr);
        delete b;
        return;
    }
    delete static_cast<VkBuf*>(p);
}

// 主机可见内存: 拷贝即 memcpy(常驻映射), 设备一致性由 COHERENT 保证
void copy_host_to_device(void* dst, const void* src, size_t bytes) {
    memcpy(static_cast<VkBuf*>(dst)->mapped, src, bytes);
}
void copy_device_to_host(void* dst, const void* src, size_t bytes) {
    memcpy(dst, static_cast<const VkBuf*>(src)->mapped, bytes);
}

// ---- 计算派发入口(内核文件使用) ----
namespace device {

VkDevice dev() { return st()->dev; }
std::mutex& mutex() { return st()->mu; }

VkCommandBuffer begin_cmd() {
    VkState* v = st();
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = v->pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(v->dev, &ai, &cb);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

void submit_wait(VkCommandBuffer cb) {
    VkState* v = st();
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    const VkResult rs = vkQueueSubmit(v->queue, 1, &si, VK_NULL_HANDLE);
    const VkResult rw = vkQueueWaitIdle(v->queue);
    if (rs != VK_SUCCESS || rw != VK_SUCCESS) {   // [诊断] 设备丢失/提交失败
        std::fprintf(stderr, "[vk] submit st=%d wait st=%d\n", (int)rs, (int)rw);
    }
    vkFreeCommandBuffers(v->dev, v->pool, 1, &cb);
}

void d2d(VkBuffer dst, VkBuffer src, size_t bytes) {
    VkCommandBuffer cb = begin_cmd();
    VkBufferCopy c{};
    c.size = bytes;
    vkCmdCopyBuffer(cb, src, dst, 1, &c);
    submit_wait(cb);
}

void fill0(VkBuffer buf, size_t bytes) {
    VkCommandBuffer cb = begin_cmd();
    vkCmdFillBuffer(cb, buf, 0, bytes, 0);
    submit_wait(cb);
}

void* raw_buf(void* p) { return static_cast<VkBuf*>(p); }   // VkBuf 句柄
void* mapped_of(void* p) { return static_cast<VkBuf*>(p)->mapped; }


bool get_pipeline(PipeKey key, const size_t* push_sizes,   // [storage, push]
                  Pipeline* out) {
    VkState* v = st();
    static std::unordered_map<PipeKey, Pipeline, PipeKeyHash> cache;
    auto it = cache.find(key);
    if (it != cache.end()) {
        *out = it->second;
        return true;
    }
    // 存储缓冲绑定(通用形态: 按内核需要 2..8 个)
    const uint32_t nb = key.nbind == 0 ? 3 : key.nbind;
    VkDescriptorSetLayoutBinding lb[8];
    for (uint32_t i = 0; i < nb; ++i) {
        lb[i].binding = i;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = nb;
    dli.pBindings = lb;
    VkDescriptorSetLayout dsl;
    if (vkCreateDescriptorSetLayout(v->dev, &dli, nullptr, &dsl) != VK_SUCCESS) {
        return false;
    }
    // 推送常量(规约着色器 ~96B > 默认 128 上限内; 足够)
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 128;
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VkPipelineLayout layout;
    if (vkCreatePipelineLayout(v->dev, &pli, nullptr, &layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(v->dev, dsl, nullptr);
        return false;
    }
    VkShaderModuleCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sci.codeSize = key.bytes;
    sci.pCode = key.code;
    VkShaderModule mod;
    const VkResult rsm = vkCreateShaderModule(v->dev, &sci, nullptr, &mod);
    std::fprintf(stderr, "[pg] shader st=%d\n", (int)rsm);
    if (rsm != VK_SUCCESS) {
        vkDestroyPipelineLayout(v->dev, layout, nullptr);
        vkDestroyDescriptorSetLayout(v->dev, dsl, nullptr);
        return false;
    }
    static VkSpecializationMapEntry me[2] = {{0, 0, 4}, {1, 4, 4}};
    static uint32_t spec_store[2];            // pData 缺失曾致驱动崩溃
    spec_store[0] = key.c0;
    spec_store[1] = key.c1;
    VkSpecializationInfo spi{};
    spi.mapEntryCount = 2;
    spi.pMapEntries = me;
    spi.dataSize = 8;
    spi.pData = spec_store;
    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = &spi;
    cpi.layout = layout;
    VkPipeline pipe;
    VkResult r = vkCreateComputePipelines(v->dev, VK_NULL_HANDLE, 1, &cpi,
                                          nullptr, &pipe);
    std::fprintf(stderr, "[pg] pipeline st=%d\n", (int)r);
    vkDestroyShaderModule(v->dev, mod, nullptr);
    if (r != VK_SUCCESS) {
        vkDestroyPipelineLayout(v->dev, layout, nullptr);
        vkDestroyDescriptorSetLayout(v->dev, dsl, nullptr);
        return false;
    }
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = v->dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    if (vkAllocateDescriptorSets(v->dev, &dai, &ds) != VK_SUCCESS) {
        vkDestroyPipeline(v->dev, pipe, nullptr);
        vkDestroyPipelineLayout(v->dev, layout, nullptr);
        vkDestroyDescriptorSetLayout(v->dev, dsl, nullptr);
        return false;
    }
    Pipeline p{layout, pipe, dsl, ds};
    cache[key] = p;
    *out = p;
    return true;
}

// 绑定 N 缓冲 + 推送常量 + 派发(同步等待)
bool dispatch_bufs(const Pipeline& p, const VkBuffer* bufs, uint32_t nbuf,
                   const void* push, size_t push_size,
                   uint32_t gx, uint32_t gy, uint32_t gz) {
    VkState* v = st();
    VkDescriptorBufferInfo bi[8] = {};
    VkWriteDescriptorSet w[8] = {};   // pNext 未清零曾致驱动崩溃
    for (uint32_t i = 0; i < nbuf; ++i) {
        bi[i] = VkDescriptorBufferInfo{bufs[i], 0, VK_WHOLE_SIZE};
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = p.ds;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(v->dev, nbuf, w, 0, nullptr);
    VkCommandBuffer cb = begin_cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1,
                            &p.ds, 0, nullptr);
    if (push) {
        vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           (uint32_t)push_size, push);
    }
    vkCmdDispatch(cb, gx, gy, gz);
    submit_wait(cb);
    return true;
}

// 绑定 3 缓冲(a/b/y)的常用形态
bool dispatch(const Pipeline& p, VkBuffer a, VkBuffer b, VkBuffer y,
              const void* push, size_t push_size,
              uint32_t gx, uint32_t gy, uint32_t gz) {
    const VkBuffer bufs[3] = {a, b, y};
    return dispatch_bufs(p, bufs, 3, push, push_size, gx, gy, gz);
}

VkBuffer buf_of(void* p) { return static_cast<VkBuf*>(p)->buf; }

} // namespace device

void copy_device_to_device(void* dst, const void* src, size_t bytes) {
    device::d2d(device::buf_of(dst), device::buf_of(const_cast<void*>(src)), bytes);
}

void memset0(void* p, size_t bytes) {
    device::fill0(device::buf_of(p), bytes);
}

// ---- 内核桥(隐藏 Vk* 类型; kernels 文件经不透明句柄使用) ----
namespace device_bridge {

void* get_pipeline(const void* code, size_t bytes, unsigned c0, unsigned c1,
                   unsigned nbind) {
    static std::unordered_map<uint64_t, void*> bridge_cache;   // 指针+常量→管线
    const uint64_t key = (uint64_t)(uintptr_t)code ^ ((uint64_t)c0 << 40)
                         ^ ((uint64_t)c1 << 48) ^ ((uint64_t)nbind << 56);
    auto it = bridge_cache.find(key);
    if (it != bridge_cache.end()) return it->second;
    device::Pipeline p;
    if (!device::get_pipeline({(const uint32_t*)code, bytes, c0, c1, nbind},
                              nullptr, &p)) {
        return nullptr;
    }
    // Pipeline 是缓存值拷贝(句柄型字段), 堆上驻留一份供桥使用
    void* held = new device::Pipeline(p);
    bridge_cache[key] = held;
    return held;
}

bool dispatch(void* pipe, void* a, void* b, void* y,
              const void* push, size_t push_size,
              unsigned gx, unsigned gy, unsigned gz) {
    device::Pipeline* p = static_cast<device::Pipeline*>(pipe);
    return device::dispatch(*p, device::buf_of(a), device::buf_of(b),
                            device::buf_of(y), push, push_size, gx, gy, gz);
}

bool dispatch_bufs(void* pipe, void* const* bufs, int nbuf,
                   const void* push, size_t push_size,
                   unsigned gx, unsigned gy, unsigned gz) {
    device::Pipeline* p = static_cast<device::Pipeline*>(pipe);
    VkBuffer bs[8];
    for (int i = 0; i < nbuf; ++i) {
        bs[i] = device::buf_of(bufs[i]);
    }
    return device::dispatch_bufs(*p, bs, (uint32_t)nbuf, push, push_size,
                                 gx, gy, gz);
}

} // namespace device_bridge

} // namespace vulkan_backend
} // namespace znn

#endif // ZNN_VULKAN
