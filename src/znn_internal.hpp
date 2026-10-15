// ============================================================================
// znn_internal.hpp — 通用内部定义源 (仅库内使用, 不随库发布)
//
// 只放跨子系统共用的基础设施: 设备存储 / Tensor 内部实体 / 计算图节点 /
// 求导引擎 / 算子注册表。子系统(层/优化器/算子)的类声明在各自目录的
// 头文件中, 不塞入本文件。
// ============================================================================
#pragma once
#include "znn.hpp"
#include "znn_ext.hpp"
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

namespace znn {

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

struct GraphNode;                 // 前置声明, 下方定义

// 公共 Tensor 的唯一实体: 存储视图 + 求导元数据
struct Tensor::Impl {
    Storage*   storage = nullptr;     // null = 空句柄
    int64_t    offset  = 0;           // 元素级偏移(视图)
    Shape      shape;
    int64_t    strides[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    DType      dtype  = DType::F32;
    Device     device = Device::CPU;

    bool       requires_grad = false;
    Impl*      grad    = nullptr;     // 惰性分配, 与本 Impl 同形状
    GraphNode* grad_fn = nullptr;     // 不拥有; 由 AutogradEngine 节点池管理

    static Impl* create(const Shape& shape, DType dtype, Device device);
    Impl*        retain();
    void         release();

    bool   is_contiguous() const;
    size_t nbytes() const;            // numel × 元素大小
};

// 层对象逻辑上可变(参数被 Optimizer 更新), const 方法内取可变参数指针
Tensor* mut(const Tensor& t);

// ============================================================ 计算图

// 前向建图节点: 创建序即拓扑序(动态图天然性质, 反向免排序)
struct GraphNode {
    const OpDef* op = nullptr;              // 反向规则入口
    std::vector<Tensor> saved_inputs;       // 前向输入(默认全存)
    std::vector<Tensor> saved_outputs;      // 前向输出(默认全存)
    std::vector<GraphNode*> consumers;      // 消费本节点输出的下游节点
    int pending = 0;                        // 尚未到齐的上游梯度数
};

// 自动求导引擎: 前向记录(建 tape)与反向执行(消费计数触发)集中于此。
// 实现范围: 节点池分配/图所有权/梯度缓冲合并(等价 PyTorch InputBuffer)/
// 广播梯度规约(BackwardCtx::reduce_to)/NoGrad 感知。
class AutogradEngine {
public:
    static AutogradEngine& instance();

    // 前向: 有 grad 规则且任一输入 requires_grad 时由 OpRegistry 调用
    GraphNode* record(const OpDef* op,
                      Span<const Tensor*> inputs,
                      Span<const Tensor*> outputs);
    // 反向: 从标量损失出发, 逆拓扑序填充各参数 .grad()
    void backward(const Tensor& loss);

private:
    AutogradEngine() = default;
};

// ============================================================ 注册表与派发

// 注册"机制"(查找/派发/兜底); 注册"数据"在 operators/operator_list.cpp。
// 启动惰性构建只读哈希表, 运行期无锁。
class OpRegistry {
public:
    static OpRegistry& instance();

    const OpDef* find(const char* name) const;      // 调试/测试用
    // 完整派发: 形状推理 → 输出分配 → 内核选择(空槽走兜底链) → 建图
    Tensor run(const OpDef* op, Span<const Tensor*> inputs, const void* attrs);

private:
    OpRegistry();
    std::unordered_map<std::string, const OpDef*> table_;
};

} // namespace znn
