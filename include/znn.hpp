// ============================================================================
// znn.hpp — ZInfer 接口定义 (v0.4 讨论稿)
// 与 znn_defines.hpp(结构定义)配对, 两者构成对外全部 API。
//
// ── 接口约定(全文适用, 只在此说明一次) ──────────────────────────────────
// [调用形态] 一切操作挂在类名下: Tensor/Activator/Loss/Runtime 的静态
//        方法, Layer/Optimizer 的虚接口 + create/destroy 工厂。
//        无全局自由函数。实例方法仅保留"本张量的形状/属性变换";
//        产生新张量的运算一律为静态方法。
// [命名法] 类型=名词; *Param=create() 构造参数体; *Options=运算选项;
//        统一 snake_case。
// [生命周期] Layer/Optimizer: create() 返回裸指针, 谁创建谁 destroy()。
//        自定义层直接继承 Layer 实现 forward 即可用, 无需注册。
//        Tensor 为引用计数句柄(含其 grad), 随最后一个引用自动释放。
// [设备] 创建默认 CPU; 跨设备搬运唯一入口 to_device()。数据读取只有
//        两个原语(模板在库内显式实例化, v1: float/int64_t):
//          data<T>()  仅 CPU 张量合法, 零拷贝内部缓冲;
//          item<T>()  任意设备合法, 同步读取单个数值。
//        GPU 张量批量读取 = 先 to_device(Device::CPU) 再 data<T>()。
//        批量搬运必须显式(隐式同步是性能陷阱), 隐式同步仅 item 单值。
// [求导] requires_grad(true) 的张量参与建图; 标量损失 backward() 后
//        各参数 grad() 就绪, 重复 backward() 梯度累加。训练中的验证/
//        推理段用 Runtime::set_grad_enabled(false) 全局暂停建图。
// [错误] 非法用法抛 std::runtime_error 派生异常(编译选项可关闭)。
// ============================================================================
#pragma once
#include "znn_defines.hpp"
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace znn {

// ---------------------------------------------------------------- Runtime
// 运行时环境开关与查询

class Runtime {
public:
    static bool device_available(Device dev);   // 后端是否编入且可用(编译开关决定)
    static void set_grad_enabled(bool yes);     // false = 全局暂停建图(验证/推理段)
    static bool grad_enabled();
};

// ---------------------------------------------------------------- Tensor

class Tensor {
public:
    // ---- 创建(目标设备默认 CPU) ----
    static Tensor zeros(Shape shape, DType dt = DType::F32, Device dev = Device::CPU);
    static Tensor ones (Shape shape, DType dt = DType::F32, Device dev = Device::CPU);
    static Tensor empty(Shape shape, DType dt = DType::F32, Device dev = Device::CPU);
    static Tensor full (Shape shape, double value, DType dt = DType::F32, Device dev = Device::CPU);
    // 拷贝 host 内存构造张量(拷贝完成即与源内存无关)
    static Tensor from_host(const void* data, Shape shape,
                            DType dt = DType::F32, Device dev = Device::CPU);
    static Tensor arange(int64_t n, DType dt = DType::F32, Device dev = Device::CPU);
    static Tensor randn(Shape shape, unsigned seed = 0, DType dt = DType::F32, Device dev = Device::CPU);

    // ---- 逐元素(输入间自动广播) ----
    static Tensor add(const Tensor& a, const Tensor& b);
    static Tensor sub(const Tensor& a, const Tensor& b);
    static Tensor mul(const Tensor& a, const Tensor& b);
    static Tensor div(const Tensor& a, const Tensor& b);
    static Tensor neg(const Tensor& x);
    static Tensor exp(const Tensor& x);
    static Tensor log(const Tensor& x);
    static Tensor sqrt(const Tensor& x);

    // ---- 规约 ----
    static Tensor sum (const Tensor& x, int64_t dim, bool keepdim = false);
    static Tensor mean(const Tensor& x, int64_t dim, bool keepdim = false);

    // ---- 矩阵 ----
    static Tensor matmul(const Tensor& a, const Tensor& b);   // 2D 乘法, 高维按广播批处理
    static Tensor linear(const Tensor& x, const Tensor& w, const Tensor* b = nullptr);  // x @ w.T + b

    // ---- 拼接 ----
    static Tensor cat (const std::vector<Tensor>& ts, int64_t dim = 0);
    static Tensor cat (const Tensor& a, const Tensor& b, int64_t dim = 0);
    static Tensor stack(const std::vector<Tensor>& ts, int64_t dim = 0);

    // ---- 卷积/池化(NCHW; w 形状 [out, in/groups, kh, kw]; k = {kh, kw}) ----
    static Tensor conv2d    (const Tensor& x, const Tensor& w, const Conv2dOptions& o = {});
    static Tensor max_pool2d(const Tensor& x, Shape k, const Pool2dOptions& o = {});
    static Tensor avg_pool2d(const Tensor& x, Shape k, const Pool2dOptions& o = {});

    // ---- 检索 ----
    static Tensor embedding(const Tensor& weight, const Tensor& indices);   // indices: I64

    // ---- 注意力(q/k/v: [B,H,L,D] 同形; causal=true 屏蔽未来位) ----
    static Tensor attention(const Tensor& q, const Tensor& k,
                            const Tensor& v, bool causal = false);

public:
    // ---- 元信息 ----
    const Shape& shape()     const;
    int64_t      size(int64_t dim) const;    // 支持负索引, -1 = 最后一维
    int64_t      numel()     const;          // 元素总数
    int64_t      dim_count() const;          // = shape().rank
    DType        dtype()     const;
    Device       device()    const;

    // ---- 句柄生命周期(实现于库内; 拷贝=共享存储并递增引用计数) ----
    Tensor();
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    // ---- 视图(零拷贝, 与原张量共享存储) ----
    Tensor view    (Shape shape)      const;  // 要求存储连续
    Tensor reshape (Shape shape)      const;
    Tensor flatten (int64_t start_dim = 0, int64_t end_dim = -1) const;
    Tensor squeeze (int64_t dim = -1) const;  // -1 = 去掉所有长度 1 的维
    Tensor unsqueeze(int64_t dim)     const;
    Tensor slice   (int64_t dim, int64_t begin, int64_t end) const;
    Tensor transpose(int64_t d0, int64_t d1) const;
    Tensor permute (std::initializer_list<int64_t> dims) const;

    // ---- 设备/拷贝 ----
    Tensor to_device(Device dev) const;      // 跨设备同步拷贝; 同设备则共享返回
    Tensor clone() const;                    // 同设备深拷贝(存储独立, 仍参与建图)

    // ---- 数据读取(仅两个原语; GPU 张量先 to_device(Device::CPU)) ----
    template <class T> T* data() const;   // 仅 CPU 张量合法, 零拷贝内部缓冲
    template <class T> T  item() const;   // 任意设备合法, 同步读取单值

    // ---- 自动求导 ----
    void   requires_grad(bool yes = true);   // 参数张量在训练前设置
    bool   requires_grad() const;
    Tensor& grad();                          // 首次访问惰性分配; 供读取与清零
    void   backward() const;                 // 本张量须为标量; 反向填充各 .grad()
    Tensor detach() const;                   // 共享存储但脱离计算图(冻结中间量)

private:
    struct Impl;
    Impl* impl_;               // 引用计数句柄, 生命周期由库内部管理
    friend struct Impl;        // 内部实现经由 Impl 访问句柄
};

// ---------------------------------------------------------------- Activator
// 命名静态方法: 调用处与论文同构, 编译期分发; 新激活 = 加一个方法。

class Activator {
public:
    static Tensor relu   (const Tensor& x);
    static Tensor gelu   (const Tensor& x);
    static Tensor sigmoid(const Tensor& x);
    static Tensor tanh   (const Tensor& x);
    static Tensor silu   (const Tensor& x);
    static Tensor softmax(const Tensor& x, int64_t dim = -1);
};

// ---------------------------------------------------------------- Loss
// 损失函数返回标量张量。

class Loss {
public:
    // 交叉熵 = softmax + 负对数似然(分类标准损失); targets: I64 标签
    static Tensor cross_entropy(const Tensor& logits, const Tensor& targets);
    // 均方误差(回归标准损失)
    static Tensor mse          (const Tensor& pred,   const Tensor& target);
};

// ---------------------------------------------------------------- Layer
// 虚接口 + 工厂。自定义层: 继承 Layer 实现 forward 即可直接使用, 无需注册。

class Layer {
public:
    // 按类型构造, 必填项缺失/非法抛异常(不返回 nullptr)
    static Layer* create(LayerType type, const LayerParam& param);
    static void   destroy(Layer* layer);       // 允许传 nullptr

    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Tensor*> params() const = 0;   // 本层可训练权重
    virtual void   train(bool mode = true) = 0;        // BN 统计更新/Dropout 生效与否

protected:
    Layer() = default;                                 // 仅经 create / 继承构造
};

// ---------------------------------------------------------------- Optimizer

class Optimizer {
public:
    // 从一组 Layer 自动收集参数并构造(常规训练)
    static Optimizer* create(OptimType type, const std::vector<Layer*>& layers, const OptimParam& param);
    // 指定参数张量(自定义层 / 仅微调部分参数)
    static Optimizer* create(OptimType type, std::vector<Tensor*> params, const OptimParam& param);
    static void destroy(Optimizer* opt);       // 允许传 nullptr

    virtual ~Optimizer() = default;

    virtual void step() = 0;      // 按 .grad() 更新参数
    virtual void clear() = 0;     // 全部参数梯度清零

protected:
    Optimizer() = default;
};

} // namespace znn
