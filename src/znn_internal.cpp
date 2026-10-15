// ============================================================================
// znn_internal.cpp — 内部类实现 (骨架阶段: 逻辑体为 ZNN_TODO 占位)
// ============================================================================
#include "znn_internal.hpp"

namespace znn {

// ============================================================ Shape 方法
// 纯结构性实现(填数组/设秩/索引规范化), 随骨架落地

Shape::Shape(std::initializer_list<int64_t> dims) {
    if (dims.size() > 8) {
        throw std::runtime_error("znn: shape rank > 8");
    }
    int64_t i = 0;
    for (int64_t d : dims) {
        this->dims[i++] = d;
    }
    rank = i;
}

namespace {

int64_t normalize_index(int64_t index, int64_t rank) {
    int64_t i = index < 0 ? index + rank : index;
    if (i < 0 || i >= rank) {
        throw std::runtime_error("znn: shape index out of range");
    }
    return i;
}

} // namespace

int64_t& Shape::operator[](int64_t index) {
    return dims[normalize_index(index, rank)];
}

int64_t Shape::operator[](int64_t index) const {
    return dims[normalize_index(index, rank)];
}

// ============================================================ Storage

Storage* Storage::create(size_t bytes, Device device) {
    // TODO: 按设备分配对齐内存(CPU: aligned new; GPU 后端各自带分配器)
    ZNN_TODO();
}

Storage::Storage(size_t bytes, Device device)
    : refcount_(1), data_(nullptr), bytes_(bytes), device_(device) {
}

Storage::~Storage() {
    // TODO: 按设备释放 data_
}

Storage* Storage::retain() {
    ZNN_TODO();
}

void Storage::release() {
    ZNN_TODO();
}

// ============================================================ Tensor::Impl

Tensor::Impl* Tensor::Impl::create(const Shape& shape, DType dtype, Device device) {
    // TODO: 分配 storage, 按 row-major 计算 strides
    ZNN_TODO();
}

Tensor::Impl* Tensor::Impl::retain() {
    ZNN_TODO();
}

void Tensor::Impl::release() {
    ZNN_TODO();
}

bool Tensor::Impl::is_contiguous() const {
    ZNN_TODO();
}

size_t Tensor::Impl::nbytes() const {
    ZNN_TODO();
}

// ---- Tensor 公共方法: 句柄生命周期 ----

Tensor::Tensor()
    : impl_(nullptr) {
}

Tensor::Tensor(const Tensor& other)
    : impl_(other.impl_ ? other.impl_->retain() : nullptr) {
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (impl_) impl_->release();
        impl_ = other.impl_ ? other.impl_->retain() : nullptr;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : impl_(other.impl_) {
    other.impl_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (impl_) impl_->release();
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

Tensor::~Tensor() {
    if (impl_) impl_->release();
}

// ---- Tensor 公共方法: 创建 ----

Tensor Tensor::zeros(Shape shape, DType dtype, Device device) {
    ZNN_TODO();
}

Tensor Tensor::ones(Shape shape, DType dtype, Device device) {
    ZNN_TODO();
}

Tensor Tensor::empty(Shape shape, DType dtype, Device device) {
    ZNN_TODO();
}

Tensor Tensor::full(Shape shape, double value, DType dtype, Device device) {
    ZNN_TODO();
}

Tensor Tensor::from_host(const void* data, Shape shape, DType dtype, Device device) {
    ZNN_TODO();
}

// ---- Tensor 公共方法: 元信息 ----

const Shape& Tensor::shape() const {
    ZNN_TODO();
}

int64_t Tensor::size(int64_t dim) const {
    ZNN_TODO();
}

int64_t Tensor::numel() const {
    ZNN_TODO();
}

int64_t Tensor::dim_count() const {
    ZNN_TODO();
}

DType Tensor::dtype() const {
    ZNN_TODO();
}

Device Tensor::device() const {
    ZNN_TODO();
}

// ---- Tensor 公共方法: 视图 ----

Tensor Tensor::view(Shape shape) const {
    ZNN_TODO();
}

Tensor Tensor::reshape(Shape shape) const {
    ZNN_TODO();
}

Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    ZNN_TODO();
}

Tensor Tensor::squeeze(int64_t dim) const {
    ZNN_TODO();
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    ZNN_TODO();
}

Tensor Tensor::slice(int64_t dim, int64_t begin, int64_t end) const {
    ZNN_TODO();
}

Tensor Tensor::transpose(int64_t d0, int64_t d1) const {
    ZNN_TODO();
}

Tensor Tensor::permute(std::initializer_list<int64_t> dims) const {
    ZNN_TODO();
}

// ---- Tensor 公共方法: 设备/拷贝 ----

Tensor Tensor::to_device(Device device) const {
    ZNN_TODO();
}

Tensor Tensor::clone() const {
    ZNN_TODO();
}

// ---- Tensor 公共方法: 数据读取(模板显式实例化在文件尾) ----

template <class T>
T* Tensor::data() const {
    ZNN_TODO();
}

template <class T>
T Tensor::item() const {
    ZNN_TODO();
}

template float*   Tensor::data<float>() const;
template int64_t* Tensor::data<int64_t>() const;
template float    Tensor::item<float>() const;
template int64_t  Tensor::item<int64_t>() const;

// ---- Tensor 公共方法: 自动求导 ----

void Tensor::requires_grad(bool yes) {
    ZNN_TODO();
}

bool Tensor::requires_grad() const {
    ZNN_TODO();
}

Tensor& Tensor::grad() {
    ZNN_TODO();
}

void Tensor::backward() const {
    ZNN_TODO();
}

Tensor Tensor::detach() const {
    ZNN_TODO();
}

// ============================================================ AutogradEngine

AutogradEngine& AutogradEngine::instance() {
    static AutogradEngine engine;
    return engine;
}

GraphNode* AutogradEngine::record(const OpDef* op,
                                  Span<const Tensor*> inputs,
                                  Span<const Tensor*> outputs) {
    // TODO: 节点池分配 → 保存输入输出 → 链接上下游消费计数
    ZNN_TODO();
}

void AutogradEngine::backward(const Tensor& loss) {
    // TODO: 逆拓扑遍历, pending 归零触发, 梯度缓冲合并, 广播规约
    ZNN_TODO();
}

Tensor BackwardCtx::reduce_to(const Tensor& g, const Shape& target_shape) {
    // TODO: 按广播语义对 g 求和规约回 target_shape
    ZNN_TODO();
}

// ============================================================ OpRegistry

OpRegistry& OpRegistry::instance() {
    static OpRegistry registry;
    return registry;
}

OpRegistry::OpRegistry() {
    for (const OpDef* const* it = g_znn_op_list; *it != nullptr; ++it) {
        table_.emplace((*it)->name, *it);
    }
}

const OpDef* OpRegistry::find(const char* name) const {
    auto it = table_.find(name);
    return it == table_.end() ? nullptr : it->second;
}

Tensor OpRegistry::run(const OpDef* op, Span<const Tensor*> inputs, const void* attrs) {
    // TODO: 形状推理 → 输出分配 → 内核选择(空槽兜底链) → autograd 建图
    ZNN_TODO();
}

// ============================================================ Optimizer 工厂

Optimizer* Optimizer::create(OptimType type, const std::vector<Layer*>& layers,
                             const OptimParam& param) {
    // TODO: 收集各层 params() 后委托给下方 create
    ZNN_TODO();
}

Optimizer* Optimizer::create(OptimType type, std::vector<Tensor*> params,
                             const OptimParam& param) {
    // TODO: switch(type) 分发到 SgdImpl/AdamImpl/AdamWImpl
    ZNN_TODO();
}

void Optimizer::destroy(Optimizer* opt) {
    delete opt;
}

SgdImpl::SgdImpl(std::vector<Tensor*> params, const OptimParam& param)
    : params_(std::move(params)), param_(param) {
}

void SgdImpl::step() {
    // TODO: p -= lr * (grad + momentum 项 + weight_decay 项)
    ZNN_TODO();
}

void SgdImpl::clear() {
    // TODO: 各参数梯度清零
    ZNN_TODO();
}

AdamImpl::AdamImpl(std::vector<Tensor*> params, const OptimParam& param)
    : params_(std::move(params)), param_(param) {
}

void AdamImpl::step() {
    // TODO: 动量更新 + 偏差校正
    ZNN_TODO();
}

void AdamImpl::clear() {
    // TODO: 各参数梯度清零
    ZNN_TODO();
}

// ============================================================ Layer 工厂与实现

// 层对象逻辑上可变(参数被 Optimizer 更新), const 方法内取可变指针
static Tensor* mut(const Tensor& t) {
    return const_cast<Tensor*>(&t);
}

Layer* Layer::create(LayerType type, const LayerParam& param) {
    // TODO: switch(type) 分发 + 必填项校验(缺失抛异常, 不返回 nullptr)
    ZNN_TODO();
}

void Layer::destroy(Layer* layer) {
    delete layer;
}

LinearLayer::LinearLayer(int64_t, int64_t, bool bias)
    : has_bias_(bias) {
    // TODO: w_/b_ 按形状分配并初始化
}

Tensor LinearLayer::forward(const Tensor& x) {
    // TODO: Tensor::linear(x, w_, has_bias_ ? &b_ : nullptr)
    ZNN_TODO();
}

std::vector<Tensor*> LinearLayer::params() const {
    return has_bias_ ? std::vector<Tensor*>{mut(w_), mut(b_)}
                     : std::vector<Tensor*>{mut(w_)};
}

Conv2dLayer::Conv2dLayer(int64_t, int64_t, Shape kernel,
                         const Conv2dOptions& options, bool bias)
    : options_(options), has_bias_(bias) {
    // TODO: w_/b_ 按形状分配并初始化
}

Tensor Conv2dLayer::forward(const Tensor& x) {
    // TODO: Tensor::conv2d(x, w_, options_)
    ZNN_TODO();
}

std::vector<Tensor*> Conv2dLayer::params() const {
    return has_bias_ ? std::vector<Tensor*>{mut(w_), mut(b_)}
                     : std::vector<Tensor*>{mut(w_)};
}

BatchNorm2dLayer::BatchNorm2dLayer(int64_t, double eps, double momentum)
    : eps_(eps), momentum_(momentum) {
    // TODO: w_/b_/mean_/var_ 按通道数分配
}

Tensor BatchNorm2dLayer::forward(const Tensor& x) {
    // TODO: 训练态更新 running 统计, 推理态使用统计
    ZNN_TODO();
}

std::vector<Tensor*> BatchNorm2dLayer::params() const {
    return {mut(w_), mut(b_)};
}

LayerNormLayer::LayerNormLayer(int64_t, double eps)
    : eps_(eps) {
    // TODO: w_/b_ 按维度分配
}

Tensor LayerNormLayer::forward(const Tensor& x) {
    // TODO: 均值方差归一化 + 仿射
    ZNN_TODO();
}

std::vector<Tensor*> LayerNormLayer::params() const {
    return {mut(w_), mut(b_)};
}

EmbeddingLayer::EmbeddingLayer(int64_t, int64_t) {
    // TODO: w_ 按 [num, dim] 分配
}

Tensor EmbeddingLayer::forward(const Tensor& indices) {
    // TODO: Tensor::embedding(w_, indices)
    ZNN_TODO();
}

std::vector<Tensor*> EmbeddingLayer::params() const {
    return {mut(w_)};
}

ActivationLayer::ActivationLayer(LayerType type)
    : type_(type) {
}

Tensor ActivationLayer::forward(const Tensor& x) {
    // TODO: 按类型分发 Activator:: 静态方法
    ZNN_TODO();
}

FlattenLayer::FlattenLayer(int64_t start_dim)
    : start_dim_(start_dim) {
}

Tensor FlattenLayer::forward(const Tensor& x) {
    // TODO: x.flatten(start_dim_)
    ZNN_TODO();
}

Pool2dLayer::Pool2dLayer(LayerType type, Shape kernel, const Pool2dOptions& options)
    : type_(type), kernel_(kernel), options_(options) {
}

Tensor Pool2dLayer::forward(const Tensor& x) {
    // TODO: 按类型分发 Tensor::max_pool2d / avg_pool2d
    ZNN_TODO();
}

DropoutLayer::DropoutLayer(double rate, unsigned seed)
    : rate_(rate), seed_(seed) {
}

Tensor DropoutLayer::forward(const Tensor& x) {
    // TODO: 训练态按 rate 掩码缩放, 推理态直通
    ZNN_TODO();
}

// ============================================================ Runtime

namespace {

thread_local bool grad_enabled_flag = true;

} // namespace

bool Runtime::device_available(Device device) {
    return device == Device::CPU;
}

void Runtime::set_grad_enabled(bool yes) {
    grad_enabled_flag = yes;
}

bool Runtime::grad_enabled() {
    return grad_enabled_flag;
}

} // namespace znn
