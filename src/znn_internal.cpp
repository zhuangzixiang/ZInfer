// ============================================================================
// znn_internal.cpp — znn.hpp 的实现支撑
// (骨架阶段: 逻辑体为 ZNN_TODO 占位; 引擎在 autograd/op_registry, 层/优化器在各自目录)
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

thread_local bool grad_enabled_flag = true;

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

Tensor* mut(const Tensor& t) {
    return const_cast<Tensor*>(&t);
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

// ============================================================ Runtime

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
