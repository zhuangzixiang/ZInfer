// ============================================================================
// znn_internal.cpp — znn.hpp 的实现支撑
// (引擎在 autograd/op_dispatcher; 层/优化器在各自目录; 算子在 operators/)
// ============================================================================
#include "znn_internal.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "autograd.hpp"
#include "op_dispatcher.hpp"
#include "operators/shared.hpp"
#include <cstring>
#include <type_traits>

namespace znn {

namespace impl {

// 视图(src 任意 strided) → dst(连续)逐元素紧凑拷贝, 只经公共访问器
void copy_view_to(const Tensor& src, Tensor& dst) {
    if (dst.device() == Device::CUDA) {   // 原生: 双步长拷贝(dst 可为非连续视图)
        long long ss[8], ds[8], sh[8];
        const int rank = (int)dst.dim_count();
        for (int d = 0; d < rank; ++d) {
            ss[d] = src.strides()[d];
            ds[d] = dst.strides()[d];
            sh[d] = dst.size(d);
        }
        cuda_backend::strided_copy_cuda(src.data<float>(), dst.data<float>(),
                                        rank, ss, ds, sh, dst.numel());
        return;
    }
    const int64_t rank = dst.dim_count();
    const float* sp = src.data<float>();
    float* dp = dst.data<float>();
    const int64_t* ss = src.strides();
    const int64_t* ds = dst.strides();
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < dst.numel(); ++i) {
        int64_t so = 0, doff = 0;              // 双步长: dst 可为非连续视图(散射)
        for (int64_t d = 0; d < rank; ++d) {
            so += idx[d] * ss[d];
            doff += idx[d] * ds[d];
        }
        dp[doff] = sp[so];
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < dst.size(d)) break;
            idx[d] = 0;
        }
    }
}

} // namespace impl

// ============================================================ 内部小工具

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:
        case DType::I32: return 4;
        case DType::F16:
        case DType::BF16: return 2;
        case DType::I64: return 8;
        case DType::U8: return 1;
        default: throw std::runtime_error("znn: unknown dtype");
    }
}

void check_cpu(Device device) {
    if (device != Device::CPU) {
        throw std::runtime_error("znn: device not built in (GPU 里程碑)");
    }
}

// 见头文件声明: dst 连续, src 任意 strided 视图
void Tensor::Impl::copy_elements(const Tensor::Impl* src, Tensor::Impl* dst) {
    if (src->device == Device::CUDA || dst->device == Device::CUDA) {
        if (src->device != dst->device) {
            throw std::runtime_error("znn: cross-device copy must go through to_device");
        }
        if (src->is_contiguous()) {
            cuda_backend::copy_device_to_device(dst->storage->data(), src->storage->data(),
                                                dst->nbytes());
            return;
        }
        // 非连续视图(转置/切片): 按步长收集(此前扁平 memcpy 静默拷错 —— 存量 bug)
        if (src->dtype != DType::F32) {
            throw std::runtime_error("znn: cuda clone of non-contiguous non-F32");
        }
        long long ss[8], ds[8], sh[8], total = 1;
        const int rank = (int)dst->shape.rank;
        for (int d = 0; d < rank; ++d) {
            ss[d] = src->strides[d];
            ds[d] = dst->strides[d];
            sh[d] = dst->shape.dims[d];
            total *= dst->shape.dims[d];
        }
        cuda_backend::strided_copy_cuda(
            static_cast<const float*>(src->storage->data()) + src->offset,
            static_cast<float*>(dst->storage->data()), rank, ss, ds, sh, total);
        return;
    }
    const int64_t total = static_cast<int64_t>(dst->nbytes() / dtype_size(dst->dtype));
    const size_t esz = dtype_size(dst->dtype);
    char* out = static_cast<char*>(dst->storage->data());
    const char* base = static_cast<const char*>(src->storage->data());
    if (src->is_contiguous()) {
        std::memcpy(out, base + src->offset * esz, dst->nbytes());
        return;
    }
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int64_t rank = dst->shape.rank;
    for (int64_t i = 0; i < total; ++i) {
        int64_t off = src->offset;
        for (int64_t d = 0; d < rank; ++d) {
            off += idx[d] * src->strides[d];
        }
        std::memcpy(out + i * esz, base + off * esz, esz);
        for (int64_t d = rank - 1; d >= 0; --d) {          // 行主序推进多下标
            if (++idx[d] < dst->shape.dims[d]) break;
            idx[d] = 0;
        }
    }
}

namespace {

void require(bool ok, const char* what) {
    if (!ok) {
        throw std::runtime_error(std::string("znn: ") + what);
    }
}

int64_t normalize_index(int64_t index, int64_t rank) {
    int64_t i = index < 0 ? index + rank : index;
    if (i < 0 || i >= rank) {
        throw std::runtime_error("znn: shape index out of range");
    }
    return i;
}

} // namespace

// ============================================================ Shape 方法

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

int64_t& Shape::operator[](int64_t index) {
    return dims[normalize_index(index, rank)];
}

int64_t Shape::operator[](int64_t index) const {
    return dims[normalize_index(index, rank)];
}

bool Shape::operator==(const Shape& other) const {
    if (rank != other.rank) return false;
    for (int64_t d = 0; d < rank; ++d) {
        if (dims[d] != other.dims[d]) return false;
    }
    return true;
}

// ============================================================ Storage

Storage* Storage::create(size_t bytes, Device device) {
    if (device == Device::CUDA) {
        if (!cuda_backend::available()) {
            throw std::runtime_error("znn: CUDA device not available");
        }
        Storage* s = new Storage(bytes, device);
        if (bytes > 0) {
            s->data_ = cuda_backend::alloc(bytes);
        }
        return s;
    }
    check_cpu(device);
    Storage* s = new Storage(bytes, device);
    if (bytes > 0) {
        s->data_ = ::operator new(bytes);    // 未初始化, 调用方负责填充
    }
    return s;
}

Storage::Storage(size_t bytes, Device device)
    : refcount_(1), data_(nullptr), bytes_(bytes), device_(device) {
}

Storage::~Storage() {
    if (!data_) {
        return;
    }
    if (device_ == Device::CUDA) {
        cuda_backend::free(data_);
    } else {
        ::operator delete(data_);
    }
}

Storage* Storage::retain() {
    ++refcount_;
    return this;
}

void Storage::release() {
    if (--refcount_ == 0) {
        delete this;
    }
}

// ============================================================ Tensor::Impl

Tensor::Impl* Tensor::Impl::create(const Shape& shape, DType dtype, Device device) {
    require(shape.rank <= 8, "shape rank > 8");
    Impl* im = new Impl();
    im->shape = shape;
    im->dtype = dtype;
    im->device = device;
    int64_t stride = 1;
    for (int64_t d = shape.rank - 1; d >= 0; --d) {        // row-major 步长
        im->strides[d] = stride;
        stride *= shape.dims[d];
    }
    im->storage = Storage::create(static_cast<size_t>(stride) * dtype_size(dtype), device);
    return im;
}

Tensor::Impl* Tensor::Impl::retain() {
    ++refcount;
    return this;
}

void Tensor::Impl::release() {
    if (--refcount == 0) {
        delete this;    // ~Impl 递归释放 grad(Tensor 成员)与 storage
    }
}

Tensor::Impl::~Impl() {
    if (storage) {
        storage->release();
    }
}

Tensor::Impl* Tensor::Impl::make_view(const Impl* src, const Shape& shape,
                                      const int64_t (&strides)[8], int64_t offset) {
    Impl* v = new Impl();
    v->storage = src->storage ? src->storage->retain() : nullptr;
    v->offset = offset;
    v->shape = shape;
    for (int64_t d = 0; d < 8; ++d) {
        v->strides[d] = strides[d];
    }
    v->dtype = src->dtype;
    v->device = src->device;
    return v;
}

bool Tensor::Impl::is_contiguous() const {
    int64_t expect = 1;
    for (int64_t d = shape.rank - 1; d >= 0; --d) {
        if (shape.dims[d] != 1 && strides[d] != expect) {
            return false;
        }
        expect *= shape.dims[d];
    }
    return true;
}

size_t Tensor::Impl::nbytes() const {
    int64_t n = 1;
    for (int64_t d = 0; d < shape.rank; ++d) {
        n *= shape.dims[d];
    }
    return static_cast<size_t>(n) * dtype_size(dtype);
}

// ============================================================ 句柄生命周期

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

// ============================================================ 创建

Tensor Tensor::empty(Shape shape, DType dtype, Device device) {
    Tensor t;
    t.impl_ = Tensor::Impl::create(shape, dtype, device);
    return t;
}

Tensor Tensor::zeros(Shape shape, DType dtype, Device device) {
    Tensor t = Tensor::empty(shape, dtype, device);
    if (device == Device::CUDA) {
        cuda_backend::memset0(t.impl_->storage->data(), t.impl_->nbytes());
    } else {
        std::memset(t.impl_->storage->data(), 0, t.impl_->nbytes());
    }
    return t;
}

Tensor Tensor::ones(Shape shape, DType dtype, Device device) {
    // 宿主填充后按需上传(设备指针不可宿主直写)
    Tensor t = Tensor::empty(shape, dtype, Device::CPU);
    if (dtype == DType::F32) {
        float* p = t.data<float>();
        for (int64_t i = 0; i < t.numel(); ++i) p[i] = 1.f;
    } else if (dtype == DType::I64) {
        int64_t* p = t.data<int64_t>();
        for (int64_t i = 0; i < t.numel(); ++i) p[i] = 1;
    } else {
        throw std::runtime_error("znn: ones supports F32/I64 in v1");
    }
    return device == Device::CPU ? t : t.to_device(device);
}

Tensor Tensor::full(Shape shape, double value, DType dtype, Device device) {
    require(dtype == DType::F32, "full supports F32 in v1");
    Tensor t = Tensor::empty(shape, dtype, Device::CPU);
    float* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>(value);
    }
    return device == Device::CPU ? t : t.to_device(device);
}

Tensor Tensor::from_host(const void* data, Shape shape, DType dtype, Device device) {
    Tensor t = Tensor::empty(shape, dtype, device);
    if (t.numel() > 0) {
        if (device == Device::CUDA) {
            cuda_backend::copy_host_to_device(t.impl_->storage->data(), data,
                                              t.impl_->nbytes());
        } else {
            std::memcpy(t.impl_->storage->data(), data, t.impl_->nbytes());
        }
    }
    return t;
}

// ============================================================ 元信息

const Shape& Tensor::shape() const {
    require(impl_, "empty tensor");
    return impl_->shape;
}

Tensor::operator bool() const {
    return impl_ != nullptr;
}

int64_t Tensor::size(int64_t dim) const {
    require(impl_, "empty tensor");
    return impl_->shape[dim];
}

int64_t Tensor::numel() const {
    require(impl_, "empty tensor");
    int64_t n = 1;
    for (int64_t d = 0; d < impl_->shape.rank; ++d) {
        n *= impl_->shape.dims[d];
    }
    return n;
}

int64_t Tensor::dim_count() const {
    require(impl_, "empty tensor");
    return impl_->shape.rank;
}

DType Tensor::dtype() const {
    require(impl_, "empty tensor");
    return impl_->dtype;
}

Device Tensor::device() const {
    require(impl_, "empty tensor");
    return impl_->device;
}

const int64_t* Tensor::strides() const {
    require(impl_, "empty tensor");
    return impl_->strides;
}

bool Tensor::is_contiguous() const {
    require(impl_, "empty tensor");
    return impl_->is_contiguous();
}

// ============================================================ 视图

Tensor Tensor::view(Shape shape) const {
    require(impl_, "empty tensor");
    require(impl_->is_contiguous(), "view requires contiguous storage");
    return reshape(shape);
}

Tensor Tensor::reshape(Shape shape) const {
    require(impl_, "empty tensor");
    if (Runtime::grad_enabled() && impl_->requires_grad) {
        return reshape_for_autograd(*this, shape);   // 训练态: 材料化+逆梯度
    }
    if (!impl_->is_contiguous()) {
        return clone().reshape(shape);              // 非连续: 紧凑化后重试
    }
    const int64_t total = numel();
    int64_t got = 1;
    int64_t infer_at = -1;
    for (int64_t d = 0; d < shape.rank; ++d) {
        if (shape.dims[d] == -1) {
            require(infer_at < 0, "reshape: at most one -1");
            infer_at = d;
        } else {
            got *= shape.dims[d];
        }
    }
    if (infer_at >= 0) {
        require(total % got == 0, "reshape: size mismatch");
        shape.dims[infer_at] = total / got;
        got = total;
    }
    require(got == total, "reshape: size mismatch");
    int64_t strides[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t s = 1;
    for (int64_t d = shape.rank - 1; d >= 0; --d) {
        strides[d] = s;
        s *= shape.dims[d];
    }
    Tensor t;
    t.impl_ = Tensor::Impl::make_view(impl_, shape, strides, impl_->offset);
    return t;
}

Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    require(impl_, "empty tensor");
    const Shape& sp = impl_->shape;
    const int64_t a = normalize_index(start_dim, sp.rank);
    const int64_t b = normalize_index(end_dim, sp.rank);
    require(a <= b, "flatten: dim range invalid");
    Shape out;
    for (int64_t d = 0; d < a; ++d) out.dims[out.rank++] = sp.dims[d];
    int64_t n = 1;
    for (int64_t d = a; d <= b; ++d) n *= sp.dims[d];
    out.dims[out.rank++] = n;
    for (int64_t d = b + 1; d < sp.rank; ++d) out.dims[out.rank++] = sp.dims[d];
    return reshape(out);
}

Tensor Tensor::squeeze(int64_t dim) const {
    require(impl_, "empty tensor");
    const Shape& sp = impl_->shape;
    Shape out;
    if (dim == -1) {                                       // 去掉所有长度 1 的维
        for (int64_t d = 0; d < sp.rank; ++d) {
            if (sp.dims[d] != 1) out.dims[out.rank++] = sp.dims[d];
        }
    } else {
        const int64_t i = normalize_index(dim, sp.rank);
        require(sp.dims[i] == 1, "squeeze: dim is not 1");
        for (int64_t d = 0; d < sp.rank; ++d) {
            if (d != i) out.dims[out.rank++] = sp.dims[d];
        }
    }
    return reshape(out);
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    require(impl_, "empty tensor");
    const Shape& sp = impl_->shape;
    const int64_t i = normalize_index(dim, sp.rank + 1);
    Shape out;
    for (int64_t d = 0; d < i; ++d) out.dims[out.rank++] = sp.dims[d];
    out.dims[out.rank++] = 1;
    for (int64_t d = i; d < sp.rank; ++d) out.dims[out.rank++] = sp.dims[d];
    return reshape(out);
}

Tensor Tensor::slice(int64_t dim, int64_t begin, int64_t end) const {
    require(impl_, "empty tensor");
    if (Runtime::grad_enabled() && impl_->requires_grad) {
        return slice_for_autograd(*this, dim, begin, end);   // 训练态: 材料化+散射梯度
    }
    const Shape& sp = impl_->shape;
    const int64_t d = normalize_index(dim, sp.rank);
    require(begin >= 0 && end <= sp.dims[d] && begin < end, "slice: range invalid");
    Shape shape = sp;
    shape.dims[d] = end - begin;
    int64_t strides[8];
    for (int64_t k = 0; k < 8; ++k) strides[k] = impl_->strides[k];
    Tensor t;
    t.impl_ = Tensor::Impl::make_view(impl_, shape, strides,
                                      impl_->offset + begin * impl_->strides[d]);
    return t;
}

Tensor Tensor::transpose(int64_t d0, int64_t d1) const {
    require(impl_, "empty tensor");
    if (Runtime::grad_enabled() && impl_->requires_grad) {
        return transpose_for_autograd(*this, d0, d1);   // 训练态: 材料化+逆梯度
    }
    const Shape& sp = impl_->shape;
    const int64_t a = normalize_index(d0, sp.rank);
    const int64_t b = normalize_index(d1, sp.rank);
    require(a != b, "transpose: dims must differ");
    Shape shape = sp;
    int64_t strides[8];
    for (int64_t k = 0; k < 8; ++k) strides[k] = impl_->strides[k];
    shape.dims[a] = sp.dims[b];
    shape.dims[b] = sp.dims[a];
    strides[a] = impl_->strides[b];
    strides[b] = impl_->strides[a];
    Tensor t;
    t.impl_ = Tensor::Impl::make_view(impl_, shape, strides, impl_->offset);
    return t;
}

Tensor Tensor::permute(std::initializer_list<int64_t> dims) const {
    require(impl_, "empty tensor");
    if (Runtime::grad_enabled() && impl_->requires_grad) {
        return permute_for_autograd(*this, dims);       // 训练态: 材料化+逆排列梯度
    }
    const Shape& sp = impl_->shape;
    require(static_cast<int64_t>(dims.size()) == sp.rank, "permute: rank mismatch");
    Shape shape;
    shape.rank = sp.rank;
    int64_t strides[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t d = 0;
    for (int64_t k : dims) {
        const int64_t i = normalize_index(k, sp.rank);
        shape.dims[d] = sp.dims[i];
        strides[d] = impl_->strides[i];
        ++d;
    }
    Tensor t;
    t.impl_ = Tensor::Impl::make_view(impl_, shape, strides, impl_->offset);
    return t;
}

// ============================================================ 设备/拷贝

Tensor Tensor::to_device(Device device) const {
    require(impl_, "empty tensor");
    if (impl_->device == device) {
        return *this;                                      // 同设备: 共享返回(~1ns)
    }
    Tensor t = Tensor::empty(impl_->shape, impl_->dtype, device);
    if ((impl_->device == Device::CPU && device == Device::CUDA)
        || (impl_->device == Device::CUDA && device == Device::CPU)) {
        // 跨设备同步拷贝(v1 同步语义; 非连续视图先紧凑化)
        Tensor src = impl_->is_contiguous() ? *this : clone();
        if (device == Device::CUDA) {
            cuda_backend::copy_host_to_device(t.impl_->storage->data(),
                                              src.impl_->storage->data(),
                                              t.impl_->nbytes());
        } else {
            cuda_backend::copy_device_to_host(t.impl_->storage->data(),
                                              src.impl_->storage->data(),
                                              t.impl_->nbytes());
        }
        return t;
    }
    Impl::copy_elements(impl_, t.impl_);                   // CPU↔CPU
    return t;
}

Tensor Tensor::clone() const {
    require(impl_, "empty tensor");
    Tensor t = Tensor::empty(impl_->shape, impl_->dtype, impl_->device);
    Impl::copy_elements(impl_, t.impl_);
    return t;
}

// ============================================================ 数据读取

template <class T>
T* Tensor::data() const {
    require(impl_ && impl_->storage, "empty tensor");
    if ((std::is_same<T, float>::value && impl_->dtype != DType::F32) ||
        (std::is_same<T, int64_t>::value && impl_->dtype != DType::I64)) {
        throw std::runtime_error("znn: data<T> dtype mismatch");
    }
    return reinterpret_cast<T*>(static_cast<char*>(impl_->storage->data())
                                + impl_->offset * dtype_size(impl_->dtype));
}

template <class T>
T Tensor::item() const {
    require(impl_, "empty tensor");
    require(numel() == 1, "item on non-scalar");
    if (impl_->device == Device::CUDA) {                   // GPU: 自动同步拷回单值
        T v;
        cuda_backend::copy_device_to_host(&v, impl_->storage->data(),
                                          sizeof(T));
        return v;
    }
    return *data<T>();
}

template float*   Tensor::data<float>() const;
template int64_t* Tensor::data<int64_t>() const;
template float    Tensor::item<float>() const;
template int64_t  Tensor::item<int64_t>() const;

// ============================================================ 自动求导

void Tensor::requires_grad(bool yes) {
    require(impl_, "empty tensor");
    impl_->requires_grad = yes;
}

bool Tensor::requires_grad() const {
    require(impl_, "empty tensor");
    return impl_->requires_grad;
}

Tensor& Tensor::grad() {
    require(impl_, "empty tensor");
    if (!impl_->grad) {                                    // 惰性分配, 零初始化
        impl_->grad = Tensor::zeros(impl_->shape, impl_->dtype, impl_->device);
    }
    return impl_->grad;
}

void Tensor::backward() const {
    require(impl_, "empty tensor");
    AutogradEngine::instance().backward(*this);
}

Tensor Tensor::detach() const {
    require(impl_, "empty tensor");
    int64_t strides[8];
    for (int64_t k = 0; k < 8; ++k) strides[k] = impl_->strides[k];
    Tensor t;
    t.impl_ = Tensor::Impl::make_view(impl_, impl_->shape, strides, impl_->offset);
    return t;
}

// ============================================================ Runtime

namespace {
thread_local bool grad_enabled_flag = true;
}

bool Runtime::device_available(Device device) {
    if (device == Device::CPU) {
        return true;
    }
    if (device == Device::CUDA) {
        return cuda_backend::available();
    }
    return false;                                          // Metal/Vulkan 未编入
}

void Runtime::set_precision(Precision p) {
    OpDispatcher::instance().set_precision(p);
}
Precision Runtime::precision() {
    return OpDispatcher::instance().precision();
}

void Runtime::set_grad_enabled(bool yes) {
    grad_enabled_flag = yes;
}

bool Runtime::grad_enabled() {
    return grad_enabled_flag;
}

} // namespace znn
