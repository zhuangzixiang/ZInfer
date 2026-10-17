// ============================================================================
// cuda_backend.hpp — CUDA 后端内部接口(znn_internal 经此调用)
//
// 职责: 设备内存分配/释放/拷贝/清零 + CUDA 算子内核工厂。
// 非CUDA 构建时由 cuda_stub.cpp 提供同名实现(明确报错/返回空内核),
// CUDA 构建时由 cuda_device.cu / cuda_kernels.cu 提供真实实现。
// 链路位置: Storage 与 Tensor::to_device 经此分发;
// 各算子文件经 shared.hpp 的内核工厂填充 kernels[CUDA] 槽位。
// ============================================================================
#pragma once
#include <cstddef>
#include <cstdint>

// 前置声明: .cu 文件必须先 include operator.hpp 再 include 本文件
namespace znn {
class Tensor;
struct KernelContext;
template <class T> struct Span;
enum class Precision : std::uint8_t;   // 定义在公共头 znn.hpp

namespace cuda_backend {

// ---- 设备管理 ----
bool   available();                                   // 是否编入且设备可用
void*  alloc(size_t bytes);                           // cudaMalloc
void   free(void* p);                                 // cudaFree
void   memset0(void* p, size_t bytes);                // cudaMemset(0)
void   copy_host_to_device(void* dst, const void* src, size_t bytes);
void   copy_device_to_host(void* dst, const void* src, size_t bytes);
void   copy_device_to_device(void* dst, const void* src, size_t bytes);

// ---- CUDA 算子内核工厂(非CUDA 构建返回 nullptr, 派发器走 CPU 兜底) ----
// 逐元素二元
void add_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sub_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void mul_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void div_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
// 逐元素一元
void neg_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void exp_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void log_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sqrt_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void relu_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void gelu_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sigmoid_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void tanh_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void silu_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
// 第二批(softmax/规约/损失/卷积/池化)
void softmax_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sum_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void mean_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void cross_entropy_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void mse_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void conv2d_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void max_pool2d_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void avg_pool2d_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
// 矩阵
void matmul_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);

// 第三批(embedding/视图族/cat/stack/arange/randn/LN/BN/attention)
void embedding_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void reshape_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void transpose_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void permute_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void slice_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void cat_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void stack_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void arange_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void randn_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void layer_norm_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void batch_norm2d_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void linear_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void attention_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);

// conv2d 梯度(专用, 铁律: 不允许 CPU 桥接)
void conv2d_grad_cuda(const float* x, const float* w, const float* g,
                      float* dx, float* dw,
                      int N, int C, int H, int W, int O,
                      int KH, int KW, int HO, int WO,
                      int sh, int sw, int ph, int pw, int dh, int dw_, int G);

// ---- 共享 GEMM 原语(cuBLAS 委托, 实现: cuda_gemm.cu) ----
// 行主序 C[M,N] = A[M,K] × op(B); ld 为行距(列切片时传母矩阵整宽);
// b_transposed=true 时 B 按 [N,K] 行主序传入(× w^T, 权重布局免转置拷贝)
// 返回 false = cuBLAS 不可用, 调用方走朴素内核兜底
// (仅 CUDA 内核内部使用, 无 stub 版本)
#if defined(ZNN_CUDA)
bool gemm_cuda(const float* A, int lda, const float* B, int ldb,
               float* C, int ldc, int M, int N, int K,
               bool b_transposed, Precision p);
bool gemm_batched_cuda(const float* A, int lda, long long sa,
                       const float* B, int ldb, long long sb,
                       float* C, int ldc, long long sc,
                       int M, int N, int K, int batch,
                       bool b_transposed, Precision p);
#endif

// ---- 优化器 CUDA 入口(审计修复2) ----
void sgd_step_cuda(float* p, const float* g, float* vel, long long n,
                   float lr, float momentum, float wd);
void adam_step_cuda(float* p, const float* g, float* m, float* v, long long n,
                    float lr, float b1, float b2, float eps, float wd,
                    float bias1, float bias2);
void memset0_cuda(void* p, size_t bytes);

// unary 反向内部算子(options = UnaryBwd 枚举一字节)
void unary_bwd_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
// 梯度专用内核(反向路径, 铁律: 原生无桥接)
// 通用广播/置换: out[i] = scale × g[Σ idx[d]·eff[d]](sum/mean/permute 反向)
void grad_bcast_cuda(const float* g, float* out, long long total, int rank,
                     const long long* dims, const long long* eff, float scale);
void max_pool_grad_cuda(const float* x, const float* g, float* dx, long long total,
                        int H, int W, int HO, int WO,
                        int kh, int kw, int sh, int sw, int ph, int pw);
void avg_pool_grad_cuda(const float* g, float* dx, long long total,
                        int H, int W, int HO, int WO,
                        int kh, int kw, int sh, int sw, int ph, int pw);
void embedding_grad_cuda(const float* g, const long long* idx, float* dw,
                         long long total, int D);
// 通用双步长拷贝(源/目标各自步长, 视图散射写入共用)
void strided_copy_cuda(const float* src, float* dst, int rank,
                       const long long* ss, const long long* ds,
                       const long long* sh, long long total);
// GradScaler: 梯度 ×inv, 发现 inf/nan 时 *bad = 1(调用方置零初始化)
void grad_unscale_cuda(float* g, float* bad, long long n, float inv);


} // namespace cuda_backend
} // namespace znn
