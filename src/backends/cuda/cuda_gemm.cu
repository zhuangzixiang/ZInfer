// ============================================================================
// cuda_gemm.cu — CUDA 后端共享 GEMM 原语(全部 GEMM 类算子的唯一 cuBLAS 入口)
//
// 行主序语义: C[M,N] = A[M,K] × op(B)
//   b_transposed=false: B 为 [K,N] 行主序(matmul)
//   b_transposed=true : B 为 [N,K] 行主序(权重布局, 即 × w^T, 免转置拷贝)
// 列主序映射: C 行主序内存 = C^T 列主序[N,M] = op(B) × A^T;
//   NN 时 op(B)=B^T 恰为 B 内存的列主序视图(OP_N);
//   NT 时 op(B)=B 须用 OP_T 显式转置(ld 只改视图不改内容朝向)
// ld 为各矩阵行距(元素数), 允许 ld > 行长(母矩阵列切片, conv2d 分组用)
// 精度: FP16 → 输入转半精度 Tensor Core(累加/输出仍 FP32);
//       TF32/AUTO → FAST_TF32; FP32 或 Ex 不可用 → sgemm
// 返回 false = cuBLAS 不可用或调用失败, 调用方走朴素内核兜底(结果仍正确)
// ============================================================================
#include "operator.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "dynlib.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace znn {
namespace cuda_backend {

namespace {
constexpr int kGemmBlk = 256;

// 列主序核心: gemm(m=N, n=M, k=K, A'=B, B'=A, C), B' 恒为 OP_N
cublasOperation_t b_op(bool b_transposed) {
    return b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
}

// FP16 转换: 按 [batch, rows, cols] 视图(x/y 同布局, 行距 ld 批距 bs)
__global__ void cast_f32_f16_k(const float* x, __half* y, long long total,
                               int rows, int cols, int ld, long long bs) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i % cols);
    const long long t = i / cols;
    const long long idx = t / rows * bs + t % rows * ld + c;
    y[idx] = __float2half(x[idx]);
}

// 半精度副本转换(A/B 布局参数与 GEMM 一致), 返回 false = 分配失败
bool cast_f32_f16(const float* x, __half* y, long long rows, long long cols,
                  int ld, long long bs, int batch) {
    const long long total = batch * rows * cols;
    cast_f32_f16_k<<<(unsigned)((total + kGemmBlk - 1) / kGemmBlk), kGemmBlk>>>(
        x, y, total, (int)rows, (int)cols, ld, bs);
    return true;
}
} // namespace

bool gemm_cuda(const float* A, int lda, const float* B, int ldb,
               float* C, int ldc, int M, int N, int K,
               bool b_transposed, Precision p) {
    if (!dynlib::available_cublas()) return false;
    const float alpha = 1.0f, beta = 0.0f;
    if (p == Precision::FP16 && dynlib::cublas_gemm_ex) {
        // 半精度副本: 读取带宽减半 + FP16 Tensor Core; 失败落到 TF32。
        // 转换/分配按真实存储形状 [rows, ld](列切片时 ld > 行宽)
        const long long a_rows = M, b_rows = b_transposed ? N : K;
        __half* ha = (__half*)cuda_backend::alloc(sizeof(__half) * (size_t)a_rows * lda);
        __half* hb = (__half*)cuda_backend::alloc(sizeof(__half) * (size_t)b_rows * ldb);
        bool ok = ha && hb;
        if (ok) ok = cast_f32_f16(A, ha, M, K, lda, 0, 1);
        if (ok) ok = cast_f32_f16(B, hb, b_rows, b_transposed ? K : N, ldb, 0, 1);
        if (ok) {
            ok = dynlib::cublas_gemm_ex(
                     dynlib::cublas_handle, b_op(b_transposed), CUBLAS_OP_N,
                     N, M, K, &alpha,
                     hb, CUDA_R_16F, ldb,
                     ha, CUDA_R_16F, lda, &beta,
                     C, CUDA_R_32F, ldc,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                 == CUBLAS_STATUS_SUCCESS;
        }
        cuda_backend::free(ha);
        cuda_backend::free(hb);
        if (ok) return true;
    }
    if (p != Precision::FP32 && dynlib::cublas_gemm_ex) {
        return dynlib::cublas_gemm_ex(
                   dynlib::cublas_handle, b_op(b_transposed), CUBLAS_OP_N,
                   N, M, K, &alpha,
                   B, CUDA_R_32F, ldb,
                   A, CUDA_R_32F, lda, &beta,
                   C, CUDA_R_32F, ldc,
                   CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT)
            == CUBLAS_STATUS_SUCCESS;
    }
    return dynlib::cublas_sgemm(
               dynlib::cublas_handle, b_op(b_transposed), CUBLAS_OP_N,
               N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc)
        == CUBLAS_STATUS_SUCCESS;
}

// 批量版: batch 个同形状 GEMM, 矩阵间元素跨度 sa/sb/sc(attention 按头并行)
bool gemm_batched_cuda(const float* A, int lda, long long sa,
                       const float* B, int ldb, long long sb,
                       float* C, int ldc, long long sc,
                       int M, int N, int K, int batch,
                       bool b_transposed, Precision p) {
    if (!dynlib::available_cublas()) return false;
    const float alpha = 1.0f, beta = 0.0f;
    if (p == Precision::FP16 && dynlib::cublas_gemm_batched_ex) {
        const long long a_rows = M, b_rows = b_transposed ? N : K;
        const size_t a_elems = (size_t)(batch - 1) * sa + (size_t)a_rows * lda;
        const size_t b_elems = (size_t)(batch - 1) * sb + (size_t)b_rows * ldb;
        __half* ha = (__half*)cuda_backend::alloc(sizeof(__half) * a_elems);
        __half* hb = (__half*)cuda_backend::alloc(sizeof(__half) * b_elems);
        bool ok = ha && hb;
        if (ok) ok = cast_f32_f16(A, ha, M, K, lda, sa, batch);
        if (ok) ok = cast_f32_f16(B, hb, b_rows, b_transposed ? K : N, ldb, sb, batch);
        if (ok) {
            ok = dynlib::cublas_gemm_batched_ex(
                     dynlib::cublas_handle, b_op(b_transposed), CUBLAS_OP_N,
                     N, M, K, &alpha,
                     hb, CUDA_R_16F, ldb, sb,
                     ha, CUDA_R_16F, lda, sa, &beta,
                     C, CUDA_R_32F, ldc, sc,
                     batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                 == CUBLAS_STATUS_SUCCESS;
        }
        cuda_backend::free(ha);
        cuda_backend::free(hb);
        if (ok) return true;
    }
    if (p != Precision::FP32 && dynlib::cublas_gemm_batched_ex) {
        return dynlib::cublas_gemm_batched_ex(
                   dynlib::cublas_handle, b_op(b_transposed), CUBLAS_OP_N,
                   N, M, K, &alpha,
                   B, CUDA_R_32F, ldb, sb,
                   A, CUDA_R_32F, lda, sa, &beta,
                   C, CUDA_R_32F, ldc, sc,
                   batch, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT)
            == CUBLAS_STATUS_SUCCESS;
    }
    if (!dynlib::cublas_sgemm_batched) return false;
    return dynlib::cublas_sgemm_batched(
               dynlib::cublas_handle, b_op(b_transposed), CUBLAS_OP_N,
               N, M, K, &alpha, B, ldb, sb, A, lda, sa, &beta, C, ldc, sc,
               batch)
        == CUBLAS_STATUS_SUCCESS;
}

} // namespace cuda_backend
} // namespace znn
