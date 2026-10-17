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
// 精度: TF32/AUTO 且 GemmEx 可用 → Tensor Core(FAST_TF32), 否则纯 FP32
// 返回 false = cuBLAS 不可用或调用失败, 调用方走朴素内核兜底(结果仍正确)
// ============================================================================
#include "operator.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "dynlib.hpp"

namespace znn {
namespace cuda_backend {

namespace {
// 列主序核心: gemm(m=N, n=M, k=K, A'=B, B'=A, C), B' 恒为 OP_N
cublasOperation_t b_op(bool b_transposed) {
    return b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
}
} // namespace

bool gemm_cuda(const float* A, int lda, const float* B, int ldb,
               float* C, int ldc, int M, int N, int K,
               bool b_transposed, Precision p) {
    if (!dynlib::available_cublas()) return false;
    const float alpha = 1.0f, beta = 0.0f;
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
