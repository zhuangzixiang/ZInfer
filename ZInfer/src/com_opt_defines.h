#ifndef __ZINFER_COM_OPT_DEFINE_H__
#define __ZINFER_COM_OPT_DEFINE_H__

#ifdef _WIN32
#  define USE_OPT_SIMD
#  include <nmmintrin.h>
#endif

#ifdef __APPLE__
#  define USE_OPT_SIMD
#  ifdef __aarch64__
#    include "sse2neon.h"
#  else
#    include <nmmintrin.h>
#  endif
#endif





/**************************************************/
/*  函数映射                                       */
/**************************************************/
#include <immintrin.h>
#include <type_traits>

// 主模板声明
template<typename T, typename Enable = void>
struct SimdType;

// float
template<>
struct SimdType<float>
{
    using scalar_type = float;
    using vec_type = __m256;
    using vec_typei = __m256i;
    static constexpr int alignment = 32;
    static constexpr int elements_per_vector = 8;

    // 加载/存储操作
    static vec_type load(const scalar_type* ptr) { return _mm256_loadu_ps(ptr); }
    static void store(scalar_type* ptr, vec_type a) { _mm256_storeu_ps(ptr, a); }

    // 算术运算
    static vec_type add(vec_type a, vec_type b) { return _mm256_add_ps(a, b); }
    static vec_type mul(vec_type a, vec_type b) { return _mm256_mul_ps(a, b); }
    static vec_type fmadd(vec_type a, vec_type b, vec_type c) { return _mm256_fmadd_ps(a, b, c); }

    // 特殊操作
    static vec_type zero() { return _mm256_setzero_ps(); }
    static vec_type set1(scalar_type val) { return _mm256_set1_ps(val); }

    // 比较操作
    static vec_type cmp_gt(vec_type a, vec_type b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
    static vec_type blend(vec_type a, vec_type b, vec_type mask) { return _mm256_blendv_ps(a, b, mask); }

    // 类型转换
    static vec_typei cast_to_int(vec_type a) { return _mm256_castps_si256(a); }
    static vec_type cast_from_int(vec_typei a) { return _mm256_castsi256_ps(a); }

    // 水平求和
    static scalar_type horizontal_sum(vec_type v) {
        // 步骤1：高低128位分别求和
        __m256 v_low = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)); // 交换双字
        __m256 v_sum1 = _mm256_add_ps(v, v_low);

        // 步骤2：跨通道求和
        __m256 v_perm = _mm256_permute2f128_ps(v_sum1, v_sum1, 0x01);
        __m256 v_sum2 = _mm256_add_ps(v_sum1, v_perm);

        // 步骤3：提取最终结果
        return _mm_cvtss_f32(_mm256_castps256_ps128(v_sum2)) +
            _mm_cvtss_f32(_mm256_extractf128_ps(v_sum2, 1));
    }
};

// double
template<>
struct SimdType<double>
{
    using scalar_type = double;
    using vec_type = __m256d;
    using vec_typei = __m256i;
    static constexpr int alignment = 32;
    static constexpr int elements_per_vector = 4;

    static vec_type load(const scalar_type* ptr) { return _mm256_loadu_pd(ptr); }
    static void store(scalar_type* ptr, vec_type a) { _mm256_storeu_pd(ptr, a); }

    static vec_type add(vec_type a, vec_type b) { return _mm256_add_pd(a, b); }
    static vec_type mul(vec_type a, vec_type b) { return _mm256_mul_pd(a, b); }
    static vec_type fmadd(vec_type a, vec_type b, vec_type c) { return _mm256_fmadd_pd(a, b, c); }

    static vec_type zero() { return _mm256_setzero_pd(); }
    static vec_type set1(scalar_type val) { return _mm256_set1_pd(val); }

    static vec_type cmp_gt(vec_type a, vec_type b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }
    static vec_type blend(vec_type a, vec_type b, vec_type mask) { return _mm256_blendv_pd(a, b, mask); }

    static vec_typei cast_to_int(vec_type a) { return _mm256_castpd_si256(a); }
    static vec_type cast_from_int(vec_typei a) { return _mm256_castsi256_pd(a); }

    // 水平求和
    static scalar_type horizontal_sum(vec_type v) {
        // 步骤1：高低128位分别求和
        __m256d v_low = _mm256_shuffle_pd(v, v, 0x5); // 交换双精度
        __m256d v_sum1 = _mm256_add_pd(v, v_low);

        // 步骤2：跨通道求和
        __m256d v_perm = _mm256_permute2f128_pd(v_sum1, v_sum1, 0x01);
        __m256d v_sum2 = _mm256_add_pd(v_sum1, v_perm);

        // 提取结果
        return _mm_cvtsd_f64(_mm256_castpd256_pd128(v_sum2)) +
            _mm_cvtsd_f64(_mm256_extractf128_pd(v_sum2, 1));
    }
};

// int32_t
template<>
struct SimdType<int32_t>
{
    using scalar_type = int32_t;
    using vec_type = __m256i;
    static constexpr int alignment = 32;
    static constexpr int elements_per_vector = 8;

    static vec_type load(const scalar_type* ptr) { return _mm256_loadu_si256(reinterpret_cast<const vec_type*>(ptr)); }
    static void store(scalar_type* ptr, vec_type a) { _mm256_storeu_si256(reinterpret_cast<vec_type*>(ptr), a); }

    static vec_type add(vec_type a, vec_type b) { return _mm256_add_epi32(a, b); }
    static vec_type mul(vec_type a, vec_type b) { return _mm256_mullo_epi32(a, b); }
    static vec_type fmadd(vec_type a, vec_type b, vec_type c) { return _mm256_add_epi32(_mm256_mullo_epi32(a, b), c); }

    static vec_type zero() { return _mm256_setzero_si256(); }
    static vec_type set1(scalar_type val) { return _mm256_set1_epi32(val); }

    static vec_type cmp_gt(vec_type a, vec_type b) { return _mm256_cmpgt_epi32(a, b); }
    static vec_type blend(vec_type a, vec_type b, vec_type mask) {
        return _mm256_blendv_epi8(a, b, mask);
    }

    // 转换到浮点
    static __m256 convert_to_float(vec_type a) { return _mm256_cvtepi32_ps(a); }
    static vec_type convert_from_float(__m256 a) { return _mm256_cvtps_epi32(a); }

    // 水平求和
    static scalar_type horizontal_sum(vec_type v) {
        // 步骤1：高低128位分别相加
        __m256i v_low = _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)); // 交换高低双字
        __m256i v_sum1 = _mm256_add_epi32(v, v_low);

        // 步骤2：在128位通道内相加
        __m256i v_perm = _mm256_permute2x128_si256(v_sum1, v_sum1, 0x01); // 交换高低128位
        __m256i v_sum2 = _mm256_add_epi32(v_sum1, v_perm);

        // 步骤3：提取结果
        return _mm_extract_epi32(_mm256_castsi256_si128(v_sum2), 0) + _mm_extract_epi32(_mm256_castsi256_si128(v_sum2), 1);
    }
};

// uint32_t 类型特化（需要AVX2）
template<>
struct SimdType<uint32_t>
{
    using scalar_type = uint32_t;
    using vec_type = __m256i;
    static constexpr int alignment = 32;
    static constexpr int elements_per_vector = 8;

    static vec_type load(const scalar_type* ptr) { return _mm256_loadu_si256(reinterpret_cast<const vec_type*>(ptr)); }
    static void store(scalar_type* ptr, vec_type a) { _mm256_storeu_si256(reinterpret_cast<vec_type*>(ptr), a); }

    static vec_type add(vec_type a, vec_type b) { return _mm256_add_epi32(a, b); }
    static vec_type mul(vec_type a, vec_type b) { return _mm256_mullo_epi32(a, b); }
    static vec_type fmadd(vec_type a, vec_type b, vec_type c) { return _mm256_add_epi32(_mm256_mullo_epi32(a, b), c); }

    static vec_type zero() { return _mm256_setzero_si256(); }
    static vec_type set1(scalar_type val) { return _mm256_set1_epi32(static_cast<int32_t>(val)); }

    // 无符号比较需要特殊处理
    static vec_type cmp_gt(vec_type a, vec_type b) {
        return _mm256_cmpgt_epi32(_mm256_xor_si256(a, _mm256_set1_epi32(0x80000000)),
            _mm256_xor_si256(b, _mm256_set1_epi32(0x80000000)));
    }

    static vec_type blend(vec_type a, vec_type b, vec_type mask) {
        return _mm256_blendv_epi8(a, b, mask);
    }

    static __m256 convert_to_float(vec_type a) { return _mm256_cvtepu32_ps(a); }
    static vec_type convert_from_float(__m256 a) { return _mm256_cvtps_epu32(a); }

    // 水平求和（复用int32_t的实现）
    static scalar_type horizontal_sum(vec_type v) {
        return static_cast<uint32_t>(
            SimdType<int32_t>::horizontal_sum(v)
            );
    }
};

#if 0
// 使用示例模板函数
template<typename T>
void simd_add(const T* a, const T* b, T* result, size_t n)
{
    using simd = SimdType<T>;
    constexpr int vec_size = simd::elements_per_vector;

    size_t i = 0;
    for (; i + vec_size <= n; i += vec_size) {
        auto va = simd::load(a + i);
        auto vb = simd::load(b + i);
        auto vres = simd::add(va, vb);
        simd::store(result + i, vres);
    }

    // 处理尾部数据
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}
#endif




#endif //__ZINFER_TENSOR_H__