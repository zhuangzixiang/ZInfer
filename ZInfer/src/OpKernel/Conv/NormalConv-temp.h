
#ifndef __NORMAL_CONVOLUTION_OPT_H__
#define __NORMAL_CONVOLUTION_OPT_H__

namespace zinfer
{

#include <vector>
#include <iostream>
#include <numeric>
#include <chrono>
#include <stdexcept>

// 平台特定的头文件
#if defined(_MSC_VER)
#include <malloc.h>    // Windows
#elif defined(__ARM_NEON)
#include <arm_neon.h> // ARM NEON
#else
#include <immintrin.h> // x86 SIMD
#endif

// 内存对齐分配器（跨平台实现）
    template <typename T, size_t Alignment = 64>
    class AlignedAllocator {
    public:
        using value_type = T;

        AlignedAllocator() = default;

        template <class U>
        AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

        T* allocate(size_t n) {
            if (n > std::numeric_limits<size_t>::max() / sizeof(T))
                throw std::bad_alloc();

            size_t bytes = n * sizeof(T);
#if defined(_MSC_VER)
            void* p = _aligned_malloc(bytes, Alignment);
#else
            void* p = aligned_alloc(Alignment, bytes);
#endif
            if (!p) throw std::bad_alloc();
            return static_cast<T*>(p);
        }

        void deallocate(T* p, size_t) {
#if defined(_MSC_VER)
            _aligned_free(p);
#else
            free(p);
#endif
        }

        template <class U>
        struct rebind { using other = AlignedAllocator<U, Alignment>; };

        bool operator==(const AlignedAllocator&) const { return true; }
        bool operator!=(const AlignedAllocator&) const { return false; }
    };

    using AlignedVector = std::vector<float, AlignedAllocator<float>>;

    // SIMD配置
#if defined(__AVX512F__)
#define SIMD_ALIGN 64
    using simd_type = __m512;
    constexpr size_t SIMD_WIDTH = 16;
#elif defined(__AVX2__)
#define SIMD_ALIGN 32
    using simd_type = __m256;
    constexpr size_t SIMD_WIDTH = 8;
#elif defined(__SSE4_1__)
#define SIMD_ALIGN 16
    using simd_type = __m128;
    constexpr size_t SIMD_WIDTH = 4;
#elif defined(__ARM_NEON)
#define SIMD_ALIGN 16
    using simd_type = float32x4_t;
    constexpr size_t SIMD_WIDTH = 4;
#else
#define SIMD_ALIGN 4
    constexpr size_t SIMD_WIDTH = 1;
#endif

    // 水平归约函数
#if defined(__AVX512F__)
    inline float horizontal_sum(__m512 v) {
        return _mm512_reduce_add_ps(v);
    }
#elif defined(__AVX2__)
    inline float horizontal_sum(__m256 v) {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
    }
#elif defined(__SSE4_1__)
    inline float horizontal_sum(__m128 v) {
        __m128 shuf = _mm_movehdup_ps(v);
        __m128 sums = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
    }
#elif defined(__ARM_NEON)
    inline float horizontal_sum(float32x4_t v) {
        float32x2_t sum = vadd_f32(vget_high_f32(v), vget_low_f32(v));
        return vget_lane_f32(vpadd_f32(sum, sum), 0);
    }
#endif

    // 优化的SIMD点积实现
    float optimized_dot(const AlignedVector& a, const AlignedVector& b) {
        const size_t n = a.size();
        float sum = 0.0f;

#if (SIMD_WIDTH > 1)
        // SIMD部分
        simd_type sum_vec =
#if defined(__AVX512F__)
            _mm512_setzero_ps();
#elif defined(__AVX2__)
            _mm256_setzero_ps();
#elif defined(__SSE4_1__)
            _mm_setzero_ps();
#elif defined(__ARM_NEON)
            vdupq_n_f32(0);
#endif

        const size_t block_size = 4 * SIMD_WIDTH;
        size_t i = 0;

        // 主处理循环（带4路展开）
        for (; i + block_size <= n; i += block_size) {
#if defined(__AVX512F__)
            sum_vec = _mm512_fmadd_ps(_mm512_load_ps(&a[i]),
                _mm512_load_ps(&b[i]), sum_vec);
            sum_vec = _mm512_fmadd_ps(_mm512_load_ps(&a[i + 16]),
                _mm512_load_ps(&b[i + 16]), sum_vec);
            sum_vec = _mm512_fmadd_ps(_mm512_load_ps(&a[i + 32]),
                _mm512_load_ps(&b[i + 32]), sum_vec);
            sum_vec = _mm512_fmadd_ps(_mm512_load_ps(&a[i + 48]),
                _mm512_load_ps(&b[i + 48]), sum_vec);
#elif defined(__AVX2__)
            sum_vec = _mm256_fmadd_ps(_mm256_load_ps(&a[i]),
                _mm256_load_ps(&b[i]), sum_vec);
            sum_vec = _mm256_fmadd_ps(_mm256_load_ps(&a[i + 8]),
                _mm256_load_ps(&b[i + 8]), sum_vec);
            sum_vec = _mm256_fmadd_ps(_mm256_load_ps(&a[i + 16]),
                _mm256_load_ps(&b[i + 16]), sum_vec);
            sum_vec = _mm256_fmadd_ps(_mm256_load_ps(&a[i + 24]),
                _mm256_load_ps(&b[i + 24]), sum_vec);
#elif defined(__SSE4_1__)
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(_mm_load_ps(&a[i]),
                _mm_load_ps(&b[i])));
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(_mm_load_ps(&a[i + 4]),
                _mm_load_ps(&b[i + 4])));
#elif defined(__ARM_NEON)
            sum_vec = vmlaq_f32(sum_vec, vld1q_f32(&a[i]), vld1q_f32(&b[i]));
            sum_vec = vmlaq_f32(sum_vec, vld1q_f32(&a[i + 4]), vld1q_f32(&b[i + 4]));
#endif
        }

        // 剩余SIMD块处理
        for (; i + SIMD_WIDTH <= n; i += SIMD_WIDTH) {
#if defined(__AVX512F__)
            sum_vec = _mm512_fmadd_ps(_mm512_load_ps(&a[i]),
                _mm512_load_ps(&b[i]), sum_vec);
#elif defined(__AVX2__)
            sum_vec = _mm256_fmadd_ps(_mm256_load_ps(&a[i]),
                _mm256_load_ps(&b[i]), sum_vec);
#elif defined(__SSE4_1__)
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(_mm_load_ps(&a[i]),
                _mm_load_ps(&b[i])));
#elif defined(__ARM_NEON)
            sum_vec = vmlaq_f32(sum_vec, vld1q_f32(&a[i]), vld1q_f32(&b[i]));
#endif
        }

        sum += horizontal_sum(sum_vec);
#endif

        // 处理尾部标量元素
        for (; i < n; ++i) {
            sum += a[i] * b[i];
        }

        return sum;
        }

    // 并行化版本（使用OpenMP）
    float parallel_dot(const AlignedVector& a, const AlignedVector& b) {
        const size_t n = a.size();
        constexpr size_t CACHE_LINE_SIZE = 64 / sizeof(float);
        float sum = 0.0f;

#pragma omp parallel reduction(+:sum)
        {
#pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; i += CACHE_LINE_SIZE) {
                const size_t end = std::min(i + CACHE_LINE_SIZE, n);
                float local_sum = 0.0f;

#if (SIMD_WIDTH > 1)
                simd_type sum_vec =
#if defined(__AVX512F__)
                    _mm512_setzero_ps();
#elif defined(__AVX2__)
                    _mm256_setzero_ps();
#elif defined(__SSE4_1__)
                    _mm_setzero_ps();
#elif defined(__ARM_NEON)
                    vdupq_n_f32(0);
#endif

                size_t j = i;
                for (; j + SIMD_WIDTH <= end; j += SIMD_WIDTH) {
#if defined(__AVX512F__)
                    sum_vec = _mm512_fmadd_ps(_mm512_load_ps(&a[j]),
                        _mm512_load_ps(&b[j]), sum_vec);
#elif defined(__AVX2__)
                    sum_vec = _mm256_fmadd_ps(_mm256_load_ps(&a[j]),
                        _mm256_load_ps(&b[j]), sum_vec);
#elif defined(__SSE4_1__)
                    sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(_mm_load_ps(&a[j]),
                        _mm_load_ps(&b[j])));
#elif defined(__ARM_NEON)
                    sum_vec = vmlaq_f32(sum_vec, vld1q_f32(&a[j]), vld1q_f32(&b[j]));
#endif
                }
                local_sum += horizontal_sum(sum_vec);

                // 处理尾部元素
                for (; j < end; ++j) {
                    local_sum += a[j] * b[j];
                }
#else
                for (size_t j = i; j < end; ++j) {
                    local_sum += a[j] * b[j];
                }
#endif

                sum += local_sum;
            }
        }

        return sum;
    }

}

#endif // __NORMAL_CONVOLUTION_H__