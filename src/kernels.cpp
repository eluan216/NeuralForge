#include "neuralforge/kernels.hpp"
#if !defined(NEURALFORGE_DISABLE_SIMD) && (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define NF_AVX2 1
#endif
#if !defined(NEURALFORGE_DISABLE_SIMD) && defined(__aarch64__)
#include <arm_neon.h>
#define NF_NEON 1
#endif
namespace neuralforge {
namespace {
void scalar(float* y,const float* x,float a,std::size_t n) {
 for(std::size_t j=0;j<n;++j) y[j]+=a*x[j];
}
#if defined(NF_AVX2)
__attribute__((target("avx2")))
void avx2(float* y,const float* x,float a,std::size_t n) {
 const auto scale=_mm256_set1_ps(a);std::size_t j=0;
 for(;j+8<=n;j+=8) _mm256_storeu_ps(y+j,_mm256_add_ps(_mm256_loadu_ps(y+j),_mm256_mul_ps(scale,_mm256_loadu_ps(x+j))));
 for(;j<n;++j)y[j]+=a*x[j];
}
#endif
#if defined(NF_NEON)
void neon(float* y,const float* x,float a,std::size_t n) {
 std::size_t j=0;const auto scale=vdupq_n_f32(a);
 for(;j+4<=n;j+=4)vst1q_f32(y+j,vaddq_f32(vld1q_f32(y+j),vmulq_f32(scale,vld1q_f32(x+j))));
 for(;j<n;++j)y[j]+=a*x[j];
}
#endif
struct Dispatch {AxpyKernel fn;const char* name;};
Dispatch detect() noexcept {
#if defined(NF_AVX2)
 __builtin_cpu_init();
 if(__builtin_cpu_supports("avx2"))return {avx2,"avx2"};
#endif
#if defined(NF_NEON)
 return {neon,"neon"};
#else
 return {scalar,"scalar"};
#endif
}
const Dispatch& dispatch() noexcept {static const Dispatch value=detect();return value;}
}
AxpyKernel selected_axpy() noexcept {return dispatch().fn;}
const char* inference_backend() noexcept {return dispatch().name;}
}
