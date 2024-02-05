#pragma once
#include <limits>
#include <cmath>
#include <stdexcept>
#include <sstream>

#if defined(__aarch64__)
#ifdef __CUDACC__
#undef USE_NEON // Doesn't work with nvcc, undefined symbols
#else
#include <arm_neon.h>
#undef USE_NEON // Not yet implemented
#endif
#undef USE_AVX // x86_64 only
#undef USE_AVX2 // x86_64 only
#undef USE_SSE2 // x86_64 only
#undef USE_SSE41 // x86_64 only
#undef USE_SSE42 // x86_64 only
#undef USE_FMA // x86_64 only
#ifdef USE_NEON
typedef float32x4_t __m128;
typedef int32x4_t __m128i;
typedef float64x2_t __m128d;
#else
typedef struct {float a; float b; float c; float d;} __m128;
typedef struct {int a; int b; int c; int d;} __m128i;
typedef struct {double a; double b;} __m128d;
#endif
#else
#undef USE_NEON // ARM64 only
#ifdef __FMA__
#define USE_FMA
#endif
#if !defined(__SSE2__) && !defined(_MSC_VER)
#error Compiler must support SSE2
#endif
#define USE_SSE2

#if defined(__aarch64__)
#else
#ifdef __AVX2__
#define USE_AVX2
#endif

#ifdef __AVX__
#define USE_AVX
#endif


#ifdef __SSE4_1__
#define USE_SSE41
#endif

#ifdef __SSE4_2__
#define USE_SSE42
#endif
#endif
#endif

#ifndef _MSC_VER
#include <stdint.h>
#endif

namespace BinSearch {

#ifndef _MSC_VER
typedef  int8_t   int8;
typedef uint8_t  uint8;
typedef  int32_t   int32;
typedef uint32_t  uint32;
typedef  int64_t   int64;
typedef uint64_t  uint64;
#else
typedef  __int8   int8;
typedef unsigned __int8  uint8;
typedef  __int32   int32;
typedef unsigned __int32  uint32;
typedef  __int64   int64;
typedef unsigned __int64  uint64;
#endif

namespace Details {

#define myassert(cond, msg) if (!cond){ std::ostringstream os; os << "\nassertion failed: " << #cond << ", " << msg << "\n"; throw std::invalid_argument(os.str()); }

// log2 is not defined in VS2008
#if defined(_MSC_VER)
inline uint32 log2 (uint32 val) {
    if (val == 1) return 0;
    uint32 ret = 0;
    do {
        ret++;
        val >>= 1;
    } while (val > 1);
    return ret;
}
#endif

#ifdef _DEBUG
#define DEBUG
#endif

#ifdef _MSC_VER
#   define FORCE_INLINE __forceinline
#   define NO_INLINE __declspec(noinline)
#else
#   define NO_INLINE __attribute__((noinline))
#   ifdef DEBUG
#       define FORCE_INLINE NO_INLINE
#   else
#       define FORCE_INLINE __attribute__((always_inline)) inline
#   endif
#endif

#ifdef USE_AVX
#define COMISS "vcomiss"
#define COMISD "vcomisd"
#else
#define COMISS "comiss"
#define COMISD "comisd"
#endif

// nextafter is not defined in VS2008
#if defined(_MSC_VER) && (_MSC_VER <= 1500)
#include <float.h>
inline float mynext(float x)
{
    return _nextafterf(x, std::numeric_limits<float>::max());
}

inline double mynext(double x)
{
    return _nextafter(x, std::numeric_limits<double>::max());
}
inline float myprev(float x)
{
    return _nextafterf(x, -std::numeric_limits<float>::max());
}

inline double myprev(double x)
{
    return _nextafter(x, -std::numeric_limits<double>::max());
}
#else
inline float mynext(float x)
{
    return std::nextafterf(x, std::numeric_limits<float>::max());
}

inline double mynext(double x)
{
    return std::nextafter(x, std::numeric_limits<double>::max());
}
inline float myprev(float x)
{
    return std::nextafterf(x, -std::numeric_limits<float>::max());
}

inline double myprev(double x)
{
    return std::nextafter(x, -std::numeric_limits<double>::max());
}
#endif

template <typename T>
inline T next(T x)
{
    for (int i = 0; i < 4; ++i)
        x = mynext(x);
    return x;
}

template <typename T>
inline T prev(T x)
{
    for (int i = 0; i < 4; ++i)
        x = myprev(x);
    return x;
}

} // namespace Details
} // namespace BinSearch
