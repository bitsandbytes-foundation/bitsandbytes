#pragma once
#include <limits>
#include <cmath>
#include <stdexcept>
#include <sstream>

#ifdef __FMA__
#define USE_FMA
#endif

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

} // namepsace Details
} // namespace BinSearch
