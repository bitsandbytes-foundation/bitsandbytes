 #pragma once

#include <stddef.h>
#include <vector>
#include <limits>

#include "Portable.h"

using std::size_t;

namespace BinSearch {

enum InstrSet { Scalar, SSE, AVX, Neon };

#define ALGOENUM(x, b) x,
enum Algos
    {
#include "AlgoXCodes.h"
    };
#undef ALGOENUM

namespace Details {

    template <InstrSet I>
    struct InstrIntTraits;

    template <InstrSet I, typename T>
    struct InstrFloatTraits;

    // base class for algorithm supporting the method:
    //    uint32 scalar(T z) const
    template <typename T, Algos A, typename Enable=void>
    struct AlgoScalarBase;

    // base class for algorithm supporting the following methods, constants and definitions:
    //    static const uint32 nElem
    //    struct Constants;
    //    void initConstants(Constants& cst) const
    //    void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    // The function vectorial processes nElem items
    template <InstrSet I, typename T, Algos A, typename Enable=void>
    struct AlgoVecBase;

    template <typename T> struct IntTraits;

    template <> struct IntTraits<float>
    {
        typedef uint32 itype;
    };
    template <> struct IntTraits<double>
    {
        typedef uint64 itype;
    };

    template <int N>
    struct Body
    {
        template <uint32 D, typename T, typename Expr>
        FORCE_INLINE static void iteration(const Expr& e, uint32 *ri, const T* zi, const typename Expr::Constants& cst)
        {
            e.vectorial(ri, zi, cst);
            Body<N - 1>::template iteration<D>(e, ri + D, zi + D, cst);
        }

    };

    template <>
    struct Body<0>
    {
        template <uint32 D, typename T, typename Expr, typename H>
        FORCE_INLINE static void iteration(const Expr& e, uint32 *ri, const T* zi, const H&)
        {
        }
    };

    template <typename T, typename Algo>
    struct Loop
    {
        typedef Algo algo_type;
        static const uint32 M = 4;
        static const uint32 D = algo_type::nElem;

        FORCE_INLINE static void loop(const algo_type& e, uint32 *ri, const T* zi, uint32 n)
        {
            typename algo_type::Constants cst;
            e.initConstants(cst);

            uint32 j = 0;
            while (j + (D*M) <= n) {
                Details::Body<M>::template iteration<D>(e, ri + j, zi + j, cst);
                j += (D*M);
            }
            while (j + D <= n) {
                e.vectorial(ri + j, zi + j, cst);
                j += D;
            }
            while (D > 1 && j < n) {
                ri[j] = e.scalar(zi[j]);
                j += 1;
            }
        }
    };

    template <uint32 nIterTot, uint32 nIterLeft>
    struct _Pipeliner
    {
        template <typename Expr, typename Data>
        FORCE_INLINE static void go(const Expr& e, Data* d)
        {
            e.template run<nIterTot - nIterLeft>(d);
            _Pipeliner<nIterTot, nIterLeft - 1>::go(e, d);
        }
    };

    template <uint32 nIterTot>
    struct _Pipeliner<nIterTot, 0>
    {
        template <typename Expr, typename Data>
        FORCE_INLINE static void go(const Expr& e, Data* d)
        {
        }
    };

    template <uint32 nIter>
    struct Pipeliner
    {
        template <typename Expr, typename Data>
        FORCE_INLINE static void go(const Expr& e, Data* d)
        {
            _Pipeliner<nIter, nIter>::go(e, d);
        }
    };


#if 1
    template <class T>
    char is_complete_impl(char (*)[sizeof(T)]);

    template <class>
    long is_complete_impl(...);

    template <class T>
    struct IsComplete
    {
        static const bool value = sizeof(is_complete_impl<T>(0)) == sizeof(char);
    };
#else
    template <class T, std::size_t = sizeof(T)>
    std::true_type is_complete_impl(T *);

    std::false_type is_complete_impl(...);

    template <class T>
    struct IsComplete : decltype(is_complete_impl(std::declval<T*>())) {};
#endif

template <typename T, Algos A>
struct AlgoScalarToVec : AlgoScalarBase<T,A>
{
    typedef AlgoScalarBase<T, A> base_t;

    AlgoScalarToVec(const typename base_t::Data& d) :  base_t(d) {}
    AlgoScalarToVec(const T* px, const uint32 n) :  base_t(px, n) {}

    static const uint32 nElem = 1;

    struct Constants
    {
    };

    void initConstants(Constants& cst) const
    {
    }

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, const Constants& cst) const
    {
        *pr = base_t::scalar(*pz);
    }
};

template<bool B, class T, class F>
struct conditional { typedef T type; };

template<class T, class F>
struct conditional<false, T, F> { typedef F type; };

template <typename T, bool C>
struct CondData
{
    FORCE_INLINE CondData(T x) : v(x) {}
    FORCE_INLINE operator const T&() const { return v;}
private:
    T v;
};

template <typename T>
struct CondData<T,false>
{
    FORCE_INLINE CondData(T) {}
    FORCE_INLINE operator const T() const { return 0;}
};

template <InstrSet I, typename T, Algos A, bool L=false>
struct BinAlgoBase : Details::conditional< Details::IsComplete<Details::AlgoVecBase<I, T, A>>::value
                                 , Details::AlgoVecBase<I, T, A>
                                 , Details::AlgoScalarToVec<T,A>
                                 >::type
{
    typedef typename Details::conditional< Details::IsComplete<Details::AlgoVecBase<I, T, A>>::value
                                 , Details::AlgoVecBase<I, T, A>
                                 , Details::AlgoScalarToVec<T,A>
                                 >::type base_t;

    BinAlgoBase(const T* px, const uint32 n) :  base_t(px, n) {}
    BinAlgoBase(const typename base_t::Data& d) : base_t(d) {}
};

} // namespace Details

} // namespace BinSearch
