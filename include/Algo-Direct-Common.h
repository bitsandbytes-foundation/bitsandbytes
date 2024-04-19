#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>
#include "AAlloc.h"

namespace BinSearch {
namespace Details {

namespace DirectAux {

#define SAFETY_MULTI_PASS true

template <typename T>
struct HResults
{
    HResults(T h, double ratio, size_t n) : H(h), hRatio(ratio), nInc(n) {}
    T H;
    double hRatio;
    size_t nInc;
};


#ifdef USE_FMA
template <Algos A> struct IsDirect { static const bool value = (A == Direct) || (A == DirectFMA); };
template <Algos A> struct IsDirect2 { static const bool value = (A == Direct2) || (A == Direct2FMA); };
template <Algos A> struct IsDirectCache { static const bool value = (A == DirectCache) || (A == DirectCacheFMA); };
#else
template <Algos A> struct IsDirect { static const bool value = (A == Direct); };
template <Algos A> struct IsDirect2 { static const bool value = (A == Direct2); };
template <Algos A> struct IsDirectCache { static const bool value = (A == DirectCache); };
#endif

// general definition
template <Algos A, typename T, typename Enable = void>
struct BucketElem
{
    FORCE_INLINE void set( uint32 b, const T *)
    {
        m_b = b;
    }

    FORCE_INLINE uint32 index() const { return m_b; }

private:
    uint32 m_b;
};

// specialization for DirectCache methods

template <typename T> struct MatchingIntType;
template <> struct MatchingIntType<double> { typedef uint64 type; };
template <> struct MatchingIntType<float> { typedef uint32 type; };

template <Algos A, typename T>
struct BucketElem<A, T, typename std::enable_if< IsDirectCache<A>::value >::type >
{
    typedef typename MatchingIntType<T>::type I;

    void set(uint32 b, const T *xi)
    {
        u.u.x = xi[b];
        u.u.b = b;
    }

    FORCE_INLINE I index() const { return u.u.b; }
    FORCE_INLINE T x() const { return u.u.x; }

private:
    union {
        double dummy;
        struct
        {
            T x;
            I b;
        } u;
    } u;
};


template <bool UseFMA, unsigned char Gap, typename T>
struct DirectTraits
{
    static void checkH(T scaler, T x0, T xN)
    {
        T Dn = xN - x0;
        T ifmax = Dn * scaler;
        myassert((ifmax < std::numeric_limits<uint32>::max() - (Gap - 1)),
            "Problem unfeasible: index size exceeds uint32 capacity:"
            << " D[N] =" << Dn
            << ", H =" << scaler
            << ", H D[n] =" << ifmax << "\n"
        );
    }

    FORCE_INLINE static uint32 f(T scaler, T x0, T z)
    {
        T tmp = scaler * (z - x0);
#ifdef USE_SSE2
        return ftoi(FVec1<SSE,T>(tmp));
#else
        return static_cast<uint32>(tmp);
#endif
    }

    template <InstrSet I>
    FORCE_INLINE static typename FTOITraits<I, T>::vec_t f(const FVec<I, T>& scaler, const FVec<I, T>& x0, const FVec<I, T>& z)
    {
        return ftoi(scaler*(z-x0));
    }

    static T cst0(T scaler, T x0)
    {
        return x0;
    }
};

#ifdef USE_FMA
template <unsigned char Gap, typename T>
struct DirectTraits<true,Gap,T>
{
    typedef FVec1<SSE, T> fVec1;

    static void checkH(T scaler, T H_Times_x0, T xN)
    {
        union {
            typename FVec1<SSE, T>::vec_t v;
            T s;
        } ifmax;
        ifmax.v = mulSub(fVec1(scaler), fVec1(xN), fVec1(H_Times_x0));
        myassert((ifmax.s < std::numeric_limits<uint32>::max() - (Gap - 1)),
            "Problem unfeasible: index size exceeds uint32 capacity:"
            << " H X[0] =" << H_Times_x0
            << ", H =" << scaler
            << ", X[N] =" << xN
            << ", H X[N] - H X[0] =" << ifmax.s << "\n"
        );
    }

    FORCE_INLINE static uint32 f(T scaler, T Hx0, T xi)
    {
        return ftoi(mulSub(fVec1(scaler), fVec1(xi), fVec1(Hx0)));
    }

    template <InstrSet I>
    FORCE_INLINE static typename FTOITraits<I,T>::vec_t f(const FVec<I,T>& scaler, const FVec<I, T>& H_Times_X0, const FVec<I, T>& z)
    {
        return ftoi(mulSub(scaler, z, H_Times_X0));
    }

    static T cst0(T scaler, T x0)
    {
        return scaler*x0;
    }
};
#endif

template <unsigned char Gap, typename T, Algos A>
struct DirectInfo
{
    static const bool UseFMA = (A == DirectFMA) || (A == Direct2FMA) || (A == DirectCacheFMA);
    typedef DirectTraits<UseFMA, Gap, T> fun_t;
    typedef BucketElem<A,T> bucket_t;
    typedef AlignedVec<bucket_t> bucketvec_t;

    struct Data {
        Data() : buckets(0), xi(0), scaler(0), cst0(0) {}
        Data( const T *x      // for Direct must persist if xws=NULL
            , uint32 n
            , T H
            , bucket_t *bws   // assumed to gave size nb, as computed below
            , T *xws = NULL   // assumed to have size (n+Gap-1). Optional for Direct, unused for DirectCache, required for DirectGap
            )
            : buckets(bws)
            , scaler(H)
            , cst0(fun_t::cst0(H, x[0]))
        {
            myassert(((bws != NULL) && (isAligned(bws,64))), "bucket pointer not allocated or incorrectly aligned");

            uint32 nb = 1 + fun_t::f(H, cst0, x[n-1]);

            const uint32 npad = Gap-1;
            const uint32 n_sz = n + npad;   // size of padded vector

            if (xws) {
                myassert(isAligned(xws,8), "x pointer not allocated or incorrectly aligned");
                std::fill_n(xws, npad, x[0]);    // pad in front with x[0]
                std::copy(x, x+n, xws + npad);
                xi = xws;
            }
            else {
                myassert((Gap==1), "if Gap>1 then X workspace must be provided");
                xi = x;
            }

            populateIndex(bws, nb, xi, n_sz, scaler, cst0);
        }

        const bucket_t *buckets;
        const T *xi;
        T scaler;
        T cst0;  // could be x0 or (scaler*x0), depending if we are using FMA or not
    } data;

    static T growStep(T H)
    {
        T step;
        T P = next(H);
        while ((step = P - H) == 0)
            P = next(P);
        return step;
    }

    static HResults<T> computeH(const T *px, uint32 nx)
    {
        myassert((nx > Gap), "Array X too small");
        myassert(((Gap == 1) || (Gap == 2)), "Only tested for these values of Gap");

        const T x0 = px[0];
        const T xN = px[nx-1];

        const T range = xN - x0;
        myassert((range < std::numeric_limits<T>::max()), "range too large");

        // check that D_i are strictly increasing and compute minimum value D_{i+Offset}-D_i
        T deltaDMin = range;
        for (uint32 i = Gap; i < nx; ++i) {
            T Dnew = px[i] - x0;
            T Dold = px[i - Gap] - x0;
            myassert((Dnew > Dold),
                "Problem unfeasible: D_i sequence not strictly increasing"
                << " X[" << 0 << "]=" << x0
                << " X[" << i - Gap << "]=" << px[i - Gap]
                << " X[" << i << "]=" << px[i]
                << "\n"
            );
            T deltaD = Dnew - Dold;
            if (deltaD < deltaDMin)
                deltaDMin = deltaD;
        }

        // initial guess for H
        const T H0 = T(1.0) / deltaDMin;
        T H = H0;

        T cst0 = fun_t::cst0(H, x0);
        fun_t::checkH(H, cst0, xN);

        // adjust H by trial and error until succeed
        size_t nInc = 0;
        bool modified = false;
        size_t npasses = 0;
        T step = growStep(H);
        uint32 seg_already_checked_from = nx;
        do {
            myassert((npasses++ < 2), "verification failed\n");
            // if there has been an increase, then check only up to that point
            uint32 last_seg_to_be_checked = seg_already_checked_from - 1;
            modified = false;
            uint32 inew = 0;
            for (uint32 i = Gap; i <= last_seg_to_be_checked; ++i) {
                uint32 iold = fun_t::f(H, cst0, px[i-Gap]);
                uint32 inew = fun_t::f(H, cst0, px[i]);
                while (inew == iold) {
                    seg_already_checked_from = i;
                    last_seg_to_be_checked = nx-1;  // everything needs to be checked
                    modified = true;
                    H = H + step;
                    step *= 2;
                    // recalculate all constants and indices
                    cst0 = fun_t::cst0(H, x0);
                    fun_t::checkH(H, cst0, xN);
                    iold = fun_t::f(H, cst0, px[i - Gap]);
                    inew = fun_t::f(H, cst0, px[i]);
                }
            }
        } while (SAFETY_MULTI_PASS && modified);

        return HResults<T>(H, (((double)H) / H0) - 1.0, nInc);
    }

    static void populateIndex(BucketElem<A, T> *buckets, uint32 index_size, const T *px, uint32 x_size, T scaler, T cst0)
    {
        for (uint32 i = x_size-1, b = index_size-1, j=0; ; --i) {
            uint32 idx = fun_t::f(scaler, cst0, px[i]);
            while (b > idx) {  // in the 1st iteration it is j=0 but this condition is always false
                buckets[b].set( j, px );
                --b;
            }
            if (Gap==1 || b == idx) { // if Gap==1, which is known at compile time, the check b==idx is redundant
                j = i - (Gap-1); // subtracting (Gap-1) points to the index of the first X-element to check
                buckets[b].set(j, px);
                if (b-- == 0)
                    break;
            }
        }
    }

    DirectInfo(const Data& d)
        : data(d)
    {
    }

    DirectInfo(const T* px, const uint32 n)
    {
        HResults<T> res = computeH(px, n);

#ifdef PAPER_TEST
        nInc = res.nInc;
        hRatio = res.hRatio;
#endif
        const uint32 npad = Gap-1;
        const uint32 n_sz = n + npad;   // size of padded vector

        if (npad)
            xi.resize(n_sz);

        T H    = res.H;
        T cst0 = fun_t::cst0(H, px[0]);
        const uint32 maxIndex = fun_t::f(H, cst0, px[n-1]);
        buckets.resize(maxIndex + 1);

        data = Data(px, n, H, buckets.begin(), (npad? xi.begin(): NULL));
    }

private:
    bucketvec_t buckets;
    AlignedVec<T,8> xi;

#ifdef PAPER_TEST
public:
    double hRatio;
    size_t nInc;
#endif
};


} // namespace DirectAux
} // namespace Details
} // namespace BinSearch
