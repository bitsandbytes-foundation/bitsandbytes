#pragma once

#include "Type.h"
#include <algorithm>

namespace BinSearch {

template <InstrSet I, typename T, Algos A, bool L=false, bool R=false>
struct BinAlgo : Details::BinAlgoBase<I,T,A>
{
    typedef Details::BinAlgoBase<I,T,A> base_t;

    BinAlgo(const T* px, const uint32 n) :  base_t(px, n), x0(px[0]), xN(px[n-1]), N(n) {}
    BinAlgo(const T* px, const uint32 n, const typename base_t::Data& d) : base_t(d), x0(px[0]), xN(px[n-1]), N(n) {}

    FORCE_INLINE
    uint32 scalar(T z) const
    {
        if (!L || z >= x0)
            if (!R || z < xN)
                return base_t::scalar(z);
            else
                return N;
        else
            return std::numeric_limits<uint32>::max();
    }


    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, uint32 n) const
    {
        if (!L && !R) {
            Details::Loop<T,base_t>::loop(*this, pr, pz, n);
        }
        else {
            const uint32 nElem = base_t::nElem;
            const uint32 idealbufsize = 256;
            const uint32 bufsize = nElem * (idealbufsize / nElem + ((idealbufsize % nElem) ? 1 : 0));
            T databuf[bufsize];
            uint32 resbuf[bufsize];
            uint32 indexbuf[bufsize];

            uint32 *prend = pr + n;
            while(pr != prend) {
                uint32 cnt = 0;
                uint32 niter = std::min(bufsize, (uint32)std::distance(pr,prend));
                for (uint32 j = 0; j < niter; ++j) {
                    T z = pz[j];
                    // FIXME: use SSE2?
                    if (!L || z >= x0)
                        if (!R || z < xN) {
                            databuf[cnt] = z;
                            indexbuf[cnt] = j;
                            ++cnt;
                        }
                        else
                            pr[j] = N;
                    else
                        pr[j] = std::numeric_limits<uint32>::max();
                }
                // FIXME: merge these two loops
                Details::Loop<T,base_t>::loop(*this, resbuf, databuf, cnt);
                for (uint32 j = 0; j < cnt; ++j)
                    pr[indexbuf[j]] = resbuf[j];
                pr += niter;
                pz += niter;
            }
        }
    }

    Details::CondData<T,L> x0;
    Details::CondData<T,R> xN;
    Details::CondData<uint32,R> N;
};


} // namespace BinSearch
