#include <BinSearch.h>

#ifndef common
#define common

using namespace BinSearch;

#define BLOCK_SIZE 16384

#if defined(USE_AVX) || defined(USE_AVX2)
#define INSTR_SET AVX
#elif defined(USE_SSE41) || defined(USE_SSE42)
#define INSTR_SET SSE
#else
#define INSTR_SET Scalar
#endif

struct quantize_block_args {
    BinAlgo<INSTR_SET, float, Direct2> *bin_searcher;
    float *code;
    float *A;
    float *absmax;
    unsigned char *out;
    long long block_end;
    long long block_idx;
    long long threadidx;
		long long blocksize;
};


void *quantize_block(void *arguments);

#endif
