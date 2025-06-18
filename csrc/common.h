#include <BinSearch.h>

#ifndef common
#define common

using namespace BinSearch;

struct quantize_block_args {
    BinAlgo<Scalar, float, Direct2>* bin_searcher;
    float* code;
    float* A;
    float* absmax;
    unsigned char* out;
    long long block_end;
    long long block_idx;
    long long threadidx;
    long long blocksize;
};

void quantize_block(const quantize_block_args& args);

#endif
