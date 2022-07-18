#include <BinSearch.h>

#ifndef common
#define common

using namespace BinSearch;

struct quantize_block_args {
    BinAlgo<Scalar, float, Direct2> *bin_searcher;
    float *code;
    float *A;
    float *absmax;
    unsigned char *out;
    int block_end;
    int block_idx;
    int threadidx;
};

#define BLOCK_SIZE 4096

void *quantize_block(void *arguments);

#endif
