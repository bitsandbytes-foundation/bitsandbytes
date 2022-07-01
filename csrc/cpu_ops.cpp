#include <BinSearch.h>
#include <pthread.h>
#include <common.h>

using namespace BinSearch;

void dequantize_cpu(float *code, unsigned char *A, float *absmax, float *out, int n) {
    for (int block_idx = 0; block_idx < n; block_idx += BLOCK_SIZE) {
        int valid_items = n - block_idx >= BLOCK_SIZE ? BLOCK_SIZE : n - block_idx;
        int block_end = block_idx + valid_items;
        for (int i = block_idx; i < block_end; i++)
            out[i] = code[A[i]] * absmax[block_idx / BLOCK_SIZE];
    }
}

void quantize_cpu(float *code, float *A, float *absmax, unsigned char *out, int n) {

    // the default code is has range [-0.993, 1.0] which can cause an error in the binary search algorithm used below
    code[0] = -1.0f;

    int num_blocks = n / BLOCK_SIZE;
    num_blocks += n % BLOCK_SIZE == 0 ? 0 : 1;

    pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * num_blocks);
    struct quantize_block_args **args = (quantize_block_args **) malloc(num_blocks * sizeof(quantize_block_args *));

    for (int i = 0; i < num_blocks; i++)
        args[i] = (quantize_block_args *) malloc(sizeof(quantize_block_args));

    const uint32 elements_code = 256;
    BinAlgo<Scalar, float, Direct2> bin_searcher(code, elements_code);

    for (int block_idx = 0; block_idx < n; block_idx += BLOCK_SIZE) {
        int valid_items = n - block_idx >= BLOCK_SIZE ? BLOCK_SIZE : n - block_idx;
        int block_end = block_idx + valid_items;

        struct quantize_block_args *arg = args[block_idx / BLOCK_SIZE];
        arg->bin_searcher = &bin_searcher;
        arg->code = code;
        arg->A = A;
        arg->absmax = absmax;
        arg->out = out;
        arg->block_end = block_end;
        arg->block_idx = block_idx;
        arg->threadidx = block_idx / BLOCK_SIZE;

        pthread_create(&threads[block_idx / BLOCK_SIZE], NULL, &quantize_block, (void *) arg);
    }

    for (int i = 0; i < num_blocks; i++)
        int err = pthread_join(threads[i], NULL);

    free(threads);
    for (int i = 0; i < num_blocks; i++)
        free(args[i]);
    free(args);
}
