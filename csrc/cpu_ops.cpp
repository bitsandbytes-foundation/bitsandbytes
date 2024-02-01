#include <BinSearch.h>
#ifdef _WIN32
#include <thread>
#else
#include <pthread.h>
#endif
#include <common.h>

using namespace BinSearch;

void dequantize_cpu(float *code, unsigned char *A, float *absmax, float *out, long long blocksize, long long n) {
    for (long long block_idx = 0; block_idx < n; block_idx += blocksize) {
        long long valid_items = n - block_idx >= blocksize ? blocksize : n - block_idx;
        long long block_end = block_idx + valid_items;
        for (long long i = block_idx; i < block_end; i++)
            out[i] = code[A[i]] * absmax[block_idx / blocksize];
    }
}

void quantize_cpu(float *code, float *A, float *absmax, unsigned char *out, long long blocksize, long long n)
{

    // the default code is has range [-0.993, 1.0] which can cause an error in the binary search algorithm used below
    code[0] = -1.0f;

    long long num_blocks = n / blocksize;
    num_blocks += n % blocksize == 0 ? 0 : 1;

    const uint32 elements_code = 256;
    BinAlgo<Scalar, float, Direct2> bin_searcher(code, elements_code);

    int thread_wave_size = 256;
    // we chunk the thresds into waves of 256 since the max limit is
    // between 16k and 64k on Linux (we reach this when running BLOOM-176B with a large batch size)
    for(long long offset = 0; offset < num_blocks; offset+=thread_wave_size)
    {
      long long valid_chunks = num_blocks - offset >= thread_wave_size ? thread_wave_size : num_blocks - offset;
#ifdef _WIN32
      std::thread *threads = (std::thread *) malloc(sizeof(std::thread) * valid_chunks);
#else
      pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * valid_chunks);
#endif

      struct quantize_block_args **args = (quantize_block_args **) malloc(valid_chunks * sizeof(quantize_block_args *));

      for(long long i = 0; i < valid_chunks; i++)
          args[i] = (quantize_block_args *) malloc(sizeof(quantize_block_args));

      int chunks_processed = 0;
      for(long long block_idx = offset*blocksize; block_idx < n; block_idx += blocksize)
      {
          long long valid_items = n - block_idx >= blocksize ? blocksize : n - block_idx;
          long long block_end = block_idx + valid_items;

          struct quantize_block_args *arg = args[chunks_processed];
          arg->bin_searcher = &bin_searcher;
          arg->code = code;
          arg->A = A;
          arg->absmax = absmax;
          arg->out = out;
          arg->block_end = block_end;
          arg->block_idx = block_idx;
          arg->threadidx = block_idx / blocksize;
          arg->blocksize = blocksize;

#ifdef _WIN32
          new (&threads[chunks_processed]) std::thread(quantize_block, arg);
#else
          pthread_create(&threads[chunks_processed], NULL, &quantize_block, (void *) arg);
#endif
          chunks_processed += 1;
          if(chunks_processed == valid_chunks){ break; }
      }

      for (int i = 0; i < valid_chunks; i++)
      {
#ifdef _WIN32
          threads[i].join();
#else
          int err = pthread_join(threads[i], NULL);
#endif
      }
      free(threads);
      for (int i = 0; i < valid_chunks; i++)
          free(args[i]);
      free(args);

    }

}
