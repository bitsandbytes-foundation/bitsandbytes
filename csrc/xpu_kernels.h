#include <float.h>
#include <xpu_ops.h>

#ifndef xpu_kernels
#define xpu_kernels

template <typename T, int TILE_SIZE, int NUM_PER_TH, int DATA_TYPE> class kDequantizeBlockwise {
  public:
    SYCL_EXTERNAL void operator()(sycl::nd_item<1> item) const;

    kDequantizeBlockwise(float* code_, uint8_t* A_, float* absmax_, T* out_, const int blocksize_, const int n_)
        : code(code_), A(A_), absmax(absmax_), out(out_), blocksize(blocksize_), n(n_) {}

  private:
    float* code;
    uint8_t* A;
    float* absmax;
    T* out;
    const int blocksize;
    const int n;
};

template <typename T, size_t GROUP_SIZE, size_t NUM_PER_THREAD, size_t SUBG_SIZE, int BITS> class kgemv_4bit_inference {
  public:
    SYCL_EXTERNAL void operator()(sycl::nd_item<1> item) const;

    kgemv_4bit_inference(
        int M_, int N_, int K_, T* A_, unsigned char* B_, float* absmax_, const float* datatype_, T* out_, int lda_,
        int ldb_, int ldc_, int blocksize_
    )
        : M(M_), N(N_), K(K_), A(A_), B(B_), absmax(absmax_), datatype(datatype_), out(out_), lda(lda_), ldb(ldb_),
          ldc(ldc_), blocksize(blocksize_), quant_map() {}

    void sycl_ker_local_memory_creation(sycl::handler& cgh) { quant_map = sycl::local_accessor<T>(16, cgh); }

  private:
    int M;
    int N;
    int K;
    T* A;
    unsigned char* B;
    float* absmax;
    const float* datatype;
    T* out;
    int lda;
    int ldb;
    int ldc;
    int blocksize;
    sycl::local_accessor<T> quant_map;
};

#endif
