#ifndef xpu_ops_H
#define xpu_ops_H

#include <assert.h>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#include <functional>
#include <vector>

#include <sycl/sycl.hpp>

template <typename ker_t, int dim, int subgroup_size>
static inline void sycl_kernel_submit(sycl::nd_range<dim> range, sycl::queue q,
                                      ker_t ker) {
  auto cgf = [&](::sycl::handler & cgh)
      [[sycl::reqd_sub_group_size(subgroup_size)]] {
    cgh.parallel_for<ker_t>(range, ker);
  };
  q.submit(cgf);
}

template <typename ker_t, int dim, int subgroup_size>
static inline void sycl_comp_kernel_submit(sycl::nd_range<dim> range,
                                           sycl::queue q, ker_t ker) {
  auto cgf = [&](::sycl::handler & cgh)
      [[sycl::reqd_sub_group_size(subgroup_size)]] {
    ker.sycl_ker_local_memory_creation(cgh);
    cgh.parallel_for<ker_t>(range, ker);
  };
  q.submit(cgf);
}

typedef enum DataType_t {
  General8bit = 0,
  FP4 = 1,
  NF4 = 2,
} DataType_t;

template <typename T, int DATA_TYPE>
void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out,
                         int workgroup_size, const int n, sycl::queue *stream);
template <typename T, int BITS>
void gemm_4bit_inference(int m, int n, int k, T *A, unsigned char *B,
                         float *absmax, float *datatype, T *out, int lda,
                         int ldb, int ldc, int blocksize, sycl::queue *stream);

#endif
