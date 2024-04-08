// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_extras/dpcpp_extensions.h>
#include "ops.h"
#include "kernels.h"
#include <limits>
#include <BinSearch.h>
#include <cassert>
#include <common.h>
#include <dpct/lib_common_utils.hpp>

#include "oneapi/dnnl/dnnl.hpp"

#define ERR_NOT_IMPLEMENTED 100

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

#define THREADS_ESTIMATE 512
#define NUM_ESTIMATE 8
#define BLOCK_ESTIMATE 4096


using namespace BinSearch;
using std::cout;
using std::endl;

void histogramScatterAdd2D(float* histogram, int *index1, int *index2, float *src, int maxidx1, int n)
{
  int threads = 512;
  int num_blocks = n/threads;
  num_blocks = n % threads == 0 ? num_blocks : num_blocks + 1;
  /*
  DPCT1049:53: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
    [=](sycl::nd_item<3> item_ct1) {
      kHistogramScatterAdd2D(histogram, index1, index2, src, maxidx1, n, item_ct1);
    });
  /*
  DPCT1010:229: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);
}

template <typename T> void estimateQuantiles(T *A, float *code, float offset, int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(code, 0, 256*sizeof(float)).wait()));
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  *((T **)&buff_A) = sycl::malloc_device(size, A, ctx);
  
  q_ct1.memcpy((T*)(buff_A), (T*)(A), NUM_BLOCK);
  //sycl::buffer<T, 1> buff_A(A,sycl::range<1>(num_blocks));
  /*
  DPCT1049:54: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      
        using group_load = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
        using group_radix_sort = dpct::group::radix_sort<int, NUM_BLOCK>;
        size_t sort_temp_storage_size = group_radix_sort::get_local_memory_size(NUM_BLOCK);  
        sycl::local_accessor<uint8_t, 1> tacc(temp_storage_size, cgh);
          
        /*
        DPCT1054:293: The type of variable temp_storage is declared in device function with the name type_ct1. Adjust the code to make the type_ct1 declaration visible at the accessor declaration point.
        */
        //sycl::local_accessor<uint8_t[sizeof(type_ct1)], 0> temp_storage_ct1_acc_ct1(cgh);

        auto std_numeric_limits_T_max_ct3 = std::numeric_limits<T>::max();

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
          [=](sycl::nd_item<3> item_ct1) {
            kEstimateQuantiles<T>(buff_A, code, offset, std_numeric_limits_T_max_ct3, n, item_ct1, tacc);
          });
      });
  }
  //back memcpy
  q_ct1.memcpy((T*)(A), (T*)(buff_A), NUM_BLOCK); 
  
}

void quantize(float *code, float *A, unsigned char *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  *((float **)&buff_A) = sycl::malloc_device(size, A, ctx);
  *((unsigned char **)&buff_out = sycl::malloc_device(size, out, ctx);
  q_ct1.memcpy((float*)(buff_A), (float*)(A), NUM_BLOCK);
  q_ct1.memcpy((unsigned char*)(buff_out), (unsigned char*)(out), NUM_BLOCK);

  /*
  DPCT1049:55: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      using group_load = dpct::group::workgroup_load<NUM_BLOCK,BLOCK_LOAD_DIRECT,float>;
      size_t load_temp_storage_size = group_load::get_local_memory_size(NUM_BLOCK);
      using group_store = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT,unsigned char>;
      size_t store_temp_storage_size = group_store::get_local_memory_size(NUM_BLOCK);
      
      sycl::local_accessor<uint8_t, 1> ltacc(load_temp_storage_size, cgh);
      sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
      
      //__shared__ vars
      sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
      
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
        [=](sycl::nd_item<3> item_ct1) {
          kQuantize(code, buff_A, buff_out, n, item_ct1, smem_code_acc_ct1.get_pointer(), ltacc, stacc);
        });
    });
  }
  //back memcpy
  q_ct1.memcpy((float*)(A), (float*)(buff_A), NUM_BLOCK);
  q_ct1.memcpy((unsigned char*)(out), (unsigned char*)(buff_out), NUM_BLOCK);
  /*
  DPCT1010:232: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);
}

void dequantize(float *code, unsigned char *A, float *out, int n)
{
  int num_blocks = n/1024;
  num_blocks = n % 1024 == 0 ? num_blocks : num_blocks + 1;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  *((unsigned char **)&buff_A) = sycl::malloc_device(size, A, ctx);
  *((float **)&buff_out = sycl::malloc_device(size, out, ctx);
  q_ct1.memcpy((float*)(buff_out), (float*)(out), NUM_BLOCK);
  q_ct1.memcpy((unsigned char*)(buff_A), (unsigned char*)(A), NUM_BLOCK);

  /*
  DPCT1049:56: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
      
      //__shared__ vars
      sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
      
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
        [=](sycl::nd_item<3> item_ct1) {
          kDequantize(code, buff_A, buff_out, n, item_ct1, smem_code_acc_ct1.get_pointer());
        });
    });
   //q_ct1.wait();
   
   } 
  //back memcpy
  q_ct1.memcpy((float*)(out), (float*)(buff_out), NUM_BLOCK);
  q_ct1.memcpy((unsigned char*)(A), (unsigned char*)(buff_A), NUM_BLOCK); 
  /*
  DPCT1010:233: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);
}

template <typename T, int STOCHASTIC, int DATA_TYPE> void quantizeBlockwise(float * code, T *A, float *absmax, unsigned char *out, float *rand, int rand_offset, int blocksize, const int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  sycl::context ctx = q_ct1.get_context();
  int size= NUM_BLOCK;
  
  *((T **)&buff_A) = sycl::malloc_device(size, A, ctx);
  *((unsigned char **)&buff_out = sycl::malloc_device(size, out, ctx);
  *((float **)&buff_rand = sycl::malloc_device(size, rand, ctx);
  q_ct1.memcpy((T*)(buff_A), (T*)(A), NUM_BLOCK);
  q_ct1.memcpy((unsigned char*)(buff_out), (unsigned char*)(out), NUM_BLOCK);
  q_ct1.memcpy((float*)(buff_rand), (float*)(rand), NUM_BLOCK);
  
  for(int i=0; i< NUM_BLOCK; i++){ buff_out[i]=buff_out[(DATA_TYPE > 0) ? i/2 : i]};
  
  if(blocksize == 4096)
    /*
    DPCT1049:57: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
          size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
          using group_store = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT,unsigned char>;
          size_t store_temp_storage_size = group_store::get_local_memory_size(NUM_BLOCK);
          using group_load_float = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
          size_t load_temp_storage_size_float = group_load_float::get_local_memory_size(NUM_BLOCK);
          
          sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
          sycl::local_accessor<uint8_t, 1> ltacc_float(load_temp_storage_size_float, cgh);
          sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
          
          //__shared__ vars for funtions
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 4096, 4, STOCHASTIC, 0>(code, buff_A, absmax, buff_out, buff_rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), ltacc_T, ltacc_float, stacc);
            });
        });
    }
  else if(blocksize == 2048)
    /*
    DPCT1049:58: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          
          using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
          size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
          using group_store = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT,unsigned char>;
          size_t store_temp_storage_size = group_store::get_local_memory_size(NUM_BLOCK);
          using group_load_float = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
          size_t load_temp_storage_size_float = group_load_float::get_local_memory_size(NUM_BLOCK);
          
          sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
          sycl::local_accessor<uint8_t, 1> ltacc_float(load_temp_storage_size_float, cgh);
          sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
      
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 2048, 4, 0, DATA_TYPE>(code, buff_A, absmax, buff_out, buff_rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), ltacc_T, ltacc_float, stacc);
            });
        });
    }
  else if(blocksize == 1024)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
          size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
          using group_store = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT,unsigned char>;
          size_t store_temp_storage_size = group_store::get_local_memory_size(NUM_BLOCK);
          using group_load_float = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
          size_t load_temp_storage_size_float = group_load_float::get_local_memory_size(NUM_BLOCK);
          
          sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
          sycl::local_accessor<uint8_t, 1> ltacc_float(load_temp_storage_size_float, cgh);
          sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
      
          //__shared__vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 1024, 4, 0, DATA_TYPE>(code, buff_A, absmax, buff_out, buff_rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), ltacc_T, ltacc_float, stacc);
            });
        });
    }
  else if(blocksize == 512)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
          size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
          using group_store = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT,unsigned char>;
          size_t store_temp_storage_size = group_store::get_local_memory_size(NUM_BLOCK);
          using group_load_float = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
          size_t load_temp_storage_size_float = group_load_float::get_local_memory_size(NUM_BLOCK);
          
          sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
          sycl::local_accessor<uint8_t, 1> ltacc_float(load_temp_storage_size_float, cgh);
          sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
      
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 512, 2, 0, DATA_TYPE>(code, buff_A, absmax, buff_out, buff_rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), ltacc_T, ltacc_float, stacc);
            });
        });
    }
  else if(blocksize == 256)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
          size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
          using group_store = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT,unsigned char>;
          size_t store_temp_storage_size = group_store::get_local_memory_size(NUM_BLOCK);
          using group_load_float = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
          size_t load_temp_storage_size_float = group_load_float::get_local_memory_size(NUM_BLOCK);
          
          sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
          sycl::local_accessor<uint8_t, 1> ltacc_float(load_temp_storage_size_float, cgh);
          sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
      
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);

          
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 256, 2, 0, DATA_TYPE>(code, buff_A, absmax, buff_out, buff_rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), ltacc_T, ltacc_float, stacc);
            });
        });
    }
  else if(blocksize == 128)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
          size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
          using group_store = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT,unsigned char>;
          size_t store_temp_storage_size = group_store::get_local_memory_size(NUM_BLOCK);
          using group_load_float = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
          size_t load_temp_storage_size_float = group_load_float::get_local_memory_size(NUM_BLOCK);
          
          sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
          sycl::local_accessor<uint8_t, 1> ltacc_float(load_temp_storage_size_float, cgh);
          sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
      
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 128, 2, 0, DATA_TYPE>(code, buff_A, absmax, buff_out, buff_rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), ltacc_T, ltacc_float, stacc);
            });
        });
    }
  else if(blocksize == 64)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
        [&](sycl::handler &cgh) {
          
          using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
          size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
          using group_store = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT,unsigned char>;
          size_t store_temp_storage_size = group_store::get_local_memory_size(NUM_BLOCK);
          using group_load_float = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
          size_t load_temp_storage_size_float = group_load_float::get_local_memory_size(NUM_BLOCK);
          
          sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
          sycl::local_accessor<uint8_t, 1> ltacc_float(load_temp_storage_size_float, cgh);
          sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
          
          //__shared__ vars
          sycl::local_accessor<float, 1> smem_code_acc_ct1(sycl::range<1>(256), cgh);
          sycl::local_accessor<float, 1> smem_absmax_value_acc_ct1(sycl::range<1>(1), cgh);
          
      
          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), 
            [=](sycl::nd_item<3> item_ct1) {
              kQuantizeBlockwise<T, 64, 2, 0, DATA_TYPE>(code, buff_A, absmax, buff_out, buff_rand, rand_offset, n, item_ct1,  smem_code_acc_ct1.get_pointer(), smem_absmax_value_acc_ct1.get_pointer(), ltacc_T, ltacc_float, stacc);
            });
        });
    }


  /*
  DPCT1010:234: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //back memcpy
  q_ct1.memcpy((T*)(A), (T*)(buff_A), NUM_BLOCK);
  q_ct1.memcpy((unsigned char*)(out), (unsigned char*)(buff_out), NUM_BLOCK);
  q_ct1.memcpy((float*)(rand), (float*)(buff_rand), NUM_BLOCK);
  //CUDA_CHECK_RETURN(0);
}

template<typename T, int DATA_TYPE> void dequantizeBlockwise(float *code, unsigned char *A, float *absmax, T *out, int blocksize, const int n)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/blocksize;
  num_blocks = n % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (DATA_TYPE > 0) ? 1024 : 512;
  sycl::context ctx = q_ct1.get_context();
  
  *((unsigned char **)&buff_A) = sycl::malloc_device(tile_size, A, ctx);
  *((T **)&buff_out = sycl::malloc_device(tile_size, out, ctx);
  q_ct1.memcpy((unsigned char*)(buff_A), (unsigned char*)(A), tile_size);
  q_ct1.memcpy((T*)(buff_out), (T*)(out), tile_size);
  
  if(DATA_TYPE > 0)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
      [&](sycl::handler &cgh){
      
        using group_load = dpct::group::workgroup_load<tile_size,BLOCK_LOAD_DIRECT,unsigned char>;
        using group_store = dpct::group::workgroup_store<tile_size, BLOCK_STORE_DIRECT,T>;
        size_t store_temp_storage_size = group_store::get_local_memory_size(tile_size);
        size_t load_temp_storage_size = group_load::get_local_memory_size(tile_size);
        sycl::local_accessor<uint8_t, 1> ltacc(load_temp_storage_size, cgh);
        sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
        
        
        
      q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (n+tile_size-1)/tile_size) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)), 
        [=](sycl::nd_item<3> item_ct1) {
          kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>(code, buff_A, absmax, buff_out, blocksize/2, n, item_ct1, ltacc, stacc);
        });
      });
    }
  else
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
      [&](sycl::handler &cgh){
      
        using group_load = dpct::group::workgroup_load<tile_size,BLOCK_LOAD_DIRECT,unsigned char>;
        using group_store = dpct::group::workgroup_store<tile_size, BLOCK_STORE_DIRECT,T>;
        size_t store_temp_storage_size = group_store::get_local_memory_size(tile_size);
        size_t load_temp_storage_size = group_load::get_local_memory_size(tile_size);
        sycl::local_accessor<uint8_t, 1> ltacc(load_temp_storage_size, cgh);
        sycl::local_accessor<uint8_t, 1> stacc(store_temp_storage_size, cgh);
        
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (n+tile_size-1)/tile_size) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)), 
        [=](sycl::nd_item<3> item_ct1) {
          kDequantizeBlockwise<T, 512, 64, 8, DATA_TYPE>(code, buff_A, absmax, buff_out, blocksize, n, item_ct1);
        });
      });
    }

  /*
  DPCT1010:235: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //back memcpy
  q_ct1.memcpy((unsigned char*)(A), (unsigned char*)(buff_A), tile_size);
  q_ct1.memcpy((T*)(out), (T*)(buff_out), tile_size);
  
  //CUDA_CHECK_RETURN(0);
}


//void matmul4bite(half *A, unsigned char *B, half*out, int lda, int ldb, int rowsA, int colsA, int colsB)
//{
//	int num_blocks = (colsB+32-1)/32;
//	kMatmul_inference_4bit<NF4, half, half, half><<<num_blocks, 256>>>(A, B, out, lda, ldb, rowsA, colsA, colsB);
//  CUDA_CHECK_RETURN(cudaPeekAtLastError());
//}


template<typename T, int OPTIMIZER> void optimizer32bit(T* g, T* p,
                float* state1, float* state2, float *unorm, float max_unorm, float param_norm,
                const float beta1, const float beta2, const float eps, const float weight_decay,
                const int step, const float lr, const float gnorm_scale, bool skip_zeros, const int n)
 try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  int size= NUM_BLOCK;  
   
  *((T **)&buff_g) = sycl::malloc_device(size, g, ctx);
  *((T **)&buff_p) = sycl::malloc_device(size, p, ctx);
  *((float **)&buff_state1 = sycl::malloc_device(size, state1, ctx);
  *((float **)&buff_state2 = sycl::malloc_device(size, state2, ctx);
  q_ct1.memcpy((T*)(buff_g), (T*)(g), size);
  q_ct1.memcpy((T*)(buff_p), (T*)(p), size);
  q_ct1.memcpy((float*)(buff_state1), (float*)(state1), size);
  q_ct1.memcpy((float*)(buff_state2), (float*)(state2), size);
  
  
	switch(OPTIMIZER)
	{
		case ADAM:
      if(max_unorm > 0.0f)
			{
				CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait()));
        /*
        DPCT1049:61: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              
              using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
              using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
              using group_load_float2 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
              size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
              size_t load_temp_storage_size_float2 = group_load_float2::get_local_memory_size(NUM_BLOCK);
              size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
              sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
              sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
              sycl::local_accessor<uint8_t, 1> ltacc_float2(load_temp_storage_size_float2, cgh);
              
              
              /*
              DPCT1054:294: The type of variable temp_storage is declared in device function with the name type_ct2. Adjust the code to make the type_ct2 declaration visible at the accessor declaration point.
              */
              cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
                [=](sycl::nd_item<3> item_ct1) {
                  kPreconditionOptimizer32bit2State<T, OPTIMIZER, 4096, 8>(g, p, buff_state1, buff_state2, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, ltacc_T, ltacc_float1, ltacc_float2);
                });
            });
        }
        /*
        DPCT1010:236: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
        */
        //CUDA_CHECK_RETURN(0);
      }
			/*
			DPCT1049:59: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
			*/
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      /*
			      DPCT1054:295: The type of variable temp_storage is declared in device function with the name type_ct3. Adjust the code to make the type_ct3 declaration visible at the accessor declaration point.
			      */
                    
            using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_T1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            using group_load_float2 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_float2 = group_load_float2::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T1 = group_load_T1::get_local_memory_size(NUM_BLOCK);

            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_T1(load_temp_storage_size_T1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float2(load_temp_storage_size_float2, cgh);
            
            
            using group_store_T = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, T>;
            using group_store_float1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            using group_store_float2 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            size_t store_temp_storage_size_float1 = group_store_float1::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_float2 = group_store_float2::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_T = group_store_T::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> stacc_T(store_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float1(store_temp_storage_size_float1, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float2(store_temp_storage_size_float2, cgh);
     
       
              
			      //sycl::local_accessor<uint8_t[sizeof(type_ct3)], 0> temp_storage_ct1_acc_ct1(cgh);

			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizer32bit2State<T, OPTIMIZER>(buff_g, buff_p, buff_state1, buff_state2, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, ltacc_T, ltacc_T1, ltacc_float1, ltacc_float2,stacc_T, stacc_float1, stacc_float2);
			        });
			    });
			}
      /*
      DPCT1010:237: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //CUDA_CHECK_RETURN(0);
			break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
      if(max_unorm > 0.0f)
			{
				CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait()));
				/*
				DPCT1049:62: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
				*/
				{
				  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
				  q_ct1.submit(
				    [&](sycl::handler &cgh) {
				      /*
				      DPCT1054:296: The type of variable temp_storage is declared in device function with the name type_ct4. Adjust the code to make the type_ct4 declaration visible at the accessor declaration point.
				      */
                                  
              using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
              using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
              size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
              size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
              
              sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
              sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
              
                                
				      //sycl::local_accessor<uint8_t[sizeof(type_ct4)], 0> temp_storage_ct1_acc_ct1(cgh);

				      cgh.parallel_for(
				        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
				        [=](sycl::nd_item<3> item_ct1) {
				          kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>(buff_g, buff_p, buff_state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, ltacc_T, ltacc_float);
				        });
				    });
				}
        /*
        DPCT1010:238: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
        */
        //CUDA_CHECK_RETURN(0);
			}

			/*
			DPCT1049:60: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
			*/
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      /*
			      DPCT1054:297: The type of variable temp_storage is declared in device function with the name type_ct5. Adjust the code to make the type_ct5 declaration visible at the accessor declaration point.
			      */
			      using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_T1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T1 = group_load_T1::get_local_memory_size(NUM_BLOCK);

            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_T1(load_temp_storage_size_T1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
            
            
            using group_store_T = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, T>;
            using group_store_float1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            size_t store_temp_storage_size_float1 = group_store_float1::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_T = group_store_T::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> stacc_T(store_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float1(store_temp_storage_size_float1, cgh);
            
            
            
            //sycl::local_accessor<uint8_t[sizeof(type_ct5)], 0> temp_storage_ct1_acc_ct1(cgh);

			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizer32bit1State<T, OPTIMIZER>(buff_g, buff_p, buff_state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, ltacc_T, ltacc_T1, ltacc_float1, stacc_T,stacc_float1);
			        });
			    });
			}
      /*
      DPCT1010:239: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //CUDA_CHECK_RETURN(0);
			break;
    case LION:
      // in lion, the momentum update after the parameter update
      /*
      DPCT1049:63: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
      {
        dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            /*
            DPCT1054:298: The type of variable temp_storage is declared in device function with the name type_ct5. Adjust the code to make the type_ct5 declaration visible at the accessor declaration point.
            */
            //sycl::local_accessor<uint8_t[sizeof(type_ct5)], 0> temp_storage_ct1_acc_ct1(cgh);
              using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
              using group_load_T1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
              using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
              size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
              size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
              size_t load_temp_storage_size_T1 = group_load_T1::get_local_memory_size(NUM_BLOCK);
  
              sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
              sycl::local_accessor<uint8_t, 1> ltacc_T1(load_temp_storage_size_T1, cgh);
              sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
              
              
              using group_store_T = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, T>;
              using group_store_float1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
              size_t store_temp_storage_size_float1 = group_store_float1::get_local_memory_size(NUM_BLOCK);
              size_t store_temp_storage_size_T = group_store_T::get_local_memory_size(NUM_BLOCK);
              
              sycl::local_accessor<uint8_t, 1> stacc_T(store_temp_storage_size_T, cgh);
              sycl::local_accessor<uint8_t, 1> stacc_float1(store_temp_storage_size_float1, cgh);
              
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
              [=](sycl::nd_item<3> item_ct1) {
                kOptimizer32bit1State<T, OPTIMIZER>(buff_g, buff_p, buff_state1, unorm, max_unorm, param_norm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, skip_zeros, n, item_ct1, ltacc_T, ltacc_T1, ltacc_float1, stacc_T,stacc_float1);
              });
          });
      }
      /*
      DPCT1010:240: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //CUDA_CHECK_RETURN(0);

      if(max_unorm > 0.0f)
      {
        CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait()));
        /*
        DPCT1049:64: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
          q_ct1.submit(
            [&](sycl::handler &cgh) {
              /*
              DPCT1054:299: The type of variable temp_storage is declared in device function with the name type_ct4. Adjust the code to make the type_ct4 declaration visible at the accessor declaration point.
              */
              //sycl::local_accessor<uint8_t[sizeof(type_ct4)], 0> temp_storage_ct1_acc_ct1(cgh);
              using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
              using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
              size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
              size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
              
              sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
              sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
              
              cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
                [=](sycl::nd_item<3> item_ct1) {
                  kPreconditionOptimizer32bit1State<T, OPTIMIZER, 4096, 8>(buff_g, buff_p, buff_state1, unorm, beta1, beta2, eps, weight_decay, step, lr, gnorm_scale, n, item_ct1, ltacc_T, ltacc_float);
                });
            });
        }
        /*
        DPCT1010:241: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
        */
        //CUDA_CHECK_RETURN(0);
      }
      break;
	}
 
 //back memcpy
 q_ct1.memcpy((T*)(g), (T*)(buff_g), size);
 q_ct1.memcpy((T*)(p), (T*)(buff_p), size);
 q_ct1.memcpy((float*)(state1), (float*)(buff_state1), size);
 q_ct1.memcpy((float*)(state2), (float*)(buff_state2), size);
  
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template<typename T, int OPTIMIZER> void optimizerStatic8bit(T* buff_p, T* buff_g,
                unsigned char* buff_state1, unsigned char* buff_state2,
                float *unorm, float max_unorm, float param_norm,
                float beta1, float beta2,
                float eps, int step, float lr,
                float* quantiles1, float* quantiles2,
                float* max1, float* max2, float* new_max1, float* new_max2,
                float weight_decay,
                const float gnorm_scale, int n)
 try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int num_blocks = n/4096;
  num_blocks = n % 4096 == 0 ? num_blocks : num_blocks + 1;
  sycl::context ctx = q_ct1.get_context();
  int size = NUM_BLOCK;
  
  *((T **)&buff_g) = sycl::malloc_device(size, g, ctx);
  *((T **)&buff_p) = sycl::malloc_device(size, p, ctx);
  *((float **)&buff_state1 = sycl::malloc_device(size, state1, ctx);
  *((float **)&buff_state2 = sycl::malloc_device(size, state2, ctx);
  q_ct1.memcpy((T*)(buff_g), (T*)(g), size);
  q_ct1.memcpy((T*)(buff_p), (T*)(p), size);
  q_ct1.memcpy((float*)(buff_state1), (float*)(state1), size);
  q_ct1.memcpy((float*)(buff_state2), (float*)(state2), size);
  

  if(max_unorm > 0.0f){ CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(unorm, 0, 1*sizeof(float)).wait())); }

	switch(OPTIMIZER)
	{
		case ADAM:
			CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(new_max1, 0, 1*sizeof(float)).wait()));
			CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(new_max2, 0, 1*sizeof(float)).wait()));
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      /*
			      DPCT1054:300: The type of variable temp_storage is declared in device function with the name type_ct6. Adjust the code to make the type_ct6 declaration visible at the accessor declaration point.
			      */
			      //sycl::local_accessor<uint8_t[sizeof(type_ct6)], 0> temp_storage_ct1_acc_ct1(cgh);
			      //sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
			      //sycl::local_accessor<float, 1> smem_quantiles2_acc_ct1(sycl::range<1>(256), cgh);
                                              
            using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            using group_load_float2 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_float2 = group_load_float2::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float2(load_temp_storage_size_float2, cgh);
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);        
			      sycl::local_accessor<float, 1> smem_quantiles2_acc_ct1(sycl::range<1>(256), cgh);
              

			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kPreconditionOptimizerStatic8bit2State<T, OPTIMIZER>(buff_p, buff_g, buff_state1, buff_state2, unorm, beta1, beta2, eps, step, quantiles1, quantiles2, max1, max2, new_max1, new_max2, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), smem_quantiles2_acc_ct1.get_pointer(), ltacc_T, ltacc_float1, ltacc_float2);
			        });
			    });
			}
			/*
			DPCT1010:242: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
			*/
			//CUDA_CHECK_RETURN(0);
			/*
			DPCT1049:65: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
			*/
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
			      sycl::local_accessor<float, 1> smem_quantiles2_acc_ct1(sycl::range<1>(256), cgh);
			      /*
			      DPCT1054:301: The type of variable temp_storage is declared in device function with the name type_ct7. Adjust the code to make the type_ct7 declaration visible at the accessor declaration point.
			      */
                    
            using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_T1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            using group_load_float2 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_float2 = group_load_float2::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T1 = group_load_T1::get_local_memory_size(NUM_BLOCK);

            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_T1(load_temp_storage_size_T1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float2(load_temp_storage_size_float2, cgh);
            
            
            using group_store_T = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, T>;
            using group_store_float1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            using group_store_float2 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            size_t store_temp_storage_size_float1 = group_store_float1::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_float2 = group_store_float2::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_T = group_store_T::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> stacc_T(store_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float1(store_temp_storage_size_float1, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float2(store_temp_storage_size_float2, cgh);
     
			      //sycl::local_accessor<uint8_t[sizeof(type_ct7)], 0> temp_storage_ct1_acc_ct1(cgh);

            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);        
			      sycl::local_accessor<float, 1> smem_quantiles2_acc_ct1(sycl::range<1>(256), cgh);
            
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit2State<T, OPTIMIZER>(buff_p, buff_g, buff_state1, buff_state2, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, quantiles2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), smem_quantiles2_acc_ct1.get_pointer(), ltacc_T, ltacc_T1, ltacc_float1, ltacc_float2, stacc_T, stacc_float1, stacc_float2);
			        });
			    });
			}
			/*
			DPCT1010:243: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
			*/
			//CUDA_CHECK_RETURN(0);
		break;
		case MOMENTUM:
    case RMSPROP:
    case ADAGRAD:
			CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(new_max1, 0, 1*sizeof(float)).wait()));
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      /*
			      DPCT1054:302: The type of variable temp_storage is declared in device function with the name type_ct8. Adjust the code to make the type_ct8 declaration visible at the accessor declaration point.
			      */
			      //sycl::local_accessor<uint8_t[sizeof(type_ct8)], 0> temp_storage_ct1_acc_ct1(cgh);
			      ;
            using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
            
			      //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh)                   
       
            cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER>(buff_p, buff_g, buff_state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), ltacc_T, ltacc_float1);
			        });
			    });
			}
			/*
			DPCT1010:244: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
			*/
			//CUDA_CHECK_RETURN(0);
			/*
			DPCT1049:66: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
			*/
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      
			      /*
			      DPCT1054:303: The type of variable temp_storage is declared in device function with the name type_ct9. Adjust the code to make the type_ct9 declaration visible at the accessor declaration point.
			      */
			      //sycl::local_accessor<uint8_t[sizeof(type_ct9)], 0> temp_storage_ct1_acc_ct1(cgh);
                                                                    
            using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_T1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T1 = group_load_T1::get_local_memory_size(NUM_BLOCK);

            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_T1(load_temp_storage_size_T1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
            
            
            using group_store_T = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, T>;
            using group_store_float1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            size_t store_temp_storage_size_float1 = group_store_float1::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_T = group_store_T::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> stacc_T(store_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float1(store_temp_storage_size_float1, cgh);
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit1State<T, OPTIMIZER>(buff_p, buff_g, buff_state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1,smem_quantiles1_acc_ct1.get_pointer(), ltacc_T, ltacc_T1, ltacc_float1, stacc_T, stacc_float1);
			        });
			    });
			}
			/*
			DPCT1010:245: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
			*/
			//CUDA_CHECK_RETURN(0);
			break;
    case LION:
      // in lion, the momentum update happens after the parameter update
      /*
      DPCT1049:67: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
      {
        dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            
            /*
            DPCT1054:304: The type of variable temp_storage is declared in device function with the name type_ct9. Adjust the code to make the type_ct9 declaration visible at the accessor declaration point.
            */
            //sycl::local_accessor<uint8_t[sizeof(type_ct9)], 0> temp_storage_ct1_acc_ct1(cgh);
            
            using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_T1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T1 = group_load_T1::get_local_memory_size(NUM_BLOCK);

            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_T1(load_temp_storage_size_T1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
            
            
            using group_store_T = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, T>;
            using group_store_float1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            size_t store_temp_storage_size_float1 = group_store_float1::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_T = group_store_T::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> stacc_T(store_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float1(store_temp_storage_size_float1, cgh);
            
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
            
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), 
              [=](sycl::nd_item<3> item_ct1) {
                kOptimizerStatic8bit1State<T, OPTIMIZER>(`buff_p, buff_g, buff_state1, unorm, max_unorm, param_norm, beta1, beta2, eps, step, lr, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), ltacc_T, ltacc_T1, ltacc_float1, stacc_T, stacc_float1);
              });
          });
      }
      /*
      DPCT1010:246: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //CUDA_CHECK_RETURN(0);

      CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(new_max1, 0, 1*sizeof(float)).wait()));
      {
        dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
        q_ct1.submit(
          [&](sycl::handler &cgh) {
            /*
            DPCT1054:305: The type of variable temp_storage is declared in device function with the name type_ct8. Adjust the code to make the type_ct8 declaration visible at the accessor declaration point.
            */
            //sycl::local_accessor<uint8_t[sizeof(type_ct8)], 0> temp_storage_ct1_acc_ct1(cgh);
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
              
            //__shared__ vars
            sycl::local_accessor<float, 1> smem_quantiles1_acc_ct1(sycl::range<1>(256), cgh);
            
            cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
              [=](sycl::nd_item<3> item_ct1) {
                kPreconditionOptimizerStatic8bit1State<T, OPTIMIZER>(buff_p, buff_g, buff_state1, unorm, beta1, beta2, eps, step, quantiles1, max1, new_max1, weight_decay, gnorm_scale, n, item_ct1, smem_quantiles1_acc_ct1.get_pointer(), ltacc_T, ltacc_float1);
              });
          });
      }
      /*
      DPCT1010:247: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //CUDA_CHECK_RETURN(0);
      break;
		default:
			break;
	}
 
 //back memcpy
 q_ct1.memcpy((T*)(buff_g), (T*)(g), size);
 q_ct1.memcpy((T*)(buff_p), (T*)(p), size);
 q_ct1.memcpy((float*)(buff_state1), (float*)(state1), size);
 q_ct1.memcpy((float*)(buff_state2), (float*)(state2), size);
 
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#define BLOCKSIZE_2STATE 2048
#define NUM_2STATE 8
#define BLOCKSIZE_1STATE 2048
#define NUM_1STATE 8

template<typename T, int OPTIMIZER> void optimizerStatic8bitBlockwise(T* buff_p, T* buff_g,
                unsigned char* buff_state1, unsigned char* buff_state2, float beta1, float beta2, float eps, int step, float lr,
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n)
 try {
    
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  *((T **)&buff_g) = sycl::malloc_device(size, g, ctx);
  *((T **)&buff_p) = sycl::malloc_device(size, p, ctx);
  *((float **)&buff_state1 = sycl::malloc_device(size, state1, ctx);
  *((float **)&buff_state2 = sycl::malloc_device(size, state2, ctx);
  q_ct1.memcpy((T*)(buff_g), (T*)(g), size);
  q_ct1.memcpy((T*)(buff_p), (T*)(p), size);
  q_ct1.memcpy((float*)(buff_state1), (float*)(state1), size);
  q_ct1.memcpy((float*)(buff_state2), (float*)(state2), size);
  
   
	switch(OPTIMIZER)
	{
		case ADAM:
			num_blocks = n/BLOCKSIZE_2STATE;
			num_blocks = n % BLOCKSIZE_2STATE == 0 ? num_blocks : num_blocks + 1;
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      /*
			      DPCT1101:306: 'LANES' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
			      */
			      /*
			      DPCT1101:307: 'LANES' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
			      */
                    
            /*
			      DPCT1054:308: The type of variable temp_storage is declared in device function with the name type_ct10. Adjust the code to make the type_ct10 declaration visible at the accessor declaration point.
			      */
			      //sycl::local_accessor<uint8_t[sizeof(type_ct10)], 0> temp_storage_ct1_acc_ct1(cgh);
            using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_T1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            using group_load_float2 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_float2 = group_load_float2::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T1 = group_load_T1::get_local_memory_size(NUM_BLOCK);

            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_T1(load_temp_storage_size_T1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float2(load_temp_storage_size_float2, cgh);
            
            
            using group_store_T = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, T>;
            using group_store_float1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            using group_store_float2 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            size_t store_temp_storage_size_float1 = group_store_float1::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_float2 = group_store_float2::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_T = group_store_T::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> stacc_T(store_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float1(store_temp_storage_size_float1, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float2(store_temp_storage_size_float2, cgh);
            
            
            //__shared__ vars
            sycl::local_accessor<float, 2> smem_quantiles1_acc_ct1(sycl::range<2>(2/*LANES*/, 257), cgh);        
			      sycl::local_accessor<float, 2> smem_quantiles2_acc_ct1(sycl::range<2>(2/*LANES*/, 257), cgh);
			      sycl::local_accessor<float, 1> smem_exchange1_acc_ct1(sycl::range<1>(1), cgh);
			      sycl::local_accessor<float, 1> smem_exchange2_acc_ct1(sycl::range<1>(1), cgh);
			      
     
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, BLOCKSIZE_2STATE/NUM_2STATE), sycl::range<3>(1, 1, BLOCKSIZE_2STATE/NUM_2STATE)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit2StateBlockwise<T, OPTIMIZER, BLOCKSIZE_2STATE, NUM_2STATE>(buff_p, buff_g, buff_state1, buff_state2, beta1, beta2, eps, step, lr, quantiles1, quantiles2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros, n,item_ct1,  smem_quantiles1_acc_ct1, smem_quantiles2_acc_ct1,smem_exchange1_acc_ct1.get_pointer(), smem_exchange2_acc_ct1.get_pointer(),ltacc_T, ltacc_T1, ltacc_float1, ltacc_float2, stacc_T, stacc_float1, stacc_float2);
			        });
			    });
			}
			/*
			DPCT1010:248: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
			*/
			//CUDA_CHECK_RETURN(0);
		break;
		case MOMENTUM:
		case RMSPROP:
    case ADAGRAD:
    case LION:
			num_blocks = n/BLOCKSIZE_1STATE;
			num_blocks = n % BLOCKSIZE_1STATE == 0 ? num_blocks : num_blocks + 1;
			{
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
			      /*
			      DPCT1101:309: 'LANES' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
			      */
			      
			      /*
			      DPCT1054:310: The type of variable temp_storage is declared in device function with the name type_ct11. Adjust the code to make the type_ct11 declaration visible at the accessor declaration point.
			      */
			      //sycl::local_accessor<uint8_t[sizeof(type_ct11)], 0> temp_storage_ct1_acc_ct1(cgh);
            using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_T1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
            using group_load_float1 = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, float>;
            size_t load_temp_storage_size_float1 = group_load_float1::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
            size_t load_temp_storage_size_T1 = group_load_T1::get_local_memory_size(NUM_BLOCK);

            sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_T1(load_temp_storage_size_T1, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_float1(load_temp_storage_size_float1, cgh);
            
            
            using group_store_T = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, T>;
            using group_store_float1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, float>;
            size_t store_temp_storage_size_float1 = group_store_float1::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_T = group_store_T::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> stacc_T(store_temp_storage_size_T, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_float1(store_temp_storage_size_float1, cgh);
            
            
            //__shared__ vars
            sycl::local_accessor<float, 2> smem_quantiles1_acc_ct1(sycl::range<2>(2/*LANES*/, 257), cgh);
			      sycl::local_accessor<float, 1> smem_exchange1_acc_ct1(sycl::range<1>(1), cgh);
            
			      cgh.parallel_for(
			        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE), sycl::range<3>(1, 1, BLOCKSIZE_1STATE/NUM_1STATE)), 
			        [=](sycl::nd_item<3> item_ct1) {
			          kOptimizerStatic8bit1StateBlockwise<T, OPTIMIZER, BLOCKSIZE_1STATE, NUM_1STATE>(buff_p, buff_g, buff_state1, beta1, beta2, eps, step, lr, quantiles1, absmax1, weight_decay, gnorm_scale, skip_zeros, n, item_ct1, smem_quantiles1_acc_ct1, smem_exchange1_acc_ct1.get_pointer(), ltacc_T, ltacc_T1, ltacc_float1, stacc_T, stacc_float1);
			        });
			    });
			}
			/*
			DPCT1010:249: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
			*/
			//CUDA_CHECK_RETURN(0);
		break;
	}
 q_ct1.memcpy((T*)(g), (T*)(buff_g), size);
 q_ct1.memcpy((T*)(p), (T*)(buff_p), size);
 q_ct1.memcpy((float*)(state1), (float*)(buff_state1), size);
 q_ct1.memcpy((float*)(state2), (float*)(buff_state2), size);
  
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}



template<typename T> void percentileClipping(T * g, float *gnorm_vec, int step, const int n)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::context ctx = q_ct1.get_context();
    
  int num_blocks = n/2048;
  num_blocks = n % 2048 == 0 ? num_blocks : num_blocks + 1;
  int size = NUM_BLOCK;
  *((T **)&buff_g) = sycl::malloc_device(size, g, ctx);
  q_ct1.memcpy((T*)(buff_g), (T*)(g), size);
  
  
	CUDA_CHECK_RETURN(DPCT_CHECK_ERROR(q_ct1.memset(&gnorm_vec[step % 100], 0, 1*sizeof(float)).wait()));
  /*
  DPCT1049:68: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
    q_ct1.submit(
      [&](sycl::handler &cgh) {
        using group_load_T = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, T>;
        size_t load_temp_storage_size_T = group_load_T::get_local_memory_size(NUM_BLOCK);
        sycl::local_accessor<uint8_t, 1> ltacc_T(load_temp_storage_size_T, cgh);
                  
      cgh.parallel_for(    
       sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
      [=](sycl::nd_item<3> item_ct1) {
        kPercentileClipping<T, 2048, 4>(g, gnorm_vec, step, n, item_ct1, ltacc);
      });
    });
  }
  /*
  DPCT1010:250: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);
  //back memcpy
   q_ct1.memcpy((T*)(g), (T*)(buff_g), size);
}




//========================GEMM============================

void gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc)
 try {
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	int status;

			status = DPCT_CHECK_ERROR(dpct::gemm(*context->m_handle, transposeA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, transposeB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, m, n, k, alpha, A, dpct::library_data_t::real_int8, lda, B, dpct::library_data_t::real_int8, ldb, beta, C, dpct::library_data_t::real_int32, ldc, dpct::library_data_t::real_int32));

    if (status != 0)
    {
      std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }

}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void strided_gemmex(Context *context, bool transposeA, bool transposeB, int m, int n, int k, void *A, void *B, void *C, int lda, int ldb, int ldc,
                    long long int strideA, long long int strideB, long long int strideC, int batchCount)
 try {
  const int falpha = 1;
  const int fbeta = 0;
  const void * alpha = &falpha;
  const void * beta = &fbeta;
	int status;

  //cout << transposeA << transposeB << endl;
  //printf("%i %i %i\n", m,n,k);
  //printf("%i %i %i\n", lda,ldb,ldc);
  //printf("%i %i %i\n", strideA, strideB, strideC);
  //printf("%i\n", batchCount);

			status = DPCT_CHECK_ERROR(dpct::gemm_batch(*context->m_handle, transposeA ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, transposeB ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, m, n, k, alpha, A, dpct::library_data_t::real_int8, lda, (long long int)strideA, B, dpct::library_data_t::real_int8, ldb, (long long int)strideB, beta, C, dpct::library_data_t::real_int32, ldc, (long long int)strideC, batchCount, dpct::library_data_t::real_int32));

    if (status != 0)
    {
      std::cout << "CUBLAS ERROR: Status " << status << std::endl;
    }

}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}


#ifdef NO_CUBLASLT
#else
template<int ORDER> cublasLtOrder_t get_order()
{
	switch(ORDER)
	{
		case ROW:
      return CUBLASLT_ORDER_ROW;
			break;
    case COL:
      return CUBLASLT_ORDER_COL;
      break;
    case COL32:
      return CUBLASLT_ORDER_COL32;
      break;
    case COL_TURING:
      return CUBLASLT_ORDER_COL4_4R2_8C;
      break;
    case COL_AMPERE:
      return CUBLASLT_ORDER_COL32_2R_4R4;
      break;
		default:
			break;
  }

	return CUBLASLT_ORDER_ROW;
}

template cublasLtOrder_t get_order<ROW>();
template cublasLtOrder_t get_order<COL>();
template cublasLtOrder_t get_order<COL32>();
template cublasLtOrder_t get_order<COL_TURING>();
template cublasLtOrder_t get_order<COL_AMPERE>();
#endif


template<int ORDER> int get_leading_dim(int dim1, int dim2)
{
	switch(ORDER)
	{
		case ROW:
      return dim2;
			break;
    case COL:
      return dim1;
      break;
    case COL32:
      // 32*row tiles
      return dim1*32;
      break;
    case COL_TURING:
      return 32*roundoff(dim1, 8);
      break;
    case COL_AMPERE:
      // 32*32 tiles
      return 32*roundoff(dim1, 32);
      break;
		default:
			return 0;
			break;
  }
}

template int get_leading_dim<ROW>(int dim1, int dim2);
template int get_leading_dim<COL>(int dim1, int dim2);
template int get_leading_dim<COL32>(int dim1, int dim2);

template <typename T, int SRC, int TARGET, bool transpose, int DTYPE> void transform(cublasLtHandle_t ltHandle, T *A, T *out, int dim1, int dim2)
{
#ifdef NO_CUBLASLT
#else
  cublasLtOrder_t orderA = get_order<SRC>();
  cublasLtOrder_t orderOut = get_order<TARGET>();
  int ldA = get_leading_dim<SRC>(dim1, dim2);
  int ldOut = get_leading_dim<TARGET>(dim1, dim2);

  cublasLtMatrixLayout_t A_desc = NULL, out_desc = NULL;
  cublasLtMatrixTransformDesc_t A2Out_desc = NULL;
  oneapi::mkl::transpose opTranspose = oneapi::mkl::transpose::trans;
  float transformAlpha = 1.0f, transformBeta = 0.0f;


  if(DTYPE == 8)
  {
    /*
    DPCT1007:251: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    checkCublasStatus(cublasLtMatrixLayoutCreate(&A_desc, dpct::library_data_t::real_int8, dim1, dim2, ldA));
    /*
    DPCT1007:252: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    checkCublasStatus(cublasLtMatrixLayoutCreate(&out_desc, dpct::library_data_t::real_int8, dim1, dim2, ldOut));
  }
  else if(DTYPE == 32)
  {
    /*
    DPCT1007:253: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    checkCublasStatus(cublasLtMatrixLayoutCreate(&A_desc, dpct::library_data_t::real_int32, dim1, dim2, ldA));
    /*
    DPCT1007:254: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    checkCublasStatus(cublasLtMatrixLayoutCreate(&out_desc, dpct::library_data_t::real_int32, dim1, dim2, ldOut));
  }
  else
  {
    printf("ERROR WRONG TYPE FOR TRANSFORM: %i\n", DTYPE);
  }

  /*
  DPCT1007:255: Migration of cublasLtMatrixLayoutSetAttribute is not supported.
  */
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
  /*
  DPCT1007:256: Migration of cublasLtMatrixLayoutSetAttribute is not supported.
  */
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(out_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderOut, sizeof(orderOut)));

  /*
  DPCT1007:257: Migration of cublasLtMatrixTransformDescCreate is not supported.
  */
  checkCublasStatus(cublasLtMatrixTransformDescCreate(&A2Out_desc, dpct::library_data_t::real_float));

  /*
  DPCT1007:258: Migration of cublasLtMatrixTransformDescSetAttribute is not supported.
  */
  if(transpose){ checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(A2Out_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose))); }

  checkCublasStatus(cublasLtMatrixTransform(ltHandle, A2Out_desc, &transformAlpha, A, A_desc, &transformBeta, NULL, NULL, out, out_desc, 0));

  /*
  DPCT1007:259: Migration of cublasLtMatrixLayoutDestroy is not supported.
  */
  if (A_desc) checkCublasStatus(cublasLtMatrixLayoutDestroy(A_desc));
  /*
  DPCT1007:260: Migration of cublasLtMatrixLayoutDestroy is not supported.
  */
  if (out_desc) checkCublasStatus(cublasLtMatrixLayoutDestroy(out_desc));
  /*
  DPCT1007:261: Migration of cublasLtMatrixTransformDescDestroy is not supported.
  */
  if (A2Out_desc) checkCublasStatus(cublasLtMatrixTransformDescDestroy(A2Out_desc));
#endif
}

template void transform<int8_t, ROW, COL, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, ROW, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL32, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, ROW, COL32, false, 32>(cublasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_TURING, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, ROW, COL_AMPERE, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int8_t, COL32, ROW, false, 8>(cublasLtHandle_t ltHandle, int8_t *A, int8_t *out, int dim1, int dim2);
template void transform<int32_t, COL32, ROW, false, 32>(cublasLtHandle_t ltHandle, int32_t *A, int32_t *out, int dim1, int dim2);

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int igemmlt(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
 try {
    using namespace dnnl;
    using tag = memory::format_tag;
    using dt = memory::data_type;
    auto dev = sycl::device(sycl::gpu_selector_v);
    auto ctx = sycl::context(dev);
    
    dnnl::engine engine = sycL_interop::make_engine(dev, ctx);
    // column major 
    const memory::dims a_strides = memory::dims {1, lda};
    const auto a_md = memory::desc({m, k}, dt::s8, a_strides);
    const memory::dims b_strides = memory::dims {ldb, 1};
    const auto b_md = memory::desc({k, n}, dt::s8, b_strides);
    const memory::dims c_strides = memory::dims {ldc, 1};
    const auto c_md = DTYPE_OUT == 32 ? memory::desc({m, n}, dt::s32 c_strides) : memory::desc({m, n}, dt::s8 c_strides);
    
    //memory align
    memory a_mem(a_md, engine A);
    memory b_mem(b_md, engine, B);
    memory c_mem(c_md, engine, C);
    memory scales_C_mem({{1}, dt::f32, {1}}, engine, row_scale);
    
    //create dnnl stream
    auto q_ct1 = sycl::queue(ctx, dev);
    dnnl::stream stream = sycl_interop::make_stream(q_ct1);
    
    primitive_attr attr;
    if (SCALE_ROWS) {
        attr.set_scales_mask(DNNL_ARG_DST, /* mask */ 1 << 1);
    }
    
    auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md, attr);
    auto matmul_prim = matmul(matmul_pd);
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});

    if (SCALE_ROWS) {
      matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scales_C_mem});
    }
    matmul_prim.execute(stream, matmul_args);
    stream.wait();

//#ifdef NO_CUBLASLT
//	return ERR_NOT_IMPLEMENTED;
//#else
    //int has_error = 0;
    //cublasLtMatmulDesc_t matmulDesc = NULL;
    //cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    //oneapi::mkl::transpose opT = oneapi::mkl::transpose::trans;
    //cublasLtPointerMode_t alphaVec = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
    //cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
    //cublasLtOrder_t col_turing = CUBLASLT_ORDER_COL4_4R2_8C;
    //cublasLtOrder_t col_ampere = CUBLASLT_ORDER_COL32_2R_4R4;

    /*
    DPCT1007:262: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    //has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, dpct::library_data_t::real_int8, m, k, lda));
    /*
    DPCT1007:263: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    //has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, dpct::library_data_t::real_int8, n, k, ldb));

    /*
    DPCT1007:264: Migration of cublasLtMatrixLayoutSetAttribute is not supported.
    */
    //has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
    //if(FORMATB == COL_TURING)
      /*
      DPCT1007:265: Migration of cublasLtMatrixLayoutSetAttribute is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_turing, sizeof(col_turing)));
    //else
      /*
      DPCT1007:266: Migration of cublasLtMatrixLayoutSetAttribute is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_ampere, sizeof(col_ampere)));
    
    //if(DTYPE_OUT == 32)
     //{
      /*
      DPCT1007:267: Migration of cublasLtMatmulDescCreate is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, dpct::library_data_t::real_int32));
      /*
      DPCT1007:268: Migration of cublasLtMatmulDescSetAttribute is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      /*
      DPCT1007:269: Migration of cublasLtMatrixLayoutCreate is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, dpct::library_data_t::real_int32, m, n, ldc));
      /*
      DPCT1007:270: Migration of cublasLtMatrixLayoutSetAttribute is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      //int alpha = 1, beta = 0;
      /*
      DPCT1007:271: Migration of cublasLtMatmul is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int32_t*)C, Cdesc, (int32_t*)C, Cdesc, NULL, NULL, 0, &q_ct1));
    //}
    //else
    //{
      /*
      DPCT1007:272: Migration of cublasLtMatmulDescCreate is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, dpct::library_data_t::real_float));
      /*
      DPCT1007:273: Migration of cublasLtMatmulDescSetAttribute is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      /*
      DPCT1007:274: Migration of cublasLtMatrixLayoutCreate is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, dpct::library_data_t::real_int8, m, n, ldc));
      /*
      DPCT1007:275: Migration of cublasLtMatrixLayoutSetAttribute is not supported.
      */
      //has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      //if(!SCALE_ROWS)
      //{
        //float alpha = 1.0f, beta = 0.0f;
        /*
        DPCT1007:276: Migration of cublasLtMatmul is not supported.
        */
        //has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, &q_ct1));
      //}
      //else
      //{
        /*
        DPCT1007:277: Migration of cublasLtMatmulDescSetAttribute is not supported.
        */
        //has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &alphaVec, sizeof(alphaVec)));
        /*
        DPCT1007:278: Migration of cublasLtMatmul is not supported.
        */
        //has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc, row_scale, A, Adesc, B, Bdesc, NULL, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, &q_ct1));
      //}
    //}


    /*
    DPCT1007:279: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    //if (Cdesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    /*
    DPCT1007:280: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    //if (Bdesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    /*
    DPCT1007:281: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    //if (Adesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    /*
    DPCT1007:282: Migration of cublasLtMatmulDescDestroy is not supported.
    */
    //if (matmulDesc) has_error |= checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    //if(has_error == 1)
      //printf("error detected");

    //return has_error;
//#endif // NO_CUBLASLT
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int fill_up_to_nearest_multiple(int value, int multiple)
{
  return value + (value % multiple == 0 ? 0 : (multiple - (value % multiple)));
}

void dequant_mm_int32_fp16(int *A, float *rowStats, float *colStats, sycl::half *out, float* newRowStats, float* newcolStats, sycl::half *bias, int numRows, int numCols)
{
  int threads = 512;
  int tileCols = fill_up_to_nearest_multiple(numCols, 32);
  int n = numRows*tileCols;
  int subtile_rows = 128;
  int tilesize = 32*subtile_rows;
  int num_blocks = numRows/subtile_rows;
  num_blocks += (numRows % subtile_rows == 0) ? 0 : 1;
  num_blocks = num_blocks*(tileCols/32);
  assert(threads <= tilesize);

  kdequant_mm_int32_fp16<4, 128, 512><<<num_blocks, threads>>>(A, rowStats, colStats, out, newRowStats, newcolStats, bias, numRows, numCols, tileCols, n);
  /*
  DPCT1010:283: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_CHECK_RETURN(0);
}


//===========================Row col stats=================================

#define STATS_THREADS 64
#define STATS_ITEMS 4
#define STATS_ROWS 16
void getColRowStats(sycl::half * A, float *rowStats, float *colStats, int *nnz_count_row, float nnz_threshold, int rows, int cols)
{
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  
  int tile_cols = STATS_THREADS*STATS_ITEMS;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, STATS_ROWS);
	int row_tiles = (tiledRows/STATS_ROWS);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;
  
  int size = NUM_BLOCK;
  *((sycl::half **)&buff_A) = sycl::malloc_device(size, A, ctx);
  q_ct1.memcpy((sycl::half*)(buff_A), (sycl::half*)(A), size);
  

  if(nnz_threshold == 0.0)
    {
    
			  dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
			  q_ct1.submit(
			    [&](sycl::handler &cgh) {
            using group_load_half = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, sycl::half>;
            using group_exchange = dpct::group::exchange<float, ITEMS_PER_THREAD>;
            
            size_t load_temp_storage_size_half = group_load_half::get_local_memory_size(NUM_BLOCK);
            size_t exchange_temp_storage_size = group_exchange::get_local_memory_size(NUM_BLOCK);
                
            sycl::local_accessor<uint8_t, 1> exacc(exchange_temp_storage_size, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_half(load_temp_storage_size_half, cgh);
                        
       cgh.parallel_for(      
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
            [=](sycl::nd_item<3> item_ct1) {
              kgetColRowStats<sycl::half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(buff_A, rowStats, colStats,
               nnz_count_row,    nnz_threshold, rows, cols, tiledRows, tiledCols, ltacc_half, exacc);
            });
       });
    }
  else if(nnz_threshold != 0.0)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
		    [&](sycl::handler &cgh) {
            using group_load_half = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, sycl::half>;
            using group_exchange = dpct::group::exchange<float, ITEMS_PER_THREAD>;
            
            size_t load_temp_storage_size_half = group_load_half::get_local_memory_size(NUM_BLOCK);
            size_t exchange_temp_storage_size = group_exchange::get_local_memory_size(NUM_BLOCK);
                
            sycl::local_accessor<uint8_t, 1> exacc(exchange_temp_storage_size, cgh);
            sycl::local_accessor<uint8_t, 1> ltacc_half(load_temp_storage_size_half, cgh);
                        
      cgh.parallel_for(      
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
          [=](sycl::nd_item<3> item_ct1) {
            kgetColRowStats<sycl::half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(buff_A, rowStats, colStats,
             nnz_count_row, nnz_threshold, rows, cols, tiledRows, tiledCols, ltacc_half, exacc);
      });
    });
    }
    
  /*
  DPCT1010:284: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);

}

void doubleRowColQuant(sycl::half * A, float *rowStats, float *colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, sycl::half *val, int *nnz_block_ptr, float threshold, int rows, int cols)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  *((sycl::half **)&buff_A) = sycl::malloc_device(size, A, ctx);
  *((char **)&buff_out_row_normed) = sycl::malloc_device(size, out_row_normed, ctx);
  *((char **)&buff_out_col_normed = sycl::malloc_device(size, out_col_normed, ctx);
  q_ct1.memcpy((sycl::half*)(buff_A), (sycl::half*)(A), size);
  q_ct1.memcpy((char*)(buff_out_row_normed), (char*)(out_row_normed), size);
  q_ct1.memcpy((char*)(buff_out_col_normed), (char*)(out_col_normed), size);
  
  int threads = 64;
  int items_per_thread = 4;
  int tile_cols = threads*items_per_thread;
  int tile_rows = 16;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;


  if(threshold > 0.0f)
    {
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
		    [&](sycl::handler &cgh) {
            using group_load_half = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, sycl::half>;
            using group_store_char1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, char>;
            using group_store_char2 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, char>;
            
            size_t load_temp_storage_size_half = group_load_half::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_char1 = group_store_char1::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_char2 = group_store_char2::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> ltacc_half(load_temp_storage_size_half, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_char1(store_temp_storage_size_char1, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_char2(store_temp_storage_size_char2, cgh);
                        
        cgh.parallel_for(      
           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
           [=](sycl::nd_item<3> item_ct1) {
          
                kDoubleRowColQuant<sycl::half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(buff_A, rowStats, colStats, buff_out_col_normed, buff_out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols, ltacc_half, stacc_char1, stacc_char2);
          });
      });
    }
  else
    {
  
      dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp16});
      q_ct1.submit(
		    [&](sycl::handler &cgh) {
            using group_load_half = dpct::group::workgroup_load<NUM_BLOCK, BLOCK_LOAD_DIRECT, sycl::half>;
            using group_store_char1 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, char>;
            using group_store_char2 = dpct::group::workgroup_store<NUM_BLOCK, BLOCK_STORE_DIRECT, char>;
            
            size_t load_temp_storage_size_half = group_load_half::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_char1 = group_store_char1::get_local_memory_size(NUM_BLOCK);
            size_t store_temp_storage_size_char2 = group_store_char2::get_local_memory_size(NUM_BLOCK);
            
            sycl::local_accessor<uint8_t, 1> ltacc_half(load_temp_storage_size_half, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_char1(store_temp_storage_size_char1, cgh);
            sycl::local_accessor<uint8_t, 1> stacc_char2(store_temp_storage_size_char2, cgh);
                        
        cgh.parallel_for(      
           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
           [=](sycl::nd_item<3> item_ct1) {
          
                kDoubleRowColQuant<sycl::half, STATS_THREADS, STATS_ITEMS, STATS_ROWS, STATS_THREADS*STATS_ITEMS, 0>(A, rowStats, colStats, out_col_normed, out_row_normed, rowidx, colidx, val, nnz_block_ptr, threshold, rows, cols, tiledCols, ltacc_half, stacc_char1, stacc_char2);
          });
      });
  
  }
  /*
  DPCT1010:285: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);
  q_ct1.memcpy((sycl::half*)(A), (sycl::half*)(buff_A), size);
  q_ct1.memcpy((char*)(out_row_normed), (char*)(buff_out_row_normed), size);
  q_ct1.memcpy((char*)(out_col_normed), (char*)(buff_out_col_normed), size);
  
}

template <int FORMAT, int TRANSPOSE> void transformRowToFormat(char * A, char *out, int rows, int cols)
{
  
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
	int num_blocks = 0;
  int size = NUM_BLOCK;
  
  *((char **)&buff_A) = sycl::malloc_device(size, A, ctx);
  *((char **)&buff_out) = sycl::malloc_device(size, out, ctx);
  q_ct1.memcpy((char*)(buff_A), (sycl::half*)(A), size);
  q_ct1.memcpy((char*)(buff_out), (char*)(out), size);
  
  
  int threads = 256;
  int items_per_thread = 8;
  // we load 128 column values per warp
  int tile_cols = 32*items_per_thread;
  int tile_rows = 32;
  int tiledCols = fill_up_to_nearest_multiple(cols, tile_cols);
  int tiledRows = fill_up_to_nearest_multiple(rows, tile_rows);
	int row_tiles = (tiledRows/tile_rows);
	int col_tiles = (tiledCols/tile_cols);
	row_tiles = row_tiles > 0 ? row_tiles : 1;
	col_tiles = col_tiles > 0 ? col_tiles : 1;
  int num_blocks = row_tiles * col_tiles;

  int outCols = fill_up_to_nearest_multiple(cols, 32);
  int outRows = fill_up_to_nearest_multiple(rows, 32);
  if(FORMAT == COL_TURING)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 8);
    else
      outRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
    if(TRANSPOSE)
      outRows = fill_up_to_nearest_multiple(cols, 32);
    else
      outRows = fill_up_to_nearest_multiple(rows, 32);
  }
  else
  {
    if(TRANSPOSE)
    {
      outCols = fill_up_to_nearest_multiple(rows, 32);
      outRows = cols;
    }
  }

  /*
  DPCT1049:69: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().submit(
    [&](sycl::handler &cgh) {
    
    
    
    //__shared__ vars
      sycl::local_accessor<char, 1> smem_data_acc_ct1(sycl::range<1>(32*33*8), cgh);

      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
        [=](sycl::nd_item<3> item_ct1) {
          kTransformRowToFormat<256, 8, 32, 32*8, TRANSPOSE, FORMAT>(buff_A, buff_out, rows, cols, tiledCols, outRows, outCols, item_ct1, smem_data_acc_ct1.get_pointer());
        });
    });
  /*
  DPCT1010:286: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  
  //CUDA_CHECK_RETURN(0);
  
  q_ct1.memcpy((char*)(A), (sycl::half*)(buff_A), size);
  q_ct1.memcpy((char*)(out), (char*)(buff_out), size);
  
}

void spmm_coo(sycl::queue* handle, int *A_rowidx, int *A_colidx, sycl::half *A_vals, int A_nnz, int A_rows, int A_cols, int B_cols, int ldb, sycl::half *B, int ldc, sycl::half* C, bool transposed_B)
{ 

  try{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();

//#ifdef NO_CUBLASLT
//#else

   
    dpct::sparse::sparse_matrix_desc_t descA;
    std::shared_ptr<dpct::sparse::dense_matrix_desc> descB, descC;

    float alpha = 1.0f;
    float beta = 0.0f;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    /*
    DPCT1007:287: Migration of cusparseCreateCoo is not supported.
    */
    //CHECK_CUSPARSE( cusparseCreateCoo(&descA, A_rows, A_cols, A_nnz,
    //                                  A_rowidx, A_colidx, A_vals,
    //                                  dpct::library_data_t::real_int32,
    //                                  oneapi::mkl::index_base::zero, dpct::library_data_t::real_half) );
    // Create dense matrix C
    //CHECK_CUSPARSE( DPCT_CHECK_ERROR(descC = std::make_shared<dpct::sparse::dense_matrix_desc>(A_rows, B_cols, ldc, C, dpct::library_data_t::real_half, oneapi::mkl::layout::row_major)) );
    descC = std::make_shared<dpct::sparse::dense_matrix_desc>(A_rows, B_cols, ldc, C, dpct::library_data_t::real_half, oneapi::mkl::layout::row_major);
    // Create dense matrix B
    if(transposed_B)
    {
      int tmp = A_cols;
      A_cols = B_cols;
      B_cols = tmp;
    }

    //CHECK_CUSPARSE( DPCT_CHECK_ERROR(descB = std::make_shared<dpct::sparse::dense_matrix_desc>(A_cols, B_cols, ldb, B, dpct::library_data_t::real_half, oneapi::mkl::layout::row_major)) );
    descB = std::make_shared<dpct::sparse::dense_matrix_desc>(A_cols, B_cols, ldb, B, dpct::library_data_t::real_half, oneapi::mkl::layout::row_major);
    // allocate an external buffer if needed
    //CHECK_CUSPARSE( DPCT_CHECK_ERROR(bufferSize = 0) );
    bufferSize = 0
    //CUDA_CHECK_RETURN( DPCT_CHECK_ERROR(dBuffer = (void *)sycl::malloc_device(bufferSize, q_ct1)) );
    dBuffer = (void *)sycl::malloc_device(bufferSize, q_ct1);

    // execute SpMM
    //CHECK_CUSPARSE( DPCT_CHECK_ERROR(dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans, transposed_B ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, &alpha, descA, descB, &beta, descC, dpct::library_data_t::real_float)));
    dpct::sparse::spmm(*handle, oneapi::mkl::transpose::nontrans, transposed_B ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, &alpha, descA, descB, &beta, descC, dpct::library_data_t::real_float);
    // destroy matrix/vector descriptors
    descA.reset();
    descB.reset();
    descC.reset();
    sycl::free(dBuffer, q_ct1);
    //CHECK_CUSPARSE( DPCT_CHECK_ERROR(descA.reset()) );
    //CHECK_CUSPARSE( DPCT_CHECK_ERROR(descB.reset()) );
    //CHECK_CUSPARSE( DPCT_CHECK_ERROR(descC.reset()) );
    //CUDA_CHECK_RETURN( DPCT_CHECK_ERROR(sycl::free(dBuffer, q_ct1)) );
//#endif
  }
  catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
  }

}

template <typename T, int BITS> void spmm_coo_very_sparse_naive(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, sycl::half *values, T *B, sycl::half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB)
{

  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
    dpct::get_in_order_queue().submit(
      [&](sycl::handler &cgh) {
        /*
        DPCT1101:311: 'SMEM_SIZE' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<sycl::half, 1> smem_dequant_stats_acc_ct1(sycl::range<1>(2048/*SMEM_SIZE*/), cgh);

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, nnz_rows) * sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)), 
          [=](sycl::nd_item<3> item_ct1) {
            kspmm_coo_very_sparse_naive<T, 8, BITS>(max_count, max_idx, offset_rowidx, rowidx, colidx, values, B, out, dequant_stats, nnz, rowsA, rowsB, colsB, item_ct1, smem_dequant_stats_acc_ct1.get_pointer());
          });
      });
  }
  /*
  DPCT1010:289: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);
}


template <int FORMAT> void extractOutliers(char * A, int *idx, char *out, int idx_size, int rows, int cols)
{
  int threads = 256;
  // we load 128 column values per warp
  int tiledCols = tiledCols = fill_up_to_nearest_multiple(cols, 32);
  int tiledRows = 0;

	int num_blocks = idx_size;

  if(FORMAT == COL_TURING)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 8);
  }
  else if(FORMAT == COL_AMPERE)
  {
      tiledRows = fill_up_to_nearest_multiple(rows, 32);
	}

  /*
  DPCT1049:70: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, threads), sycl::range<3>(1, 1, threads)), 
    [=](sycl::nd_item<3> item_ct1) {
      kExtractOutliers<FORMAT>(A, idx, out, idx_size, rows, cols, tiledRows, tiledCols, item_ct1);
    });
  /*
  DPCT1010:290: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);
}




template <typename T> void gemm_host(int m, int n, int k, T * A,  T* B,  T * out,  int lda, int ldb, int ldc, int bits)
{

	int num_blocks = (m+31)/32;
 
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  int size = NUM_BLOCK;
  *((T **)&buff_A) = sycl::malloc_device(size, A, ctx);
  q_ct1.memcpy((T*)(buff_A), (T*)(A), size);
  *((T **)&buff_B) = sycl::malloc_device(size, B, ctx);
  q_ct1.memcpy((T*)(buff_B), (T*)(B), size);
  *((T **)&buff_out) = sycl::malloc_device(size, out, ctx);
  q_ct1.memcpy((T*)(buff_out), (T*)(out), size);
  

	//cout << num_blocks << endl;
	//cout << lda << endl;
	//cout << ldb << endl;
	//cout << ldc << endl;

	//cout << m << endl;
	//cout << n << endl;
	//cout << k << endl;
  //if(bits == 32)
    //gemm_device<T, 32, 128><<< num_blocks, 128, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 32, 32><<< num_blocks, 32, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
  if(bits == 16)
    //gemm_device<T, 16, 256><<< num_blocks, 256, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    {
      dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
      dpct::get_in_order_queue().submit(
        [&](sycl::handler &cgh) {
          /*
          DPCT1101:312: '8*16 + (2*16*(batch_size_warps-1))' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
          */
          //sycl::local_accessor<T, 1> smem_A_acc_ct1(sycl::range<1>(224/*8*16 + (2*16*(batch_size_warps-1))*/), cgh);
          /*
          DPCT1101:313: '2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
          */
          //__shared__ vars
          sycl::local_accessor<T, 1> smem_A_acc_ct1(sycl::range<1>(224/*8*16 + (2*16*(batch_size_warps-1))*/), cgh);
          sycl::local_accessor<T, 1> smem_B_acc_ct1(sycl::range<1>(4192/*2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))*/), cgh);

          cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 160), sycl::range<3>(1, 1, 160)), 
            [=](sycl::nd_item<3> item_ct1) {
              gemm_device<T, 16, 160>(m, n, k, buff_A, buff_B, buff_out, lda, ldb, ldc, item_ct1, smem_A_acc_ct1.get_pointer(), smem_B_acc_ct1.get_pointer());
            });
        });
    }
    //gemm_device<T, 16, 128><<< num_blocks, 128, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 96><<< num_blocks, 96, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 32><<< num_blocks, 32, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //gemm_device<T, 16, 64><<< num_blocks, 64, 0, 0 >>>(m,  n,  k, A,  B,  out, lda, ldb, ldc);
    //back memcpy
    q_ct1.memcpy((T*)(A), (T*)(buff_A), size);
    q_ct1.memcpy((T*)(B), (T*)(buff_B), size);
    q_ct1.memcpy((T*)(out), (T*)(buff_out), size);
  
}

template <typename T> void gemm_4bit_inference(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, T * out,  int lda, int ldb, int ldc, int blocksize)
{

	int num_blocks = (m+31)/32;

	//cout << num_blocks << endl;
	//cout << lda << endl;
	//cout << ldb << endl;
	//cout << ldc << endl;

	//cout << m << endl;
	//cout << n << endl;
	//cout << k << endl;
 
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  int size = NUM_BLOCK;
  *((T **)&buff_A) = sycl::malloc_device(size, A, ctx);
  q_ct1.memcpy((T*)(buff_A), (T*)(A), size);
  *(( unsigned char**)&buff_B) = sycl::malloc_device(size, B, ctx);
  q_ct1.memcpy((unsigned char*)(buff_B), (unsigned char*)(B), size);
  *((T **)&buff_out) = sycl::malloc_device(size, out, ctx);
  q_ct1.memcpy((T*)(buff_out), (T*)(out), size);
 
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
    dpct::get_in_order_queue().submit(
      [&](sycl::handler &cgh) {
        /*
        DPCT1101:314: '8*16 + (16*(batch_size_warps-1))' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
        */
        
        /*
        DPCT1101:315: '2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))' expression was replaced with a value. Modify the code to use the original expression, provided in comments, if it is correct.
        */
        
        //__shared__ vars
        sycl::local_accessor<T, 1> smem_A_acc_ct1(sycl::range<1>(176/*8*16 + (16*(batch_size_warps-1))*/), cgh);
        sycl::local_accessor<T, 1> smem_B_acc_ct1(sycl::range<1>(4192/*2*batch_size_warps*16*32 + (2*16*(batch_size_warps-1))*/), cgh);
        sycl::local_accessor<T, 1> smem_C_acc_ct1(sycl::range<1>(8*32), cgh);

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 96), sycl::range<3>(1, 1, 96)), 
          [=](sycl::nd_item<3> item_ct1) {
            kgemm_4bit_inference<T, 96>(m, n, k, buff_A, buff_B, absmax, buff_out, lda, ldb, ldc, blocksize, item_ct1, smem_A_acc_ct1.get_pointer(), smem_B_acc_ct1.get_pointer(), smem_C_acc_ct1.get_pointer());
          });
      });
  }
  //kgemm_4bit_inference<T, 256><<< num_blocks, 256, 0, 0 >>>(m,  n,  k, A,  B, absmax, out, lda, ldb, ldc, blocksize);
  //kgemm_4bit_inference<T, 160><<< num_blocks, 160, 0, 0 >>>(m,  n,  k, A,  B, absmax, out, lda, ldb, ldc, blocksize);
  //kgemm_4bit_inference<T, 32><<< num_blocks, 32, 0, 0 >>>(m,  n,  k, A,  B, absmax, out, lda, ldb, ldc, blocksize);
  //back memcpy
  q_ct1.memcpy((T*)(A), (T*)(buff_A), size);
  q_ct1.memcpy((unsigned char*)(B), (unsigned char*)(buff_B), size);
  q_ct1.memcpy((T*)(out), (T*)(buff_out), size);
  
}

template <typename T, int BITS> void gemm_4bit_inference_naive(int m, int n, int k, T * A,  unsigned char* B,  float *absmax, float *datatype, T * out,  int lda, int ldb, int ldc, int blocksize)
{

	int num_blocks = (m+3)/4;
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::context ctx = q_ct1.get_context();
    
  int size = NUM_BLOCK;
  *((T **)&buff_A) = sycl::malloc_device(size, A, ctx);
  q_ct1.memcpy((T*)(buff_A), (T*)(A), size);
  *(( unsigned char**)&buff_B) = sycl::malloc_device(size, B, ctx);
  q_ct1.memcpy((unsigned char*)(buff_B), (unsigned char*)(B), size);
  *((T **)&buff_out) = sycl::malloc_device(size, out, ctx);
  q_ct1.memcpy((T*)(buff_out), (T*)(out), size);

  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp16});
    dpct::get_in_order_queue().submit(
      [&](sycl::handler &cgh) {
        sycl::local_accessor<T, 1> quant_map_acc_ct1(sycl::range<1>(16), cgh);

        cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)), 
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            kgemm_4bit_inference_naive<T, 128, BITS>(m, n, k, buff_A, buff_B, absmax, datatype, buff_out, lda, ldb, ldc, blocksize, item_ct1, quant_map_acc_ct1.get_pointer());
          });
      });
  }
  /*
  DPCT1010:291: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);
  q_ct1.memcpy((T*)(A), (T*)(buff_A), size);
  q_ct1.memcpy((unsigned char*)(B), (unsigned char*)(buff_B), size);
  q_ct1.memcpy((T*)(out), (T*)(buff_out), size);
    
}

template <typename T, int FUNC> void func(T *A, T *B, T value, long n)
{
  int threads = 512;
  int blocks = n/threads;
  blocks = n % threads == 0 ? blocks : blocks + 1;
  blocks = blocks > 65535 ? 65535 : blocks;
  /*
  DPCT1049:71: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, 512), sycl::range<3>(1, 1, 512)), 
    [=](sycl::nd_item<3> item_ct1) {
      kfunc<T, FUNC>(A, B, value, n, item_ct1);
    });
  /*
  DPCT1010:292: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  //CUDA_CHECK_RETURN(0);
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void func<float, FILL>(float *A, float *B, float value, long n);
template void func<unsigned char, FILL>(unsigned char *A, unsigned char *B, unsigned char value, long n);
template void func<float, ARANGE>(float *A, float *B, float value, long n);
template void func<float, _MUL>(float *A, float *B, float value, long n);

template void gemm_4bit_inference<half>(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, half * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<half, 16>(int m, int n, int k, half * A,  unsigned char* B,  float *absmax, float *datatype, half * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<__nv_bfloat16, 16>(int m, int n, int k, __nv_bfloat16 * A,  unsigned char* B,  float *absmax, float *datatype, __nv_bfloat16 * out,  int lda, int ldb, int ldc, int blocksize);
template void gemm_4bit_inference_naive<float, 32>(int m, int n, int k, float * A,  unsigned char* B,  float *absmax, float *datatype, float * out,  int lda, int ldb, int ldc, int blocksize);

//template void gemm_host<float>(int m, int n, int k, float * A,  float* B,  float * out,  int lda, int ldb, int ldc, int bits);
template void gemm_host<half>(int m, int n, int k, half * A,  half* B,  half * out,  int lda, int ldb, int ldc, int bits);
template void extractOutliers<COL_TURING>(char * A, int *idx, char *out, int idx_size, int rows, int cols);
template void extractOutliers<COL_AMPERE>(char * A, int *idx, char *out, int idx_size, int rows, int cols);

template void spmm_coo_very_sparse_naive<half, 16>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, half *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);
template void spmm_coo_very_sparse_naive<signed char, 8>(int *max_count, int *max_idx, int *offset_rowidx, int *rowidx, int *colidx, half *values, signed char *B, half *out, float *dequant_stats, int nnz_rows, int nnz, int rowsA, int rowsB, int colsB);

template int igemmlt<COL_TURING, 32, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_TURING, 8, 1>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 32, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);
template int igemmlt<COL_AMPERE, 8, 1>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc);

template void transformRowToFormat<COL32, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL32, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_TURING, 1>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 0>(char * A, char *out, int rows, int cols);
template void transformRowToFormat<COL_AMPERE, 1>(char * A, char *out, int rows, int cols);

template void estimateQuantiles(half *A, float *code, float offset, int n);
template void estimateQuantiles(float *A, float *code, float offset, int n);

template void quantizeBlockwise<half, 1, General8bit>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 0, General8bit>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 0, FP4>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<half, 0, NF4>(float * code, half *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 1, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, General8bit>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, FP4>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<float, 0, NF4>(float * code, float *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 1, General8bit>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 0, General8bit>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 0, FP4>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);
template void quantizeBlockwise<__nv_bfloat16, 0, NF4>(float * code, __nv_bfloat16 *A, float *absmax, unsigned char *out, float* rand, int rand_offset, int blocksize, const int n);

template void dequantizeBlockwise<float, General8bit>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<float, FP4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<float, NF4>(float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n);
template void dequantizeBlockwise<half, General8bit>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<half, FP4>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<half, NF4>(float *code, unsigned char *A, float *absmax, half *out, int blocksize, const int n);
template void dequantizeBlockwise<__nv_bfloat16, General8bit>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n);
template void dequantizeBlockwise<__nv_bfloat16, FP4>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n);
template void dequantizeBlockwise<__nv_bfloat16, NF4>(float *code, unsigned char *A, float *absmax, __nv_bfloat16 *out, int blocksize, const int n);

#define MAKE_optimizer32bit(name, gtype) \
template void optimizer32bit<gtype, name>(gtype* g, gtype* p, \
                float* state1, float* state2, float* unorm, float max_unorm, float param_norm, \
                const float beta1, const float beta2, const float eps, const float weight_decay, \
                const int step, const float lr, const float gnorm_scale, const bool skip_zeros, const int n);

MAKE_optimizer32bit(ADAM, half)
MAKE_optimizer32bit(ADAM, float)
MAKE_optimizer32bit(ADAM, __nv_bfloat16)
MAKE_optimizer32bit(MOMENTUM, half)
MAKE_optimizer32bit(MOMENTUM, float)
MAKE_optimizer32bit(RMSPROP, half)
MAKE_optimizer32bit(RMSPROP, float)
MAKE_optimizer32bit(LION, half)
MAKE_optimizer32bit(LION, float)
MAKE_optimizer32bit(LION, __nv_bfloat16)
MAKE_optimizer32bit(ADAGRAD, half)
MAKE_optimizer32bit(ADAGRAD, float)

#define MAKE_optimizerStatic8bit(name, gtype) \
template void optimizerStatic8bit<gtype, name>(gtype* p, gtype* g, unsigned char* state1, unsigned char* state2, \
                float *unorm, float max_unorm, float param_norm, \
                float beta1, float beta2, \
                float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, \
                float* max1, float* max2, float* new_max1, float* new_max2, \
                float weight_decay, \
                const float gnorm_scale, int n); \

MAKE_optimizerStatic8bit(ADAM, half)
MAKE_optimizerStatic8bit(ADAM, float)
MAKE_optimizerStatic8bit(MOMENTUM, half)
MAKE_optimizerStatic8bit(MOMENTUM, float)
MAKE_optimizerStatic8bit(RMSPROP, half)
MAKE_optimizerStatic8bit(RMSPROP, float)
MAKE_optimizerStatic8bit(LION, half)
MAKE_optimizerStatic8bit(LION, float)

#define MAKE_optimizerStatic8bitBlockwise(gtype, optim_name) \
template void optimizerStatic8bitBlockwise<gtype, optim_name>(gtype* p, gtype* g, \
                unsigned char* state1, unsigned char* state2, float beta1, float beta2, float eps, int step, float lr,  \
                float* quantiles1, float* quantiles2, float* absmax1, float* absmax2, float weight_decay, const float gnorm_scale, bool skip_zeros, int n); \

MAKE_optimizerStatic8bitBlockwise(half, ADAM);
MAKE_optimizerStatic8bitBlockwise(float, ADAM);
MAKE_optimizerStatic8bitBlockwise(half, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(float, MOMENTUM);
MAKE_optimizerStatic8bitBlockwise(half, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(float, RMSPROP);
MAKE_optimizerStatic8bitBlockwise(half, LION);
MAKE_optimizerStatic8bitBlockwise(float, LION);
MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, LION);
MAKE_optimizerStatic8bitBlockwise(half, ADAGRAD);
MAKE_optimizerStatic8bitBlockwise(float, ADAGRAD);

template void percentileClipping(float * g, float *gnorm_vec, int step, const int n);
template void percentileClipping(half * g, float *gnorm_vec, int step, const int n);

MAKE_optimizerStatic8bitBlockwise(__nv_bfloat16, ADAM);
