// compat_device.cuh — Device-only portability layer (CUB, reduction ops, MMA)
//
// Include this from .cu kernel files only (compiled by nvcc/hipcc).
// Do NOT include from .cpp files — use compat.cuh instead for host-safe types.

#pragma once

#include "compat.cuh"

// ============================================================================
// CUB / hipCUB — namespace alias
//
// Usage: bnb_cub::BlockLoad<...>, bnb_cub::BlockReduce<...>, etc.
// This single alias eliminates ~90% of the cub:: vs hipcub:: differences.
// ============================================================================

#if BNB_HIP

#include <hipcub/hipcub.hpp>
namespace bnb_cub = hipcub;

#else // CUDA

#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <math_constants.h>
#include <mma.h>
namespace bnb_cub = cub;

#endif

// ============================================================================
// Reduction operators — CUB's Max()/Sum() API differs across versions
// ============================================================================

#if BNB_HIP

#define BNB_MAX_OP hipcub::Max()
#define BNB_SUM_OP hipcub::Sum()

#else // CUDA

// CCCL 2.8.2+ moved to cuda::maximum<>{}, older versions use cub::Max()
#if defined(CCCL_VERSION) && CCCL_VERSION >= 2008002
#include <cuda/std/functional>
#define BNB_MAX_OP                                                                                                     \
    cuda::maximum<> {}
#else
#define BNB_MAX_OP cub::Max()
#endif
#define BNB_SUM_OP cub::Sum()

#endif
