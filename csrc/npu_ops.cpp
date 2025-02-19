#include <iostream>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "npu_ops.h"

#include "aclrtlaunch_dequantize_blockwise_fp32_nf4.h"
#include "aclrtlaunch_dequantize_blockwise_fp16_nf4.h"


extern "C" {

int32_t get_dequantize_blockwise_nf4_tiling(uint32_t blocksize, uint32_t n, BlockwiseNf4TilingData *tiling) {
    tiling->ubSize = 196 * 1024;
    uint32_t coreNum = 40;
    uint32_t totalPkgNum = (n + blocksize - 1) / blocksize;
    uint32_t singleCorePkgNum = (totalPkgNum + coreNum - 1) / coreNum;
    coreNum = (totalPkgNum + singleCorePkgNum - 1) / singleCorePkgNum;
    uint32_t singleCoreNumel = singleCorePkgNum * blocksize;
    uint32_t singleCoreNumelTail = n % singleCoreNumel;
    if (singleCoreNumelTail == 0) {
        singleCoreNumelTail = singleCoreNumel;
    }
    tiling->coreNum = coreNum;
    tiling->blocksize = blocksize;
    tiling->numel = n;
    tiling->singleCoreNumel = singleCoreNumel;
    tiling->singleCoreNumelTail = singleCoreNumelTail;
    return 0;
}

void dequantizeBlockwiseNf4(uint8_t *A, uint8_t *absmax, uint8_t *out, uint32_t blocksize, uint32_t n, void* stream, const uint32_t type_mode) {
    uint32_t blockDim = 40;
    size_t tilingSize = sizeof(struct BlockwiseNf4TilingData);
    BlockwiseNf4TilingData *tilingHost;
    tilingHost = (struct BlockwiseNf4TilingData *)malloc(tilingSize);
    uint32_t error = get_dequantize_blockwise_nf4_tiling(blocksize, n, tilingHost);
    if (error != 0) {
        printf("[!] error\n");
    }
    uint8_t *tilingDevice = nullptr;
    aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpyAsync((void *)tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (type_mode == 1) {
        ACLRT_LAUNCH_KERNEL(dequantize_blockwise_fp32_nf4)(blockDim, stream, A, absmax, out, tilingDevice);
    } else if (type_mode == 2) {
        ACLRT_LAUNCH_KERNEL(dequantize_blockwise_fp16_nf4)(blockDim, stream, A, absmax, out, tilingDevice);
    }
    aclrtFree(tilingDevice);
}

}
