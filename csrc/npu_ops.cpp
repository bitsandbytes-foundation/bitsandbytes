#include <iostream>
#include <math.h>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "npu_ops.h"

#include "aclrtlaunch_dequantize_blockwise_fp32_nf4.h"
#include "aclrtlaunch_dequantize_blockwise_fp16_nf4.h"
#include "aclrtlaunch_row_col_stats.h"
#include "aclrtlaunch_row_col_quant.h"


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
        printf("An error occurred.\n");
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

namespace get_row_col_quant_tiling {
    const uint32_t DEFAULT_BUFFER_NUM = 1;
    const uint32_t TQUE_ROW_COL_FP16_NUM = 1;
    const uint32_t TQUE_ROW_COL_FLOAT_NUM = 2;
    const uint32_t TQUE_ROW_COL_INT8_NUM = 2;
    const uint32_t TQUE_ROW_COL_INT32_NUM = 1;
    const uint32_t TBUF_ROW_COL_FLOAT_X_NUM = 2;
    const uint32_t TBUF_ROW_COL_FLOAT_THR_REPEAT_NUM = 1;
    const uint32_t TBUF_ROW_COL_BITMAP_NUM = 1;
    const uint32_t TBUF_ROW_COL_NORM_CAST_NUM = 2;
    const uint32_t TBUF_ROW_COL_ROW_SELECT_NUM = 1;
    const uint32_t TBUF_ROW_COL_REPEAT_127_NUM = 1;
    const uint32_t DTYPE_FLOAT16_SIZE = 2;

    struct RowColQuantCalculator : public RowColQuantTilingData {
    public:
        bool CalcTiling(uint32_t totalCore, uint64_t ubSize, int32_t dtype);
        bool SetTotalShape(int rows, int cols);
        bool SetInputAttr(uint32_t outliersNum, float threshold, bool isColQuant, bool isOutlierIndex);

    private:
        inline bool CalcTileColMax(uint64_t ubSize, uint16_t bufferNum);
        inline bool CalcOutlierTiling(uint32_t totalCore, uint64_t ubSize);
        inline uint32_t CalcTileRowMaxByCol(uint64_t ubSize, uint16_t bufferNum, uint64_t tileCol);
        inline void SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_);
        inline uint32_t getBaseColLenUpBound();
        inline uint32_t getBaseRowLenUpBound();
        inline bool MustBeSingleBaseRowLen(uint32_t baseColLen_);
        inline bool isInvalidBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_);
        inline bool CalcOptBaseShape(uint64_t ubSize);

        uint32_t tileColMax = 0;
        uint32_t inputDTypeLen = 2;
        // Indicates the minimum processing data unit of the UB. Unit:element.
        // Formula: 32B/sizeof(DType). For example, if Dtype is BF16, ubMinBlockLen = 32/2 = 16
        uint32_t ubMinBlockLen = 0;
        // Length of the L2 cache line. Unit:element.
        // Formula: 512B/sizeof(DType). For example, if the Dtype is BF16, cacheLineLen = 512/2 = 256
        uint32_t cacheLineLen = 0;
        // baseColLen aligned package Len. element:Unit. 512-byte alignment or 32-byte alignment
        uint32_t alignPackLen = 0;
        // Maximum amount of data that can be transferred by an operator UB at a time. Unit:element
        uint32_t maxTileLen = 0;
        uint32_t optBaseRowLen = 0;
        uint32_t optBaseColLen = 0;
        uint64_t optTotalTileNum = 0;
        uint64_t optBaseSize = 0;
        uint64_t optBaseTileNum = 0;
    };

    inline bool GetLengthByType(int32_t dtype, uint32_t& dsize)
    {
        dsize = sizeof(int16_t);
        return true;
    }

    inline bool RowColQuantCalculator::SetTotalShape(int rows, int cols)
    {
        rowLen = rows;
        colLen = cols;
        return true;
    }

    inline bool RowColQuantCalculator::CalcTileColMax(uint64_t ubSize, uint16_t bufferNum)
    {
        auto base = bufferNum * (sizeof(int16_t) + sizeof(int8_t)) + sizeof(float) * 3 + sizeof(int8_t);
        if (isColQuant == 1) {
            base += bufferNum * (sizeof(float) + sizeof(int8_t)) + sizeof(float);
        }
        if (isOutlierIndex == 1) {
            base += bufferNum * (sizeof(int32_t) * 2 + sizeof(int16_t));
        } else {
            base += bufferNum * (sizeof(float) + sizeof(int32_t));
        }

        tileColMax = ALIGNDOWN((ubSize - 32) / base, L2_CACHE_LINE_SIZE);
        return true;
    }

    inline uint32_t RowColQuantCalculator::CalcTileRowMaxByCol(uint64_t ubSize, uint16_t bufferNum, uint64_t tileCol)
    {
        auto base = (bufferNum * (sizeof(int16_t) + sizeof(int8_t)) + sizeof(float) * 3 + sizeof(int8_t)) * tileCol + sizeof(float);
        if (isColQuant == 1) {
            base += bufferNum * sizeof(int8_t) * tileCol;
            ubSize -= (bufferNum * sizeof(float) + sizeof(float)) * tileCol;
        }
        if (isOutlierIndex == 1) {
            base += bufferNum * (sizeof(int32_t) * 2 + sizeof(int16_t)) * tileCol;
        } else {
            ubSize -= bufferNum * (sizeof(float) + sizeof(int32_t)) * tileCol;
        }

        return (ubSize - 32) / base;
    }

    inline void RowColQuantCalculator::SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_)
    {
        uint64_t totalTileNum = DIVCEIL(rowLen, baseRowLen_) * DIVCEIL(colLen, baseColLen_);
        uint64_t baseSize = baseRowLen_ * baseColLen_;
        uint64_t baseTileNum = (rowLen / baseRowLen_) * (colLen / baseColLen_);

        optBaseRowLen = baseRowLen_;
        optBaseColLen = baseColLen_;
        optTotalTileNum = totalTileNum;
        optBaseSize = baseSize;
        optBaseTileNum = baseTileNum;
    }

    inline uint32_t RowColQuantCalculator::getBaseColLenUpBound()
    {
        uint32_t upBound = std::min(colLen, (uint64_t)tileColMax);
        if (is32BAligned == 1) {
            upBound = std::min(upBound, (uint32_t)DISCONTINE_COPY_MAX_BLOCKLEN);
        } else {
            upBound = std::min(upBound, (uint32_t)DISCONTINE_COPY_MAX_BLOCKLEN / inputDTypeLen);
        }

        return upBound;
    }

    inline uint32_t RowColQuantCalculator::getBaseRowLenUpBound()
    {
        return std::min(rowLen, (uint64_t)DISCONTINE_COPY_MAX_BLOCKCNT);
    }

    inline bool RowColQuantCalculator::MustBeSingleBaseRowLen(uint32_t baseColLen_)
    {
        if (is32BAligned == 1) {
            return ((colLen * 2 - baseColLen_) > (DISCONTINE_COPY_MAX_STRIDE * ubMinBlockLen));
        }

        return (((colLen * 2 - baseColLen_) * inputDTypeLen) > DISCONTINE_COPY_MAX_STRIDE);
    }

    inline bool RowColQuantCalculator::isInvalidBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_)
    {
        return ((baseRowLen_ < 1) || (baseRowLen_ > 1 && MustBeSingleBaseRowLen(baseColLen_)));
    }

    inline bool RowColQuantCalculator::CalcOptBaseShape(uint64_t ubSize)
    {
        uint32_t baseColLen_ = getBaseColLenUpBound();
        if (MustBeSingleBaseRowLen(baseColLen_)) {
            SaveOptBaseShape(1, baseColLen_);
            return true;
        }

        uint32_t baseRowLen_ = std::min(CalcTileRowMaxByCol(ubSize, DEFAULT_BUFFER_NUM, baseColLen_), getBaseRowLenUpBound());
        if (isInvalidBaseShape(baseRowLen_, baseColLen_)) {
            return (optTotalTileNum > 0);
        }
        SaveOptBaseShape(baseRowLen_, baseColLen_);

        return true;
    }

    inline bool RowColQuantCalculator::CalcOutlierTiling(uint32_t totalCore, uint64_t ubSize) {
        uint32_t MIN_BLOCK_ALIGN_LEN = UB_MIN_BLOCK_SIZE / sizeof(float);
        uint32_t baseCoreCalcLens = ALIGNUP(DIVCEIL(colLen, totalCore), MIN_BLOCK_ALIGN_LEN);
        baseCoreNumForOutlier = colLen / baseCoreCalcLens;
        usedCoreNumForOutlier = baseCoreNumForOutlier;
        baseCoreParam.colLen = baseCoreCalcLens;
        if (baseCoreCalcLens >= tileColMax) {
            baseCoreParam.loopNum = (uint32_t)baseCoreCalcLens / (uint32_t)tileColMax;
            baseCoreParam.tileCol = tileColMax;
        }
        if (baseCoreCalcLens % tileColMax != 0) {
            baseCoreParam.isTailExist = 1;
            baseCoreParam.tailCol = baseCoreCalcLens % tileColMax;
        }

        if (colLen % baseCoreCalcLens != 0) {
            usedCoreNumForOutlier += 1;
            tailCoreParam.colLen = ALIGNUP(colLen % baseCoreCalcLens, MIN_BLOCK_ALIGN_LEN);
            if (tailCoreParam.colLen >= tileColMax) {
                tailCoreParam.loopNum = (uint32_t)tailCoreParam.colLen / (uint32_t)tileColMax;
                tailCoreParam.tileCol = tileColMax;
            }
            if (tailCoreParam.colLen % tileColMax != 0) {
                tailCoreParam.isTailExist = 1;
                tailCoreParam.tailCol = tailCoreParam.colLen % tileColMax;
            }
        }

        return true;
    }

    bool RowColQuantCalculator::CalcTiling(uint32_t totalCore, uint64_t ubSize, int32_t dtype)
    {
        if (!GetLengthByType(dtype, inputDTypeLen)) {
            printf("Unsupported input data type %d\n", dtype);
            return false;
        }
        ubMinBlockLen = UB_MIN_BLOCK_SIZE / inputDTypeLen;  // min block size
        cacheLineLen = L2_CACHE_LINE_SIZE / inputDTypeLen;  // bandwidth max efficiency
        alignPackLen = cacheLineLen;

        ubSize -= UB_RESERVED_BUFF;
        if (!CalcTileColMax(ubSize, DEFAULT_BUFFER_NUM)) {
            return false;
        }

        is32BAligned = colLen % ubMinBlockLen == 0;

        if (!CalcOptBaseShape(ubSize)) {
            return false;
        }
        baseRowLen = optBaseRowLen;
        baseColLen = optBaseColLen;
        usedCoreNum = std::min(optTotalTileNum, (uint64_t)totalCore);
        usedCoreNumForOutlier = usedCoreNum;
        if (isOutlierIndex == 0) {
            CalcOutlierTiling(totalCore, ubSize);
        }
        return true;
    }

    bool RowColQuantCalculator::SetInputAttr(uint32_t outliers_num, float in_threshold, bool is_col_quant, bool is_outlier_index)
    {
        outliersNum = outliers_num;
        threshold = in_threshold;
        isColQuant = (is_col_quant ? 1 : 0);
        isOutlierIndex = (is_outlier_index ? 1 : 0);
        return true;
    }

    uint32_t TilingForRowColQuant(uint32_t outliers_num, float in_threshold, bool is_col_quant, bool is_outlier_index,
                                  int rows, int cols, uint32_t totalCore,
                                  get_row_col_quant_tiling::RowColQuantCalculator *tilingCalc)
    {
        uint64_t ubSize = 192 * 1024;
        if (totalCore < 0 || totalCore >= MAX_CORE_NUMBER || ubSize <= UB_RESERVED_BUFF) {
            printf("Compile Info is invalid, coreNum:%u, ubSize:%lu\n", totalCore, ubSize);
            return 1;
        }

        ubSize -= UB_RESERVED_BUFF;

        if (!tilingCalc->SetInputAttr(outliers_num, in_threshold, is_col_quant, is_outlier_index)) {
            printf("Parse input attrs failed\n");
            return 1;
        }

        if (!tilingCalc->SetTotalShape(rows, cols) || !tilingCalc->CalcTiling(totalCore, ubSize, DTYPE_FLOAT16_SIZE)) {
            return 1;
        }
        return 0;
    }
}

void rowColQuant(uint8_t *A, uint8_t *rowStats, uint8_t *colStats, uint8_t *outRowNormed, uint8_t *outColNormed,
                 uint8_t *outliersRowIdx, uint8_t *outliersColIdx, uint8_t *outliersValue, uint32_t outliersNum,
                 float threshold, int rows, int cols, void* stream) {
    uint32_t blockDim = 40;
    bool isColQuant = false;
    bool isOutlierIndex = false;
    size_t tilingSize = sizeof(struct get_row_col_quant_tiling::RowColQuantCalculator);
    get_row_col_quant_tiling::RowColQuantCalculator tilingHost;
    uint32_t error = get_row_col_quant_tiling::TilingForRowColQuant(outliersNum, threshold, isColQuant, isOutlierIndex,
                                                                    rows, cols, blockDim, &tilingHost);
    if (error != 0) {
        printf("An error occurred.\n");
    }
    uint8_t *tilingDevice = nullptr;
    aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpyAsync((void *)tilingDevice, tilingSize, &tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    ACLRT_LAUNCH_KERNEL(row_col_quant)(tilingHost.usedCoreNumForOutlier, stream, A, rowStats, colStats, outRowNormed,
                                       outColNormed, outliersRowIdx, outliersColIdx, outliersValue, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    aclrtFree(tilingDevice);
}


namespace get_row_col_stats_tiling {
    const uint32_t PACK_SIZE = 512;      // pack unit in cache 512B
    const uint32_t ALIGN_SIZE = 32;      // align unit in cache 32B
    const uint32_t DEFAULT_BUFFER_NUM = 1;

    inline uint32_t RoundUpToN(uint32_t m, uint32_t n)
    {
        return (m / n) * n;
    }

    inline uint32_t GetLength(int32_t dtype, uint32_t &dsize)
    {
        switch (dtype) {
            case 0:
            case 1:
                dsize = sizeof(int16_t);
                return true;
            default:
                return false;
        }
    }

    struct RowColStatsTilingCalculator : public RowColStatsTiling
    {
    public:
        bool CalcTiling(uint32_t m, uint32_t k, uint32_t core_num, uint64_t ub_size, uint32_t dtype_len, int32_t dtype,
                        float th, bool is_outlier_idx)
        {
            M = m;
            K = k;
            threshold = th;
            is_outlier_index = is_outlier_idx;

            uint64_t element_num = m * k;
            // align to 32B by hardware
            uint32_t align_num = ALIGN_SIZE / dtype_len;
            // align to L2 cacheline 512B (for bandwidth max efficiency)
            uint32_t pack_align_num = PACK_SIZE / dtype_len;
            buffer_num = DEFAULT_BUFFER_NUM;

            ub_sizes = ub_size;
            max_elements_per_ub = GetMaxDatasPerUB(ub_size, dtype_len, 0, 0, buffer_num);
            uint64_t align_elements_per_ub = (max_elements_per_ub / pack_align_num) * pack_align_num;

            if (element_num <= align_elements_per_ub) {
                used_cores = 1;
            } else if (element_num >= align_elements_per_ub * core_num) {
                used_cores = core_num;
            } else {
                used_cores = (element_num + align_elements_per_ub - 1) / align_elements_per_ub;
            }

            if (K <= 4096 /*align_elements_per_ub / 4*/) {
                if (!TilingForRow(core_num, align_num)) {
                    printf("CalcTiling failed for TilingForRow \n");
                    return false;
                }
            } else if (M <= 4096) {
                if (!TilingForCol(core_num, align_num)) {
                    printf("CalcTiling failed for TilingForCol \n");
                    return false;
                }
            } else {
                if ((used_cores != core_num) || (!TilingForBlock(core_num, align_num))) {
                    printf("CalcTiling failed for TilingForBlock \n");
                    return false;
                }
            }
            return true;
        }

        bool TilingForRow(uint32_t core_num, uint32_t align_num) {
            uint32_t core_k = K;

            uint32_t align_lines = align_num;
            uint64_t min_core_lines = (M / (used_cores * align_lines)) * align_lines;
            std::fill(core_rows, core_rows + core_num, min_core_lines);
            std::fill(core_cols, core_cols + core_num, core_k);
            uint64_t left_lines = M - min_core_lines * used_cores;
            align_lines = (min_core_lines == 0) ? 1 : align_lines;
            uint32_t index = 0;
            for (uint64_t len = align_lines; len <= left_lines; len += align_lines) {
                core_rows[index % used_cores] += align_lines;
                index++;
            }
            core_rows[used_cores - 1] += M % align_lines;

            uint64_t sum_rows = 0;
            for (uint32_t i = 0; i < used_cores; i++) {
                start_offs[i] = sum_rows * K;
                sum_rows += core_rows[i];
            }
            return true;
        }

        bool TilingForCol(uint32_t core_num, uint32_t align_num) {
            uint32_t core_m = M;

            uint32_t align_lines = align_num;
            uint64_t min_core_lines = (K / (used_cores * align_lines)) * align_lines;
            std::fill(core_rows, core_rows + core_num, core_m);
            std::fill(core_cols, core_cols + core_num, min_core_lines);
            uint64_t left_lines = K - min_core_lines * used_cores;
            uint32_t index = 0;
            for (uint64_t len = align_lines; len <= left_lines; len += align_lines) {
                core_cols[index % used_cores] += align_lines;
                index++;
            }
            core_cols[used_cores - 1] += K % align_lines;

            uint64_t sum_cols = 0;
            for (uint32_t i = 0; i < used_cores; i++) {
                start_offs[i] = sum_cols;
                sum_cols += core_cols[i];
            }
            return true;
        }

        bool TilingForBlock(uint32_t core_num, uint32_t align_num) {
            uint32_t rcore_num = 4;
            uint32_t ccore_num = used_cores / rcore_num;

            uint32_t align_lines = align_num;
            uint64_t min_core_rows = (M / (rcore_num * align_lines)) * align_lines;
            uint64_t min_core_cols = (K / (ccore_num * align_lines)) * align_lines;
            std::fill(core_rows, core_rows + core_num, min_core_rows);
            std::fill(core_cols, core_cols + core_num, min_core_cols);
            uint64_t left_rows = M - min_core_rows * rcore_num;
            uint32_t index = 0;
            for (uint64_t len = align_lines; len <= left_rows; len += align_lines) {
                for (uint32_t i = 0; i < ccore_num; i++) {
                    core_rows[(index % rcore_num) * ccore_num + i] += align_lines;
                }
                index++;
            }
            for (uint32_t i = 0; i < ccore_num; i++) {
                core_rows[(rcore_num - 1) * ccore_num + i] += M % align_lines;
            }

            uint64_t left_cols = K - min_core_cols * ccore_num;
            index = 0;
            for (uint64_t len = align_lines; len <= left_cols; len += align_lines) {
                for (uint32_t i = 0; i < rcore_num; i++) {
                    core_cols[index % ccore_num + i * ccore_num] += align_lines;
                }
                index++;
            }
            for (uint32_t i = 0; i < rcore_num; i++) {
                core_cols[ccore_num - 1 + i * ccore_num] += K % align_lines;
            }

            uint64_t sum_row = 0;
            uint64_t sum_col = 0;
            for (uint32_t i = 0; i < rcore_num; i++) {
                for (uint32_t j = 0; j < ccore_num; j++) {
                    start_offs[i * ccore_num + j] = sum_row * K + sum_col;
                    sum_col += core_cols[j];
                }
                sum_col = 0;
                sum_row += core_rows[i * ccore_num];
            }
            return true;
        }

        uint64_t GetMaxDatasPerUB(uint64_t ub_size, uint32_t dtype_len, uint32_t tile_lines, uint32_t align_k,
                                  uint32_t buffer_num) {
            float a = ((float) 2 * (buffer_num + 1) + (float) 1 / 8);
            float b = (float) 8 * buffer_num;
            float c = (float) buffer_num * 320 + 320 - ub_size;
            float discriminant = b * b - 4 * a * c;
            float result = (2 * b * b - 4 * a * c - 2 * b * sqrt(discriminant)) / (4 * a * a);
            return static_cast<int>(std::floor(result));
        }

        inline uint32_t RoundUp(uint32_t a, uint32_t b)
        {
            return (a + b - 1) / b;
        }

    };

    uint32_t Tiling4RowColStats(int rows, int cols, uint8_t shapeSize, uint32_t core_num, int32_t dtype,
                                float threshold, bool is_col_quant, bool is_outlier_index,
                                get_row_col_stats_tiling::RowColStatsTilingCalculator *tiling_calc)
    {
        uint64_t ub_size = 192 * 1024;
        if (core_num <= 0 || core_num > MAX_CORE_NUMBER || ub_size <= UB_RESERVED_BUFF) {
            printf(" Compile Info is invalid, coreNum:%u, ubSize:%lu", core_num, ub_size);
            return 1;
        }

        uint32_t dtype_len = 0;
        if (!GetLength(dtype, dtype_len)) {
            printf(" Unsupported input data type %d", dtype);
        }
        int32_t dim = shapeSize;
        if (dim > 3 || dim < 2) {
            printf(" Unsupported input data shape dim %d", dim);
            return 1;
        }
        int32_t M = rows;
        int32_t K = cols;

        if (!tiling_calc->CalcTiling(M, K, core_num, ub_size - UB_RESERVED_BUFF, dtype_len, dtype, threshold, is_outlier_index)) {
            return 1;
        }
        return 0;
    }
}


void rowColStats(uint8_t *A, uint8_t *rowStats, uint8_t *colStats, uint8_t *outliersNum, float threshold, int rows, int cols, void *stream) {
    uint32_t blockDim = 40;
    bool is_col_quant = false;
    bool is_outlier_index = false;
    uint32_t dtype = 1;
    uint8_t shapeSize = 2;
    size_t tilingSize = sizeof(struct get_row_col_stats_tiling::RowColStatsTilingCalculator);
    get_row_col_stats_tiling::RowColStatsTilingCalculator *tilingHost;
    tilingHost = (struct get_row_col_stats_tiling::RowColStatsTilingCalculator *)malloc(sizeof(struct get_row_col_stats_tiling::RowColStatsTilingCalculator));
    uint32_t error = get_row_col_stats_tiling::Tiling4RowColStats(rows, cols, shapeSize, blockDim, dtype, threshold, is_col_quant, is_outlier_index, tilingHost);
    if (error != 0) {
        printf("An error occurred.\n");
    }
    uint8_t *tilingDevice = nullptr;
    aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpyAsync((void *)tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    ACLRT_LAUNCH_KERNEL(row_col_stats)(tilingHost->used_cores, stream, A, rowStats, colStats, outliersNum, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    aclrtFree(tilingDevice);
}

}
