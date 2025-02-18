#ifndef NPU_OPS_H
#define NPU_OPS_H
#include <cstdint>

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

// align num to multiples of rnd, round up
#define ALIGNUP(num, rnd) (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)))
// align num to multiples of rnd, round down
#define ALIGNDOWN(num, rnd) ((((rnd) == 0) || ((num) < (rnd))) ? 0 : ((num) / (rnd) * (rnd)))
// div and Round Up
#define DIVCEIL(num, div) (((div) == 0) ? 0 : (((num) + (div)-1) / (div)))

const uint32_t UB_RESERVED_BUFF = 8 * 1024;
const uint32_t MAX_CORE_NUMBER = 64;
const uint32_t L2_CACHE_LINE_SIZE = 512;
const uint32_t UB_MIN_BLOCK_SIZE = 32;
const uint32_t MAX_BLOCK_COUNT = 4095;
const uint32_t MAX_BLOCK_LEN = 65535 * 32;
const uint32_t MAX_UINT32 = 4294967295;
const uint16_t DISCONTINE_COPY_MAX_BLOCKCNT = 4095;
const uint16_t DISCONTINE_COPY_MAX_BLOCKLEN = 65535;
const uint16_t DISCONTINE_COPY_MAX_STRIDE = 65535;

// row_col_quant
struct OutlierTilingParam {
    uint64_t colLen = 0;
    uint32_t loopNum = 0;
    uint32_t tileCol = 0;
    uint16_t isTailExist = 0;
    uint32_t tailCol = 0;
};

struct RowColQuantTilingData {
    uint32_t usedCoreNum = 0; // number of vector core. Don't move, must be in the first
    uint32_t is32BAligned = 1;
    uint32_t isDoubleBuffer = 0;

    uint64_t rowLen = 1;      // row length for split vector, Unit:element
    uint64_t colLen = 1;      // column length for split vector, Unit:element
    uint32_t baseRowLen = 2;  // for one tile in one core, Unit:element
    uint32_t baseColLen = 16; // for one tile in one core, Unit:element

    float threshold = 0.0f;
    uint32_t outliersNum = 0;
    uint32_t isColQuant = 0;
    uint32_t isOutlierIndex = 0;

    uint32_t usedCoreNumForOutlier = 0;
    uint32_t baseCoreNumForOutlier = 0;
    OutlierTilingParam baseCoreParam;
    OutlierTilingParam tailCoreParam;
};

// row_col_stats
struct RowColStatsTiling {
    uint64_t start_offs[MAX_CORE_NUMBER] = {0};
    uint32_t core_rows[MAX_CORE_NUMBER] = {0};
    uint32_t core_cols[MAX_CORE_NUMBER] = {0};
    uint32_t max_elements_per_ub = 0;
    uint32_t used_cores = 0;  // number of vector core. Don't move, must be in the first
    uint32_t buffer_num = 1;
    uint32_t M = 0;
    uint32_t K = 0;
    uint32_t ub_sizes = 0;
    float threshold = 0;
    bool is_outlier_index = true;
    bool use_gather_mask = true;
};

struct RowColStatsTilingKernel {
    uint32_t tile_lines = 0;
    uint32_t tail_tile_lines = 0;
    uint32_t tile_num = 0;
    uint32_t last_tile_idx = 0;
    uint32_t M = 0;
    uint32_t K = 0;
    uint32_t core_k = 0;
    uint32_t core_m = 0;
    uint32_t align_k = 0;
    uint32_t align_m = 0;
    uint32_t align_K = 0;
    uint32_t ub_sizes = 0;
    float threshold = 0;
    bool is_outlier_index = true;
    bool use_gather_mask = true;
    uint64_t start_off = 0;
};

struct BlockwiseNf4TilingData {
    uint32_t coreNum;
    uint32_t blocksize;
    uint32_t numel;
    uint32_t singleCoreNumel;
    uint32_t singleCoreNumelTail;
    uint32_t ubSize;
};

extern "C" {

void dequantizeBlockwiseNf4(uint8_t *A, uint8_t *absmax, uint8_t *out, uint32_t blocksize, uint32_t n, void* stream, const uint32_t type_mode);

void rowColQuant(uint8_t *A, uint8_t *rowStats, uint8_t *colStats, uint8_t *outRowNormed, uint8_t *outColNormed,
                 uint8_t *outliersRowIdx, uint8_t *outliersColIdx, uint8_t *outliersValue, uint32_t outliersNum,
                 float threshold, int rows, int cols, void* stream);

void rowColStats(uint8_t *A, uint8_t *rowStats, uint8_t *colStats, uint8_t *outliersNum, float threshold, int rows, int cols, void *stream);

}
#endif
