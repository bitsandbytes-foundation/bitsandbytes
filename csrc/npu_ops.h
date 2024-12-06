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

}
#endif
