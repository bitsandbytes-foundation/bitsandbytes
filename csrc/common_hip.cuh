#pragma once

#ifdef __GFX9__
#define BNB_WARP_SIZE 64
#else 
#define BNB_WARP_SIZE 32
#endif

// These are set based on current BNB support for CDNA 2 & RDNA 3. Update as needed for future archs
#define BNB_MAX_THREADS_PER_CU 2048
#define BNB_BF16_AVAILABLE true
