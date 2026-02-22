// Minimal test: verify mma.sync.aligned.kind::mxf4nvf4 works on SM_120
// Compile: nvcc -arch=sm_120 -o test_mma_nvfp4 test_mma_nvfp4.cu
// Run: ./test_mma_nvfp4

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// MMA instruction: m16n8k64, E2M1 x E2M1 -> F32, with UE4M3 block scales
// One warp (32 threads) processes:
//   A: 16x64 E2M1 tile (4 regs per thread, 8 nibbles per reg)
//   B: 8x64 E2M1 tile (2 regs per thread, 8 nibbles per reg)
//   SFA: 4 UE4M3 scale factors for A (packed in 1 uint32)
//   SFB: 4 UE4M3 scale factors for B (packed in 1 uint32)
//   D/C: 16x8 F32 tile (4 floats per thread)

__device__ void mma_nvfp4_16x8x64(
    float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, float c0, float c1, float c2, float c3, uint32_t sfa, uint32_t sfb
) {
    uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;

    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13},"
                 "{%14},"
                 "{%15, %16},"
                 "{%17},"
                 "{%18, %19};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa),
                   "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB));
}

__global__ void test_mma_kernel(float* output) {
    // E2M1 code for 1.0: sign=0, exp=1, mant=0 -> 0b0010 = 0x2
    // Pack 8 E2M1 values of 1.0 into one uint32: each nibble = 0x2
    uint32_t a_val = 0x22222222u; // 8 x E2M1(1.0)
    uint32_t b_val = 0x22222222u; // 8 x E2M1(1.0)

    // UE4M3 code for 1.0: exp=7 (bias=7, so 2^0=1), mant=0 -> 0b01110000 = 0x38
    // Wait - UE4M3 is unsigned, 4 exp bits, 3 mantissa bits
    // For value 1.0: 2^(e-7) * (1 + m/8) = 2^0 * 1.0 = 1.0 when e=7, m=0
    // Binary: 0111 000 = 0x38
    // Pack 4 UE4M3 values of 1.0: each byte = 0x38
    uint32_t sfa_val = 0x38383838u; // 4 x UE4M3(1.0)
    uint32_t sfb_val = 0x38383838u; // 4 x UE4M3(1.0)

    // Accumulator starts at 0
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    mma_nvfp4_16x8x64(
        d0, d1, d2, d3, a_val, a_val, a_val, a_val, // A: all 1.0
        b_val, b_val,                               // B: all 1.0
        0.0f, 0.0f, 0.0f, 0.0f,                     // C: accumulator = 0
        sfa_val, sfb_val
    );

    // Each thread writes its 4 output values
    int tid = threadIdx.x;
    output[tid * 4 + 0] = d0;
    output[tid * 4 + 1] = d1;
    output[tid * 4 + 2] = d2;
    output[tid * 4 + 3] = d3;
}

int main() {
    float* d_output;
    float h_output[128]; // 32 threads * 4 values

    cudaMalloc(&d_output, 128 * sizeof(float));
    cudaMemset(d_output, 0, 128 * sizeof(float));

    // Launch 1 warp
    test_mma_kernel<<<1, 32>>>(d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_output, d_output, 128 * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: all A=1.0, all B=1.0, all scales=1.0
    // D[i][j] = sum_k (A[i][k] * SFA[i][k/16]) * (B[j][k] * SFB[j][k/16])
    //         = sum_k=0..63 (1.0 * 1.0) * (1.0 * 1.0) = 64.0
    printf("MMA NVFP4 m16n8k64 test (all ones, scales=1.0):\n");
    printf("Expected: 64.0 for all outputs\n\n");

    int pass = 1;
    for (int t = 0; t < 32; t++) {
        for (int v = 0; v < 4; v++) {
            float val = h_output[t * 4 + v];
            if (val != 64.0f)
                pass = 0;
        }
    }

    // Print first few threads
    for (int t = 0; t < 4; t++) {
        printf(
            "  Thread %2d: d0=%.1f d1=%.1f d2=%.1f d3=%.1f\n", t, h_output[t * 4], h_output[t * 4 + 1],
            h_output[t * 4 + 2], h_output[t * 4 + 3]
        );
    }
    printf("  ...\n");
    for (int t = 28; t < 32; t++) {
        printf(
            "  Thread %2d: d0=%.1f d1=%.1f d2=%.1f d3=%.1f\n", t, h_output[t * 4], h_output[t * 4 + 1],
            h_output[t * 4 + 2], h_output[t * 4 + 3]
        );
    }

    printf("\n%s\n", pass ? "PASS: All outputs are 64.0" : "FAIL: Some outputs incorrect");

    cudaFree(d_output);
    return pass ? 0 : 1;
}
