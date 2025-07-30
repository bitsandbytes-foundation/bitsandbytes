# CUDA K-bit Quantization with CUB Library: Efficient Bit Packing/Unpacking

## Background

K-bit quantization reduces neural network model sizes by representing weights with fewer bits (e.g., 2-8 bits instead of 16/32). The challenge is efficiently packing multiple k-bit values into standard data types (uint8/uint32) and unpacking them during inference. 

**Key Performance Bottleneck**: As observed in production systems, "performance bottlenecks related to bit-unpacking were observed" on data center GPUs. The unpacking operation is instruction-intensive and must be carefully overlapped with memory operations to achieve high throughput.

**Why CUB?**: The CUB library provides highly optimized primitives for memory access patterns that can be combined with custom bit manipulation logic. CUB's block-level and warp-level primitives handle the complex memory coalescing patterns, allowing us to focus on the bit extraction logic.

## Architecture Overview

### Memory Layout for K-bit Quantization

```
Original: [float32] [float32] [float32] [float32] ...
K-bit:    [k-bits][k-bits][k-bits]... packed into uint32 words
          |<------- 32 bits ------->|
```

For k-bit quantization:
- **Elements per word**: `32 / k` values fit in each uint32
- **Packing density**: Reduces memory by factor of `32 / k`
- **Boundary handling**: Values may span word boundaries when k doesn't divide 32 evenly

## CUB-Based Implementation Strategy

### Core Design: CUB + PTX Integration

```cuda
#include <cub/cub.cuh>

template<int K_BITS, int BLOCK_THREADS, int ITEMS_PER_THREAD>
class CUBKBitUnpacker {
public:
    static constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    static constexpr int PACKED_ELEMENTS_PER_WORD = 32 / K_BITS;
    
    struct TempStorage {
        union {
            typename cub::BlockLoad<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD,
                                   cub::BLOCK_LOAD_VECTORIZE>::TempStorage load;
            typename cub::BlockStore<float, BLOCK_THREADS, 
                                    ITEMS_PER_THREAD * PACKED_ELEMENTS_PER_WORD,
                                    cub::BLOCK_STORE_VECTORIZE>::TempStorage store;
        } storage;
    };
    
    __device__ __forceinline__ static void UnpackTile(
        const uint32_t* __restrict__ packed_input,
        float* __restrict__ output,
        const float* __restrict__ scales,
        TempStorage& temp_storage,
        int num_items,
        int tile_offset
    ) {
        // Stage 1: Coalesced load using CUB
        uint32_t packed_items[ITEMS_PER_THREAD];
        cub::BlockLoad<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_VECTORIZE>(temp_storage.storage.load)
            .Load(packed_input + tile_offset, packed_items);
        
        __syncthreads();
        
        // Stage 2: Unpack using PTX with instruction overlap
        float unpacked_items[ITEMS_PER_THREAD * PACKED_ELEMENTS_PER_WORD];
        
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            uint32_t current_packed = packed_items[i];
            
            // Software pipeline: start loading scale for next iteration
            float scale = scales[(tile_offset + threadIdx.x * ITEMS_PER_THREAD + i) 
                               * PACKED_ELEMENTS_PER_WORD / GROUP_SIZE];
            
            // Extract k-bit values using PTX BFE
            #pragma unroll
            for (int j = 0; j < PACKED_ELEMENTS_PER_WORD; j++) {
                uint32_t quantized_val;
                asm volatile("bfe.u32 %0, %1, %2, %3;" 
                    : "=r"(quantized_val) 
                    : "r"(current_packed), "r"(j * K_BITS), "r"(K_BITS));
                
                // Dequantize while next BFE is executing
                unpacked_items[i * PACKED_ELEMENTS_PER_WORD + j] = 
                    static_cast<float>(quantized_val) * scale;
            }
        }
        
        __syncthreads();
        
        // Stage 3: Coalesced store using CUB
        cub::BlockStore<float, BLOCK_THREADS, 
                       ITEMS_PER_THREAD * PACKED_ELEMENTS_PER_WORD,
                       cub::BLOCK_STORE_VECTORIZE>(temp_storage.storage.store)
            .Store(output + tile_offset * PACKED_ELEMENTS_PER_WORD, 
                   unpacked_items, num_items);
    }
};
```

### Instruction Overlap Strategy

The key to performance is hiding memory latency through instruction-level parallelism:

```cuda
template<int K_BITS>
__global__ void k_bit_unpack_kernel(
    const uint32_t* __restrict__ packed_data,
    float* __restrict__ output,
    const float* __restrict__ scales,
    const int num_elements
) {
    constexpr int BLOCK_THREADS = 128;
    constexpr int ITEMS_PER_THREAD = 4;
    
    // Shared memory for CUB
    __shared__ typename CUBKBitUnpacker<K_BITS, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_storage;
    
    // Grid-stride loop for processing large arrays
    int tile_offset = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD;
    int num_tiles = (num_elements + BLOCK_THREADS * ITEMS_PER_THREAD - 1) / 
                    (BLOCK_THREADS * ITEMS_PER_THREAD);
    
    for (int tile_idx = blockIdx.x; tile_idx < num_tiles; tile_idx += gridDim.x) {
        int tile_elements = min(BLOCK_THREADS * ITEMS_PER_THREAD, 
                               num_elements - tile_offset);
        
        CUBKBitUnpacker<K_BITS, BLOCK_THREADS, ITEMS_PER_THREAD>::UnpackTile(
            packed_data, output, scales, temp_storage, tile_elements, tile_offset
        );
        
        tile_offset += gridDim.x * BLOCK_THREADS * ITEMS_PER_THREAD;
    }
}
```

### Warp-Level Optimization for Cross-Word Values

When k-bit values span word boundaries, warp shuffle can efficiently handle the data movement:

```cuda
template<int K_BITS>
__device__ __forceinline__ uint32_t extract_cross_word_value(
    uint32_t current_word, 
    uint32_t next_word,
    int bit_offset
) {
    if (bit_offset + K_BITS <= 32) {
        // Simple case: value within single word
        uint32_t result;
        asm("bfe.u32 %0, %1, %2, %3;" 
            : "=r"(result) 
            : "r"(current_word), "r"(bit_offset), "r"(K_BITS));
        return result;
    } else {
        // Cross-word case: use warp shuffle to get next word
        int lane_id = threadIdx.x % 32;
        uint32_t neighbor_word = __shfl_sync(0xffffffff, current_word, lane_id + 1);
        
        // Extract bits from both words
        int bits_from_current = 32 - bit_offset;
        uint32_t low_bits, high_bits;
        
        asm("bfe.u32 %0, %1, %2, %3;" 
            : "=r"(low_bits) 
            : "r"(current_word), "r"(bit_offset), "r"(bits_from_current));
        
        asm("bfe.u32 %0, %1, %2, %3;" 
            : "=r"(high_bits) 
            : "r"(neighbor_word), "r"(0), "r"(K_BITS - bits_from_current));
        
        return low_bits | (high_bits << bits_from_current);
    }
}
```

## Complete Implementation with Dynamic K

```cuda
template<typename T>
class KBitQuantizer {
private:
    static constexpr int MAX_K = 16;
    
    // Kernel dispatcher
    template<int K>
    static void dispatch_unpack(
        const uint32_t* packed, 
        T* output, 
        const T* scales,
        int num_elements, 
        int group_size,
        cudaStream_t stream
    ) {
        constexpr int BLOCK_SIZE = 256;
        int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        k_bit_unpack_kernel<K><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            packed, output, scales, num_elements
        );
    }
    
public:
    static void unpack(
        const uint32_t* packed,
        T* output,
        const T* scales,
        int num_elements,
        int k_bits,
        int group_size,
        cudaStream_t stream = 0
    ) {
        switch(k_bits) {
            case 1: dispatch_unpack<1>(packed, output, scales, num_elements, group_size, stream); break;
            case 2: dispatch_unpack<2>(packed, output, scales, num_elements, group_size, stream); break;
            case 3: dispatch_unpack<3>(packed, output, scales, num_elements, group_size, stream); break;
            case 4: dispatch_unpack<4>(packed, output, scales, num_elements, group_size, stream); break;
            case 5: dispatch_unpack<5>(packed, output, scales, num_elements, group_size, stream); break;
            case 6: dispatch_unpack<6>(packed, output, scales, num_elements, group_size, stream); break;
            case 7: dispatch_unpack<7>(packed, output, scales, num_elements, group_size, stream); break;
            case 8: dispatch_unpack<8>(packed, output, scales, num_elements, group_size, stream); break;
            default: 
                throw std::runtime_error("Unsupported k-bit width: " + std::to_string(k_bits));
        }
    }
};
```

## Performance Best Practices

### 1. Memory Access Pattern
- CUB's `BLOCK_LOAD_VECTORIZE` ensures coalesced access
- Process multiple words per thread to amortize overhead
- Use grid-stride loops for large arrays

### 2. Instruction Scheduling
- Place scale factor loads between BFE operations
- Unroll inner loops for better ILP
- Use `#pragma unroll` judiciously to balance register usage

### 3. Warp Efficiency
- Minimize divergence in boundary handling
- Use warp shuffle for cross-word values
- Keep all threads active when possible

### 4. Optimization Checklist
- [ ] Profile memory bandwidth utilization (target >80% of peak)
- [ ] Monitor instruction throughput (IPC > 2.5)
- [ ] Check warp execution efficiency (>95%)
- [ ] Verify no bank conflicts in shared memory
- [ ] Tune block size and items per thread for specific GPU

## Conclusion

The combination of CUB's optimized memory primitives with PTX bit manipulation instructions provides an efficient solution for k-bit quantization. The key insights:

1. **CUB handles memory complexity** while custom PTX handles bit extraction
2. **Instruction overlap** through software pipelining hides latency
3. **Warp-level primitives** efficiently handle boundary cases
4. **Template-based dispatch** supports arbitrary k values with optimal performance

This approach achieves near-peak memory bandwidth while efficiently unpacking quantized values, making it suitable for high-performance inference workloads.
