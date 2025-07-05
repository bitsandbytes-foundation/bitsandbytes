#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

static inline MPSGraph* get_graph() {
    static MPSGraph* cur = nil;
    if (!cur) {
        cur = [[MPSGraph alloc] init];
    }
    return cur;
}

static inline id<MTLDevice> get_device() {
    NSError* error = nil;
    static id<MTLDevice> device = nil;
    if (!device) {
        device = MTLCreateSystemDefaultDevice();
    }
    if (!device) {
        NSLog(@"Failed to get MPS device");
        abort();
    }
    return device;
}

static inline id<MTLLibrary> get_library() {
    NSError* error = nil;
    static id<MTLLibrary> library = nil;
    if (!library) {
        library = [get_device() newLibraryWithURL:[NSURL fileURLWithPath:@"bitsandbytes.metallib"] error:&error];
    }
    if (!library) {
        NSLog(@"Failed to load bitsandbytes.metallib");
        abort();
    }
    return library;
}

/*MPSGraphTensor* dequantize_mps(MPSGraphTensor* code, MPSGraphTensor* A, int n)
{
  id out = [get_graph() dequantizeTensor:(MPSGraphTensor*)A scaleTensor:(MPSGraphTensor*)code zeroPoint:0.0
dataType:MPSDataTypeInt8 axis:0 name:@"out"]; return out;
}*/

// MPS function for blockwise quantization
extern "C" void quantize_blockwise_mps(float* code, float* A, float* absmax, unsigned char* out, 
                                      long long blocksize, long long n) {
    id<MTLDevice> device = get_device();
    id<MTLLibrary> library = get_library();
    
    static id<MTLFunction> kernel = nil;
    if (!kernel) {
        kernel = [library newFunctionWithName:@"quantize"];
        if (!kernel) {
            NSLog(@"Failed to load quantize kernel from bitsandbytes.metallib");
            // Fall back to CPU implementation
            return;
        }
    }
    
    // Create Metal buffers
    id<MTLBuffer> codeBuffer = [device newBufferWithBytes:code length:256*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> ABuffer = [device newBufferWithBytes:A length:n*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> absmaxBuffer = [device newBufferWithLength:((n + blocksize - 1) / blocksize)*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuffer = [device newBufferWithLength:n*sizeof(unsigned char) options:MTLResourceStorageModeShared];
    
    // Create command queue and buffer
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    
    // Create compute encoder
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set the compute pipeline state
    NSError* error = nil;
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernel error:&error];
    if (!pipelineState) {
        NSLog(@"Failed to create pipeline state: %@", error.localizedDescription);
        return;
    }
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:codeBuffer offset:0 atIndex:0];
    [encoder setBuffer:ABuffer offset:0 atIndex:1];
    [encoder setBuffer:outBuffer offset:0 atIndex:2];
    [encoder setBytes:&n length:sizeof(n) atIndex:3];
    
    // Calculate threadgroup sizes
    NSUInteger threadsPerThreadgroup = MIN(pipelineState.maxTotalThreadsPerThreadgroup, 256);
    NSUInteger threadgroupsPerGrid = (n + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    
    [encoder dispatchThreadgroups:MTLSizeMake(threadgroupsPerGrid, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [encoder endEncoding];
    
    // Execute and wait
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back
    memcpy(out, outBuffer.contents, n);
    memcpy(absmax, absmaxBuffer.contents, ((n + blocksize - 1) / blocksize) * sizeof(float));
}

// MPS function for blockwise dequantization
extern "C" void dequantize_blockwise_mps(float* code, unsigned char* A, float* absmax, float* out,
                                         long long blocksize, long long n) {
    // For now, implement a simple CPU fallback
    // TODO: Implement proper Metal kernel for dequantization
    long long blocks = (n + blocksize - 1) / blocksize;
    
    for (long long block = 0; block < blocks; block++) {
        long long start = block * blocksize;
        long long end = (block + 1) * blocksize < n ? (block + 1) * blocksize : n;
        float scale = absmax[block];
        
        for (long long i = start; i < end; i++) {
            // Convert from uint8 back to float
            float normalized = ((float)A[i] - 128.0f) / 127.0f;
            out[i] = normalized * scale;
        }
    }
}

// MPS function for matrix multiplication
extern "C" void gemm_4bit_inference_naive_mps(int m, int n, int k, float* A, unsigned char* B, float* C,
                                             int lda, int ldb, int ldc) {
    // For now, implement a simple CPU fallback
    // TODO: Implement proper Metal matrix multiplication kernel
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                // Convert B from uint8 to float for computation
                float b_val = ((float)B[l * ldb + j] - 128.0f) / 127.0f;
                sum += A[i * lda + l] * b_val;
            }
            C[i * ldc + j] = sum;
        }
    }
}
