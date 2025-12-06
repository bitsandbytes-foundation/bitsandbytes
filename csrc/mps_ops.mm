#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace {

typedef struct {
    void* storage;
    size_t byte_offset;
    size_t nbytes;
} BNBMPSTensor;

static inline id<MTLDevice> get_device() {
    static id<MTLDevice> device = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        device = MTLCreateSystemDefaultDevice();
    if (!device) {
            NSLog(@"bitsandbytes: failed to acquire Metal device");
        abort();
    }
    });
    return device;
}

static inline id<MTLCommandQueue> get_command_queue() {
    static id<MTLCommandQueue> queue = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        queue = [get_device() newCommandQueue];
        if (!queue) {
            NSLog(@"bitsandbytes: failed to create Metal command queue");
            abort();
        }
    });
    return queue;
}

static inline NSURL* metallib_url() {
    Dl_info info;
    if (dladdr(reinterpret_cast<const void*>(&metallib_url), &info) == 0) {
        NSLog(@"bitsandbytes: dladdr failed to resolve metallib path");
        abort();
    }
    NSString* dylibPath = [NSString stringWithUTF8String:info.dli_fname];
    NSString* directory = [dylibPath stringByDeletingLastPathComponent];
    NSString* metallibPath = [directory stringByAppendingPathComponent:@"bitsandbytes.metallib"];
    return [NSURL fileURLWithPath:metallibPath];
}

static inline id<MTLLibrary> get_library() {
    static id<MTLLibrary> library = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSError* error = nil;
        library = [get_device() newLibraryWithURL:metallib_url() error:&error];
    if (!library) {
            NSLog(@"bitsandbytes: failed to load bitsandbytes.metallib (%@)", error);
            abort();
        }
    });
    return library;
}

static inline id<MTLComputePipelineState> get_pipeline(NSString* functionName) {
    static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cache = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        cache = [[NSMutableDictionary alloc] init];
    });

    id<MTLComputePipelineState> pipeline = cache[functionName];
    if (pipeline) {
        return pipeline;
    }

    NSError* error = nil;
    id<MTLFunction> function = [get_library() newFunctionWithName:functionName];
    if (!function) {
        NSLog(@"bitsandbytes: missing Metal kernel %@", functionName);
        abort();
    }

    pipeline = [get_device() newComputePipelineStateWithFunction:function error:&error];
    [function release];

    if (!pipeline) {
        NSLog(@"bitsandbytes: failed to create pipeline for %@ (%@)", functionName, error);
        abort();
    }

    cache[functionName] = pipeline;
    return pipeline;
}

struct TensorView {
    id<MTLBuffer> buffer;
    NSUInteger offset;
};

static inline TensorView make_tensor_view(const BNBMPSTensor& tensor, const char* label) {
    TensorView view;
    view.buffer = __builtin_bit_cast(id<MTLBuffer>, tensor.storage);
    view.offset = static_cast<NSUInteger>(tensor.byte_offset);
    if (!view.buffer && tensor.nbytes > 0) {
        NSLog(@"bitsandbytes: missing MTLBuffer for %s tensor (storage=%p, bytes=%zu)", label, tensor.storage, tensor.nbytes);
        abort();
    }
    return view;
}

static inline void dispatch_quant_kernel(
    NSString* name,
    const BNBMPSTensor& input,
    const BNBMPSTensor& absmax,
    const BNBMPSTensor& out,
    uint32_t blocksize,
    uint32_t n
) {
    if (n == 0) {
        return;
    }

    uint32_t blocks = (n + blocksize - 1) / blocksize;
    TensorView inputView = make_tensor_view(input, "input");
    TensorView absmaxView = make_tensor_view(absmax, "absmax");
    TensorView outView = make_tensor_view(out, "out");

    id<MTLCommandBuffer> commandBuffer = [get_command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> pipeline = get_pipeline(name);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputView.buffer offset:inputView.offset atIndex:0];
    [encoder setBuffer:absmaxView.buffer offset:absmaxView.offset atIndex:1];
    [encoder setBuffer:outView.buffer offset:outView.offset atIndex:2];
    [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&blocksize length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&blocks length:sizeof(uint32_t) atIndex:5];

    NSUInteger threadsPerThreadgroup = pipeline.threadExecutionWidth;
    if (threadsPerThreadgroup == 0) {
        threadsPerThreadgroup = 1;
    }
    MTLSize threads = MTLSizeMake(threadsPerThreadgroup, 1, 1);
    MTLSize grid = MTLSizeMake(blocks, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

}

static inline void dispatch_dequant_kernel(
    NSString* name,
    const BNBMPSTensor& packed,
    const BNBMPSTensor& absmax,
    const BNBMPSTensor& output,
    uint32_t blocksize,
    uint32_t n
) {
    if (n == 0) {
        return;
    }

    uint32_t blocks = (n + blocksize - 1) / blocksize;
    TensorView packedView = make_tensor_view(packed, "packed");
    TensorView absmaxView = make_tensor_view(absmax, "absmax");
    TensorView outputView = make_tensor_view(output, "output");

    id<MTLCommandBuffer> commandBuffer = [get_command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    id<MTLComputePipelineState> pipeline = get_pipeline(name);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:packedView.buffer offset:packedView.offset atIndex:0];
    [encoder setBuffer:absmaxView.buffer offset:absmaxView.offset atIndex:1];
    [encoder setBuffer:outputView.buffer offset:outputView.offset atIndex:2];
    [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&blocksize length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&blocks length:sizeof(uint32_t) atIndex:5];
    NSUInteger threadsPerThreadgroup = pipeline.threadExecutionWidth;
    if (threadsPerThreadgroup == 0) {
        threadsPerThreadgroup = 1;
    }
    MTLSize threads = MTLSizeMake(threadsPerThreadgroup, 1, 1);
    MTLSize grid = MTLSizeMake(blocks, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

}  // namespace

extern "C" {

void cquantize_blockwise_fp16_fp4(BNBMPSTensor input, BNBMPSTensor absmax, BNBMPSTensor out, int blocksize, const int n) {
    dispatch_quant_kernel(@"quantize_4bit_fp16_fp4", input, absmax, out, blocksize, n);
}

void cquantize_blockwise_fp16_nf4(BNBMPSTensor input, BNBMPSTensor absmax, BNBMPSTensor out, int blocksize, const int n) {
    dispatch_quant_kernel(@"quantize_4bit_fp16_nf4", input, absmax, out, blocksize, n);
}

void cquantize_blockwise_fp32_fp4(BNBMPSTensor input, BNBMPSTensor absmax, BNBMPSTensor out, int blocksize, const int n) {
    dispatch_quant_kernel(@"quantize_4bit_fp32_fp4", input, absmax, out, blocksize, n);
}

void cquantize_blockwise_fp32_nf4(BNBMPSTensor input, BNBMPSTensor absmax, BNBMPSTensor out, int blocksize, const int n) {
    dispatch_quant_kernel(@"quantize_4bit_fp32_nf4", input, absmax, out, blocksize, n);
}

void cdequantize_blockwise_fp16_fp4(BNBMPSTensor packed, BNBMPSTensor absmax, BNBMPSTensor out, int blocksize, const int n) {
    dispatch_dequant_kernel(@"dequantize_4bit_fp16_fp4", packed, absmax, out, blocksize, n);
}

void cdequantize_blockwise_fp16_nf4(BNBMPSTensor packed, BNBMPSTensor absmax, BNBMPSTensor out, int blocksize, const int n) {
    dispatch_dequant_kernel(@"dequantize_4bit_fp16_nf4", packed, absmax, out, blocksize, n);
}

void cdequantize_blockwise_fp32_fp4(BNBMPSTensor packed, BNBMPSTensor absmax, BNBMPSTensor out, int blocksize, const int n) {
    dispatch_dequant_kernel(@"dequantize_4bit_fp32_fp4", packed, absmax, out, blocksize, n);
}

void cdequantize_blockwise_fp32_nf4(BNBMPSTensor packed, BNBMPSTensor absmax, BNBMPSTensor out, int blocksize, const int n) {
    dispatch_dequant_kernel(@"dequantize_4bit_fp32_nf4", packed, absmax, out, blocksize, n);
}

}  // extern "C"
