#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

namespace {

typedef struct {
    void* storage;
    size_t byte_offset;
    size_t nbytes;
} BNBMPSTensor;

static constexpr NSUInteger kMaxThreadsPerThreadgroup = 512;

static inline at::mps::MPSStream* get_default_stream() {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (!stream) {
        NSLog(@"bitsandbytes: PyTorch MPS stream is unavailable");
        abort();
    }
    return stream;
}

static inline id<MTLDevice> get_device() {
    return get_default_stream()->device();
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

    @synchronized(cache) {
        id<MTLComputePipelineState> pipeline = cache[functionName];
        if (pipeline) {
            return pipeline;
        }
    }

    NSError* error = nil;
    id<MTLFunction> function = [get_library() newFunctionWithName:functionName];
    if (!function) {
        NSLog(@"bitsandbytes: missing Metal kernel %@", functionName);
        abort();
    }

    id<MTLComputePipelineState> pipeline = [get_device() newComputePipelineStateWithFunction:function error:&error];
    [function release];

    if (!pipeline) {
        NSLog(@"bitsandbytes: failed to create pipeline for %@ (%@)", functionName, error);
        abort();
    }

    @synchronized(cache) {
        cache[functionName] = pipeline;
    }
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

    at::mps::MPSStream* stream = get_default_stream();
    // stream->endKernelCoalescing();
    id<MTLCommandBuffer> command_buffer_obj = stream->commandBuffer();

    id<MTLComputePipelineState> pipeline_state_obj = (id<MTLComputePipelineState>) get_pipeline(name);

    id<MTLComputeCommandEncoder> command_encoder_obj = stream->commandEncoder();

    // Set kernel arguments
    [command_encoder_obj setComputePipelineState:pipeline_state_obj];
    [command_encoder_obj setBuffer:inputView.buffer offset:inputView.offset atIndex:0];
    [command_encoder_obj setBuffer:absmaxView.buffer offset:absmaxView.offset atIndex:1];
    [command_encoder_obj setBuffer:outView.buffer offset:outView.offset atIndex:2];
    [command_encoder_obj setBytes:&n length:sizeof(uint32_t) atIndex:3];
    [command_encoder_obj setBytes:&blocksize length:sizeof(uint32_t) atIndex:4];
    [command_encoder_obj setBytes:&blocks length:sizeof(uint32_t) atIndex:5];
    NSUInteger threadsPerThreadgroup = pipeline_state_obj.threadExecutionWidth;
    if (threadsPerThreadgroup == 0) {
        threadsPerThreadgroup = 1;
    }
    MTLSize threads = MTLSizeMake(threadsPerThreadgroup, 1, 1);
    MTLSize grid = MTLSizeMake(blocks, 1, 1);
    [command_encoder_obj dispatchThreads:grid threadsPerThreadgroup:threads];
    // [command_encoder_obj endEncoding];
    stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
}

static inline void dispatch_dequant_kernel(
    NSString* name,
    const BNBMPSTensor& packed,
    const BNBMPSTensor& absmax,
    const BNBMPSTensor& output,
    uint32_t blocksize,
    uint32_t n
) {
    // NSLog(@"bitsandbytes: dispatching dequant kernel %@ with blocksize=%d, n=%d", name, blocksize, n);
    if (n == 0) {
        return;
    }
    uint32_t blocks = (n + blocksize - 1) / blocksize;
    TensorView inputView = make_tensor_view(packed, "packed");
    TensorView absmaxView = make_tensor_view(absmax, "absmax");
    TensorView outView = make_tensor_view(output, "output");

    at::mps::MPSStream* stream = get_default_stream();
    // stream->endKernelCoalescing();
    id<MTLCommandBuffer> command_buffer_obj = stream->commandBuffer();

    id<MTLComputePipelineState> pipeline_state_obj = (id<MTLComputePipelineState>) get_pipeline(name);

    id<MTLComputeCommandEncoder> command_encoder_obj = stream->commandEncoder();

    // Set kernel arguments
    [command_encoder_obj setComputePipelineState:pipeline_state_obj];
    [command_encoder_obj setBuffer:inputView.buffer offset:inputView.offset atIndex:0];
    [command_encoder_obj setBuffer:absmaxView.buffer offset:absmaxView.offset atIndex:1];
    [command_encoder_obj setBuffer:outView.buffer offset:outView.offset atIndex:2];
    [command_encoder_obj setBytes:&n length:sizeof(uint32_t) atIndex:3];
    [command_encoder_obj setBytes:&blocksize length:sizeof(uint32_t) atIndex:4];
    [command_encoder_obj setBytes:&blocks length:sizeof(uint32_t) atIndex:5];
    NSUInteger maxThreadsPerTG = pipeline_state_obj.maxTotalThreadsPerThreadgroup;
    NSUInteger desiredThreads = (blocksize + 1) / 2;
    if (desiredThreads == 0) {
        desiredThreads = 1;
    }
    NSUInteger threadsPerThreadgroup =
        std::min(maxThreadsPerTG, std::max<NSUInteger>(1, desiredThreads));
    if (threadsPerThreadgroup < pipeline_state_obj.threadExecutionWidth) {
        threadsPerThreadgroup = std::min(pipeline_state_obj.threadExecutionWidth, maxThreadsPerTG);
    }

    NSUInteger totalThreads = threadsPerThreadgroup * blocks;
    MTLSize threads = MTLSizeMake(threadsPerThreadgroup, 1, 1);
    MTLSize grid = MTLSizeMake(totalThreads, 1, 1);
    [command_encoder_obj dispatchThreads:grid threadsPerThreadgroup:threads];
    // [command_encoder_obj endEncoding];
    stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
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