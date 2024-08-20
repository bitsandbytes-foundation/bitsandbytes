#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096

static inline MPSGraph* get_graph()
{
  static MPSGraph* cur = nil;
  if(!cur) {
    cur = [[MPSGraph alloc] init];
  }
  return cur;
}

static inline id<MTLDevice> get_device()
{
  NSError *error = nil;
  static id<MTLDevice> device = nil;
  if(!device) {
    device = MTLCreateSystemDefaultDevice();
  }
  if(!device) {
    NSLog(@"Failed to get MPS device");
    abort();
  }
  return device;
}

static inline id<MTLLibrary> get_library()
{
  NSError *error = nil;
  id<MTLDevice> device = get_device();
  static id<MTLLibrary> library = nil;
  if (!library) {
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *currentPath = [fileManager currentDirectoryPath];
    NSString *libraryPath = [currentPath stringByAppendingPathComponent:@"bitsandbytes/bitsandbytes.metallib"];
    library = [device newLibraryWithURL:[NSURL fileURLWithPath:libraryPath] error:&error];
  }
  if(!library) {
    NSLog(@"Failed to load bitsandbytes.metallib");
    abort();
  }
  return library;
}

/*MPSGraphTensor* dequantize_mps(MPSGraphTensor* code, MPSGraphTensor* A, int n)
{
  id out = [get_graph() dequantizeTensor:(MPSGraphTensor*)A scaleTensor:(MPSGraphTensor*)code zeroPoint:0.0 dataType:MPSDataTypeInt8 axis:0 name:@"out"];
  return out;
}*/

extern "C" void quantize_mps(float* code, float* A, float* absmax, uint8_t* out, int blocksize, const int n)
{
    @autoreleasepool {
        id<MTLDevice> device = get_device();
        if (!device) {
            NSLog(@"Failed to get MPS device");
            return;
        }
        NSError* error = nil;
        id<MTLLibrary> library = get_library();
        if (!library) {
            NSLog(@"Failed to load bitsandbytes.metallib: %@", error);
            return;
        }
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"quantize"];
        if (!kernelFunction) {
            NSLog(@"Failed to load `quantize` function");
            return;
        }
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            NSLog(@"Failed to create pipeline state: %@", error);
            return;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:pipelineState];

        @try {
            id<MTLBuffer> codeBuffer = [device newBufferWithBytes:code length:blocksize * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> inputBuffer = [device newBufferWithBytes:A length:n * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputBuffer = [device newBufferWithLength:((n + 1) / 2) * sizeof(uint8_t) options:MTLResourceStorageModeShared];

            [computeEncoder setBuffer:codeBuffer offset:0 atIndex:0];
            [computeEncoder setBuffer:inputBuffer offset:0 atIndex:1];
            [computeEncoder setBuffer:outputBuffer offset:0 atIndex:2];
            [computeEncoder setBytes:&n length:sizeof(int) atIndex:3];

            MTLSize gridSize = MTLSizeMake(n, 1, 1);
            MTLSize threadGroupSize = MTLSizeMake(pipelineState.maxTotalThreadsPerThreadgroup, 1, 1);
            NSLog(@"Dispatching compute encoder with gridSize: %zu, threadGroupSize: %zu", gridSize.width, threadGroupSize.width);
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

            [computeEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            memcpy(out, [outputBuffer contents], ((n + 1) / 2) * sizeof(uint8_t));
        } @catch (NSException *exception) {
            NSLog(@"Exception occurred: %@", exception);
        }
    }
}