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
  static id<MTLLibrary> library = nil;
  if(!library) {
    library = [get_device() newLibraryWithFile:@"bitsandbytes.metallib" error:&error];
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


// MPSGraph function for quantize  
extern "C" MPSGraphTensor* quantize_mps(MPSGraph* graph, MPSGraphTensor* code, MPSGraphTensor* A, int n)
{
  id<MTLDevice> device = get_device();
  id<MTLLibrary> library = get_library();
  static id<MTLFunction> kernel = nil;
  if(!kernel) {
    kernel = [library newFunctionWithName:@"quantize"];
    if(!kernel) {
      NSLog(@"Failed to load bitsandbytes.metallib");
      abort();
    }
  }
  id<MTLBuffer> codeBuffer = [device newBufferWithBytes:code.data.bytes length:code.data.length options:MTLResourceStorageModeShared];  
  id<MTLBuffer> ABuffer = [device newBufferWithBytes:A.data.bytes length:A.data.length options:MTLResourceStorageModeShared];  
  NSUInteger outSize = n * sizeof(unsigned char);  
  id<MTLBuffer> outBuffer = [device newBufferWithLength:outSize options:MTLResourceStorageModeShared];  
  id<MTLBuffer> nBuffer = [device newBufferWithBytes:&n length:sizeof(uint) options:MTLResourceStorageModeShared];
  
  id<MTLCommandQueue> commandQueue = [device newCommandQueue];  
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];  
  id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];  
  [commandEncoder setComputePipelineState:kernel];  
  [commandEncoder setBuffer:codeBuffer offset:0 atIndex:0];  
  [commandEncoder setBuffer:ABuffer offset:0 atIndex:1];  
  [commandEncoder setBuffer:outBuffer offset:0 atIndex:2];  
  [commandEncoder setBuffer:nBuffer offset:0 atIndex:3];  
    
  int threadsPerThreadgroup = kernel.threadExecutionWidth;  
  MTLSize threadsPerGrid = MTLSizeMake(n, 1, 1);  
  MTLSize threadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);  
  [commandEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadgroupSize];  
    
  [commandEncoder endEncoding];  
  [commandBuffer commit];  
  [commandBuffer waitUntilCompleted];  
    
  MPSGraphTensorData* outData = [[MPSGraphTensorData alloc] initWithBytesNoCopy:outBuffer.contents length:outSize deallocator:nil];  
  return [graph tensorWithTensorData:outData shape:@[@(n)]];
}  
