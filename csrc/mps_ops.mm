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
    library = [get_device() newLibraryWithURL:[NSURL fileURLWithPath:@"bitsandbytes.metallib"] error:&error];
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
  NSLog(@"Not implemented");
  return nil;
}
