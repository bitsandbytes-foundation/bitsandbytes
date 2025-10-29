#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSStream.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/tensor_types.h>

namespace {

struct BlockParams {
    uint32_t n;
    uint32_t blocksize;
    uint32_t threads_per_group;
};

struct BufferBinding {
    id<MTLBuffer> buffer = nil;
    size_t offset = 0;
    size_t length = 0;
    const void* host_src = nullptr;
    void* host_dst = nullptr;
    bool owns_buffer = false;
};

static inline MPSGraph* get_graph() {
    static MPSGraph* cur = nil;
    if (!cur) {
        cur = [[MPSGraph alloc] init];
    }
    return cur;
}

static inline id<MTLDevice> get_device() {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (!stream) {
        stream = at::mps::getDefaultMPSStream();
    }
    if (stream) {
        return (id<MTLDevice>)stream->device();
    }

    static id<MTLDevice> fallback = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        fallback = MTLCreateSystemDefaultDevice();
    });
    if (!fallback) {
        NSLog(@"bitsandbytes: failed to acquire MPS device");
        abort();
    }
    return fallback;
}

static id<MTLLibrary> get_library();

static NSString* get_metallib_path() {
    static NSString* metallib_path = nil;
    static dispatch_once_t once_token;
    dispatch_once(&once_token, ^{
        Dl_info info;
        if (dladdr(reinterpret_cast<const void*>(&get_library), &info) && info.dli_fname) {
            NSString* dylib_path = [NSString stringWithUTF8String:info.dli_fname];
            NSString* dylib_dir = [dylib_path stringByDeletingLastPathComponent];
            NSString* candidate = [dylib_dir stringByAppendingPathComponent:@"bitsandbytes.metallib"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
                metallib_path = [candidate retain];
                return;
            }
        }

        PyGILState_STATE gil = PyGILState_Ensure();
        PyObject* module = PyImport_ImportModule("bitsandbytes");
        if (!module) {
            PyErr_Clear();
            PyGILState_Release(gil);
            return;
        }

        PyObject* file_attr = PyObject_GetAttrString(module, "__file__");
        if (!file_attr) {
            PyErr_Clear();
            Py_DECREF(module);
            PyGILState_Release(gil);
            return;
        }

        const char* module_path_cstr = PyUnicode_AsUTF8(file_attr);
        if (module_path_cstr) {
            NSString* module_path = [NSString stringWithUTF8String:module_path_cstr];
            NSString* module_dir = [module_path stringByDeletingLastPathComponent];
            NSString* candidate = [module_dir stringByAppendingPathComponent:@"bitsandbytes.metallib"];

            if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
                metallib_path = [candidate retain];
            }
        } else {
            PyErr_Clear();
        }

        Py_DECREF(file_attr);
        Py_DECREF(module);
        PyGILState_Release(gil);
    });
    return metallib_path;
}

static inline id<MTLLibrary> get_library() {
    static id<MTLLibrary> library = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSError* error = nil;
        id<MTLDevice> device = get_device();
        NSString* metallib_path = get_metallib_path();
        NSURL* url = metallib_path ? [NSURL fileURLWithPath:metallib_path]
                                   : [NSURL fileURLWithPath:@"bitsandbytes.metallib"];
        library = [device newLibraryWithURL:url error:&error];
        if (!library) {
            NSLog(@"bitsandbytes: failed to load bitsandbytes.metallib (%@)", error);
            abort();
        }
    });
    return library;
}

static id<MTLComputePipelineState> get_pipeline(NSString* function_name) {
    static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines = nil;
    static dispatch_once_t once_token;
    dispatch_once(&once_token, ^{
        pipelines = [[NSMutableDictionary alloc] init];
    });

    id<MTLComputePipelineState> pipeline = pipelines[function_name];
    if (pipeline) {
        return pipeline;
    }

    NSError* error = nil;
    id<MTLFunction> function = [get_library() newFunctionWithName:function_name];
    if (!function) {
        NSLog(@"bitsandbytes: Metal function %@ not found", function_name);
        abort();
    }

    pipeline = [get_device() newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (!pipeline) {
        NSLog(@"bitsandbytes: failed to create pipeline for %@ (%@)", function_name, error);
        abort();
    }

    pipelines[function_name] = pipeline;
    return pipeline;
}

static NSUInteger preferred_threads(id<MTLComputePipelineState> pipeline) {
    NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (max_threads == 0) {
        return 64;
    }
    return std::min<NSUInteger>(64, max_threads);
}

static bool dispatch_blockwise_kernel(
    NSString* function_name,
    BufferBinding& input,
    BufferBinding& absmax,
    BufferBinding& output,
    uint32_t n,
    uint32_t blocksize) {

    if (n == 0 || blocksize == 0) {
        return true;
    }

    id<MTLComputePipelineState> pipeline = get_pipeline(function_name);
    const NSUInteger threads_per_group = preferred_threads(pipeline);
    const uint32_t num_blocks = (blocksize > 0) ? (n + blocksize - 1) / blocksize : 0;
    if (num_blocks == 0) {
        return true;
    }

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (!stream) {
        stream = at::mps::getDefaultMPSStream();
    }
    if (!stream) {
        PyErr_SetString(PyExc_RuntimeError, "bitsandbytes: failed to acquire current MPS stream");
        return false;
    }

    id<MTLDevice> device = (id<MTLDevice>)stream->device();
    if (!device) {
        PyErr_SetString(PyExc_RuntimeError, "bitsandbytes: failed to acquire MPS device");
        return false;
    }

    __block BufferBinding input_binding = input;
    __block BufferBinding absmax_binding = absmax;
    __block BufferBinding output_binding = output;

    __block bool success = true;
    at::native::mps::dispatch_sync_with_rethrow(stream->queue(), ^(){
        @autoreleasepool {
            auto prepare_buffer = ^(BufferBinding& binding) {
                if (binding.length == 0) {
                    return;
                }
                if (!binding.buffer) {
                    binding.buffer = [device newBufferWithLength:binding.length options:MTLResourceStorageModeShared];
                    binding.owns_buffer = true;
                    binding.offset = 0;
                }
                if (binding.host_src) {
                    std::memcpy((uint8_t*)binding.buffer.contents + binding.offset, binding.host_src, binding.length);
                }
            };

            prepare_buffer(input_binding);
            prepare_buffer(absmax_binding);
            prepare_buffer(output_binding);

            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            if (!encoder) {
                PyErr_SetString(PyExc_RuntimeError, "bitsandbytes: failed to obtain command encoder");
                success = false;
                return;
            }

            [encoder setComputePipelineState:pipeline];

            if (input_binding.buffer) {
                [encoder setBuffer:input_binding.buffer offset:input_binding.offset atIndex:0];
                [encoder useResource:input_binding.buffer usage:MTLResourceUsageRead];
            }
            if (absmax_binding.buffer) {
                [encoder setBuffer:absmax_binding.buffer offset:absmax_binding.offset atIndex:1];
                [encoder useResource:absmax_binding.buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            }
            if (output_binding.buffer) {
                [encoder setBuffer:output_binding.buffer offset:output_binding.offset atIndex:2];
                [encoder useResource:output_binding.buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            }

            BlockParams params = {n, blocksize, static_cast<uint32_t>(threads_per_group)};
            [encoder setBytes:&params length:sizeof(BlockParams) atIndex:3];

            MTLSize threads = MTLSizeMake(threads_per_group, 1, 1);
            MTLSize threadgroups = MTLSizeMake(num_blocks, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
        }
    });
    if (!success) {
        return false;
    }

    const bool needs_host_read =
        (absmax_binding.host_dst && absmax_binding.length) ||
        (output_binding.host_dst && output_binding.length);

    stream->synchronize(needs_host_read ? at::mps::SyncType::COMMIT_AND_WAIT
                                        : at::mps::SyncType::COMMIT);

    auto copy_back = [](BufferBinding& binding) {
        if (binding.buffer && binding.host_dst && binding.length) {
            std::memcpy(binding.host_dst, (uint8_t*)binding.buffer.contents + binding.offset, binding.length);
        }
        if (binding.owns_buffer && binding.buffer) {
            [binding.buffer release];
            binding.buffer = nil;
        }
    };

    copy_back(input_binding);
    copy_back(absmax_binding);
    copy_back(output_binding);
    return true;
}

static bool tensor_from_pyobject(PyObject* obj, const char* name, at::Tensor& tensor) {
    if (!obj) {
        PyErr_Format(PyExc_TypeError, "%s must not be null", name);
        return false;
    }
    if (!THPVariable_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a torch.Tensor", name);
        return false;
    }
    tensor = THPVariable_Unpack(reinterpret_cast<THPVariable*>(obj));
    if (tensor.device().type() != c10::DeviceType::MPS) {
        PyErr_Format(PyExc_RuntimeError, "%s must be an MPS tensor", name);
        return false;
    }
    if (!tensor.is_contiguous()) {
        PyErr_Format(PyExc_RuntimeError, "%s must be contiguous for MPS kernels", name);
        return false;
    }
    return true;
}

static bool binding_from_tensor(const at::Tensor& tensor, BufferBinding& binding) {
    binding.buffer = at::native::mps::getMTLBufferStorage(tensor);
    if (binding.buffer == nil) {
        PyErr_SetString(PyExc_RuntimeError, "bitsandbytes: tensor does not have an associated MTLBuffer");
        return false;
    }
    binding.offset = static_cast<size_t>(tensor.storage_offset()) * tensor.element_size();
    binding.length = static_cast<size_t>(tensor.numel()) * tensor.element_size();
    binding.owns_buffer = false;
    return true;
}

static BufferBinding binding_from_host(const void* ptr, size_t length, bool copy_to_device, bool retrieve_from_device) {
    BufferBinding binding;
    binding.length = length;
    if (copy_to_device) {
        binding.host_src = ptr;
    }
    if (retrieve_from_device) {
        binding.host_dst = const_cast<void*>(ptr);
    }
    return binding;
}

} // namespace

extern "C" {

// Pointer-based entry points (used for CPU fallback / legacy paths)
void cquantize_blockwise_fp16_nf4(
    float* /*code*/,
    void* A,
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n) {
    if (!A || !absmax || !out || n <= 0 || blocksize <= 0) {
        return;
    }
    const size_t absmax_blocks = static_cast<size_t>((n + blocksize - 1) / blocksize);
    BufferBinding input_binding = binding_from_host(A, static_cast<size_t>(n) * sizeof(uint16_t), true, false);
    BufferBinding absmax_binding = binding_from_host(absmax, absmax_blocks * sizeof(float), false, true);
    BufferBinding output_binding = binding_from_host(out, static_cast<size_t>((n + 1) / 2), false, true);
    if (!dispatch_blockwise_kernel(
        @"quantize_nf4_f16",
        input_binding,
        absmax_binding,
        output_binding,
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(blocksize))) {
        return;
    }
}

void cquantize_blockwise_fp32_nf4(
    float* /*code*/,
    float* A,
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n) {
    if (!A || !absmax || !out || n <= 0 || blocksize <= 0) {
        return;
    }
    const size_t absmax_blocks = static_cast<size_t>((n + blocksize - 1) / blocksize);
    BufferBinding input_binding = binding_from_host(A, static_cast<size_t>(n) * sizeof(float), true, false);
    BufferBinding absmax_binding = binding_from_host(absmax, absmax_blocks * sizeof(float), false, true);
    BufferBinding output_binding = binding_from_host(out, static_cast<size_t>((n + 1) / 2), false, true);
    if (!dispatch_blockwise_kernel(
        @"quantize_nf4_f32",
        input_binding,
        absmax_binding,
        output_binding,
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(blocksize))) {
        return;
    }
}

void cdequantize_blockwise_fp16_nf4(
    float* /*code*/,
    unsigned char* A,
    float* absmax,
    void* out,
    int blocksize,
    int n) {
    if (!A || !absmax || !out || n <= 0 || blocksize <= 0) {
        return;
    }
    const size_t absmax_blocks = static_cast<size_t>((n + blocksize - 1) / blocksize);
    BufferBinding input_binding = binding_from_host(A, static_cast<size_t>((n + 1) / 2), true, false);
    BufferBinding absmax_binding = binding_from_host(absmax, absmax_blocks * sizeof(float), true, false);
    BufferBinding output_binding = binding_from_host(out, static_cast<size_t>(n) * sizeof(uint16_t), false, true);
    if (!dispatch_blockwise_kernel(
        @"dequantize_nf4_f16",
        input_binding,
        absmax_binding,
        output_binding,
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(blocksize))) {
        return;
    }
}

void cdequantize_blockwise_fp32_nf4(
    float* /*code*/,
    unsigned char* A,
    float* absmax,
    float* out,
    int blocksize,
    int n) {
    if (!A || !absmax || !out || n <= 0 || blocksize <= 0) {
        return;
    }
    const size_t absmax_blocks = static_cast<size_t>((n + blocksize - 1) / blocksize);
    BufferBinding input_binding = binding_from_host(A, static_cast<size_t>((n + 1) / 2), true, false);
    BufferBinding absmax_binding = binding_from_host(absmax, absmax_blocks * sizeof(float), true, false);
    BufferBinding output_binding = binding_from_host(out, static_cast<size_t>(n) * sizeof(float), false, true);
    if (!dispatch_blockwise_kernel(
        @"dequantize_nf4_f32",
        input_binding,
        absmax_binding,
        output_binding,
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(blocksize))) {
        return;
    }
}

// Tensor-aware entry points (used from Python to avoid extra copies)
void cquantize_blockwise_fp16_nf4_tensor(PyObject* A_obj, PyObject* absmax_obj, PyObject* out_obj, int blocksize) {
    at::Tensor A;
    at::Tensor absmax;
    at::Tensor out;
    if (!tensor_from_pyobject(A_obj, "A", A) ||
        !tensor_from_pyobject(absmax_obj, "absmax", absmax) ||
        !tensor_from_pyobject(out_obj, "out", out)) {
        return;
    }

    if (A.scalar_type() != at::kHalf) {
        PyErr_SetString(PyExc_TypeError, "A must be float16 for NF4 quantization");
        return;
    }
    if (absmax.scalar_type() != at::kFloat) {
        PyErr_SetString(PyExc_TypeError, "absmax must be float32");
        return;
    }
    if (out.scalar_type() != at::kByte) {
        PyErr_SetString(PyExc_TypeError, "out must be uint8");
        return;
    }

    BufferBinding input_binding;
    BufferBinding absmax_binding;
    BufferBinding output_binding;
    if (!binding_from_tensor(A, input_binding) ||
        !binding_from_tensor(absmax, absmax_binding) ||
        !binding_from_tensor(out, output_binding)) {
        return;
    }

    if (!dispatch_blockwise_kernel(
        @"quantize_nf4_f16",
        input_binding,
        absmax_binding,
        output_binding,
        static_cast<uint32_t>(A.numel()),
        static_cast<uint32_t>(blocksize))) {
        return;
    }
}

void cquantize_blockwise_fp32_nf4_tensor(PyObject* A_obj, PyObject* absmax_obj, PyObject* out_obj, int blocksize) {
    at::Tensor A;
    at::Tensor absmax;
    at::Tensor out;
    if (!tensor_from_pyobject(A_obj, "A", A) ||
        !tensor_from_pyobject(absmax_obj, "absmax", absmax) ||
        !tensor_from_pyobject(out_obj, "out", out)) {
        return;
    }

    if (A.scalar_type() != at::kFloat) {
        PyErr_SetString(PyExc_TypeError, "A must be float32 for NF4 quantization");
        return;
    }
    if (absmax.scalar_type() != at::kFloat) {
        PyErr_SetString(PyExc_TypeError, "absmax must be float32");
        return;
    }
    if (out.scalar_type() != at::kByte) {
        PyErr_SetString(PyExc_TypeError, "out must be uint8");
        return;
    }

    BufferBinding input_binding;
    BufferBinding absmax_binding;
    BufferBinding output_binding;
    if (!binding_from_tensor(A, input_binding) ||
        !binding_from_tensor(absmax, absmax_binding) ||
        !binding_from_tensor(out, output_binding)) {
        return;
    }

    if (!dispatch_blockwise_kernel(
        @"quantize_nf4_f32",
        input_binding,
        absmax_binding,
        output_binding,
        static_cast<uint32_t>(A.numel()),
        static_cast<uint32_t>(blocksize))) {
        return;
    }
}

void cdequantize_blockwise_fp16_nf4_tensor(PyObject* A_obj, PyObject* absmax_obj, PyObject* out_obj, int blocksize) {
    at::Tensor A;
    at::Tensor absmax;
    at::Tensor out;
    if (!tensor_from_pyobject(A_obj, "A", A) ||
        !tensor_from_pyobject(absmax_obj, "absmax", absmax) ||
        !tensor_from_pyobject(out_obj, "out", out)) {
        return;
    }

    if (A.scalar_type() != at::kByte) {
        PyErr_SetString(PyExc_TypeError, "A must be uint8 for NF4 dequantization");
        return;
    }
    if (absmax.scalar_type() != at::kFloat) {
        PyErr_SetString(PyExc_TypeError, "absmax must be float32");
        return;
    }
    if (out.scalar_type() != at::kHalf) {
        PyErr_SetString(PyExc_TypeError, "out must be float16");
        return;
    }

    BufferBinding input_binding;
    BufferBinding absmax_binding;
    BufferBinding output_binding;
    if (!binding_from_tensor(A, input_binding) ||
        !binding_from_tensor(absmax, absmax_binding) ||
        !binding_from_tensor(out, output_binding)) {
        return;
    }

    if (!dispatch_blockwise_kernel(
        @"dequantize_nf4_f16",
        input_binding,
        absmax_binding,
        output_binding,
        static_cast<uint32_t>(out.numel()),
        static_cast<uint32_t>(blocksize))) {
        return;
    }
}

void cdequantize_blockwise_fp32_nf4_tensor(PyObject* A_obj, PyObject* absmax_obj, PyObject* out_obj, int blocksize) {
    at::Tensor A;
    at::Tensor absmax;
    at::Tensor out;
    if (!tensor_from_pyobject(A_obj, "A", A) ||
        !tensor_from_pyobject(absmax_obj, "absmax", absmax) ||
        !tensor_from_pyobject(out_obj, "out", out)) {
        return;
    }

    if (A.scalar_type() != at::kByte) {
        PyErr_SetString(PyExc_TypeError, "A must be uint8 for NF4 dequantization");
        return;
    }
    if (absmax.scalar_type() != at::kFloat) {
        PyErr_SetString(PyExc_TypeError, "absmax must be float32");
        return;
    }
    if (out.scalar_type() != at::kFloat) {
        PyErr_SetString(PyExc_TypeError, "out must be float32");
        return;
    }

    BufferBinding input_binding;
    BufferBinding absmax_binding;
    BufferBinding output_binding;
    if (!binding_from_tensor(A, input_binding) ||
        !binding_from_tensor(absmax, absmax_binding) ||
        !binding_from_tensor(out, output_binding)) {
        return;
    }

    if (!dispatch_blockwise_kernel(
        @"dequantize_nf4_f32",
        input_binding,
        absmax_binding,
        output_binding,
        static_cast<uint32_t>(out.numel()),
        static_cast<uint32_t>(blocksize))) {
        return;
    }
}

} // extern "C"
