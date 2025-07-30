.github/scripts/auditwheel_show.py
==================================
main() # Read(file_path=.github/scripts/auditwheel_show.py, offset=5, limit=23)

.github/scripts/set_platform_tag.py
===================================
get_platform_tag(architecture) # Read(file_path=.github/scripts/set_platform_tag.py, offset=6, limit=13)
main() # Read(file_path=.github/scripts/set_platform_tag.py, offset=21, limit=8)

ast_test.py
===========
FunctionCallVisitor(ast.NodeVisitor) # Read(file_path=ast_test.py, offset=10, limit=65)
__init__(self) # Read(file_path=ast_test.py, offset=11, limit=5)
visit_FunctionDef(self, node) # Read(file_path=ast_test.py, offset=17, limit=15)
visit_Call(self, node) # Read(file_path=ast_test.py, offset=33, limit=24)
_get_attribute_source(self, node) # Read(file_path=ast_test.py, offset=58, limit=11)
visit_Name(self, node) # Read(file_path=ast_test.py, offset=70, limit=5)
build_call_graph(file_path) # Read(file_path=ast_test.py, offset=77, limit=24)
generate_random_trace(graph, min_depth, max_depth) # Read(file_path=ast_test.py, offset=103, limit=53)
print_stack_trace(trace, file_path, objects_used, function_locations) # Read(file_path=ast_test.py, offset=158, limit=38)
main() # Read(file_path=ast_test.py, offset=198, limit=53)

benchmarking/int8/training_benchmark.py
=======================================
test_bench_8bit_training(batch, seq, model, hidden) # Read(file_path=benchmarking/int8/training_benchmark.py, offset=28, limit=32)

benchmarking/matmul_benchmark.py
================================
test_bench_matmul(batch, seq, model, hidden) # Read(file_path=benchmarking/matmul_benchmark.py, offset=30, limit=170)

benchmarking/optimizer_benchmark.py
===================================
test_stream_optimizer_bench(dim1, gtype, optim_name, mode) # Read(file_path=benchmarking/optimizer_benchmark.py, offset=23, limit=34)

benchmarking/switchback/speed_benchmark.py
==========================================
get_time(k, fn, info_dict) # Read(file_path=benchmarking/switchback/speed_benchmark.py, offset=24, limit=14)

bitsandbytes/__init__.py
========================
_import_backends() # Read(file_path=bitsandbytes/__init__.py, offset=44, limit=19)

bitsandbytes/_ops.py
====================
_(A, CA, CB, SCA, SCB, outlier_cols, bias) # Read(file_path=bitsandbytes/_ops.py, offset=28, limit=17)
_(A, B, row_stats, col_stats, bias, dtype) # Read(file_path=bitsandbytes/_ops.py, offset=55, limit=10)
_(A, B) # Read(file_path=bitsandbytes/_ops.py, offset=74, limit=5)
_(A, B, out) # Read(file_path=bitsandbytes/_ops.py, offset=90, limit=8)
_(A, threshold) # Read(file_path=bitsandbytes/_ops.py, offset=107, limit=10)
_(A, stats) # Read(file_path=bitsandbytes/_ops.py, offset=123, limit=3)
_(A, stats) # Read(file_path=bitsandbytes/_ops.py, offset=130, limit=3)
_(A, row_stats, col_stats, dtype, bias) # Read(file_path=bitsandbytes/_ops.py, offset=142, limit=9)
_(A, threshold) # Read(file_path=bitsandbytes/_ops.py, offset=160, limit=11)
_(A, absmax, blocksize, quant_type, shape, dtype) # Read(file_path=bitsandbytes/_ops.py, offset=180, limit=10)
_(A, absmax, blocksize, quant_type, shape, dtype, out) # Read(file_path=bitsandbytes/_ops.py, offset=199, limit=13)
_(A, blocksize, quant_type, quant_storage) # Read(file_path=bitsandbytes/_ops.py, offset=221, limit=10)
_(A, absmax, code, blocksize, dtype) # Read(file_path=bitsandbytes/_ops.py, offset=240, limit=4)
_(A, absmax, code, blocksize, dtype, out) # Read(file_path=bitsandbytes/_ops.py, offset=253, limit=8)
_(A, code, blocksize) # Read(file_path=bitsandbytes/_ops.py, offset=267, limit=7)
_(A, B, shapeB, absmax, code, blocksize) # Read(file_path=bitsandbytes/_ops.py, offset=283, limit=15)
_(A, B, shapeB, absmax, code, blocksize, out) # Read(file_path=bitsandbytes/_ops.py, offset=307, limit=25)
_(A, absmax, blocksize, shape, dtype) # Read(file_path=bitsandbytes/_ops.py, offset=342, limit=9)

bitsandbytes/autograd/_functions.py
===================================
GlobalOutlierPooler() # Read(file_path=bitsandbytes/autograd/_functions.py, offset=24, limit=27)
__init__(self) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=27, limit=2)
initialize(self) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=30, limit=3)
get_instance(cls) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=35, limit=5)
add_outliers(self, outlier_idx, feature_dim) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=41, limit=7)
get_current_outlier_idx(self) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=49, limit=2)
get_inverse_transform_indices(transform_tile, tile_size) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=57, limit=29)
undo_layout(permuted_tensor, tile_indices) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=99, limit=16)
MatmulLtState() # Read(file_path=bitsandbytes/autograd/_functions.py, offset=118, limit=38)
reset_grads(self) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=143, limit=9)
tile_indices(self) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=154, limit=2)
MatMul8bitLt(torch.autograd.Function) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=158, limit=142)
forward(ctx, A, B, out, bias, state) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=160, limit=97)
backward(ctx, grad_output) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=259, limit=41)
MatMul8bitFp(torch.autograd.Function) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=302, limit=55)
forward(ctx, A, B, out, bias, state) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=309, limit=22)
backward(ctx, grad_output) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=333, limit=24)
MatMul4Bit(torch.autograd.Function) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=359, limit=55)
forward(ctx, A, B, out, bias, quant_state) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=364, limit=28)
backward(ctx, grad_output) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=394, limit=20)
matmul(A, B, out, state, threshold, bias) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=416, limit=16)
matmul_4bit(A, B, quant_state, out, bias) # Read(file_path=bitsandbytes/autograd/_functions.py, offset=434, limit=33)

bitsandbytes/backends/cpu/ops.py
================================
_(A, B) # Read(file_path=bitsandbytes/backends/cpu/ops.py, offset=20, limit=5)
_(A, code, blocksize) # Read(file_path=bitsandbytes/backends/cpu/ops.py, offset=28, limit=40)
_(A, absmax, code, blocksize, dtype) # Read(file_path=bitsandbytes/backends/cpu/ops.py, offset=71, limit=27)
_(A, absmax, blocksize, shape, dtype) # Read(file_path=bitsandbytes/backends/cpu/ops.py, offset=104, limit=17)

bitsandbytes/backends/cuda/ops.py
=================================
_(A, B) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=15, limit=3)
_(A, B, out) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=21, limit=2)
_int8_linear_matmul_impl(A, B, out) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=25, limit=61)
_(A, row_stats, col_stats, dtype, bias) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=89, limit=36)
_(A, threshold) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=128, limit=39)
_(A, threshold) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=170, limit=17)
_get_col_absmax(A, threshold) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=189, limit=19)
_(A, code, blocksize) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=211, limit=30)
_(A, absmax, code, blocksize, dtype) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=244, limit=4)
_(A, absmax, code, blocksize, dtype, out) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=251, limit=11)
_dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=264, limit=27)
_(A, blocksize, quant_type, quant_storage) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=294, limit=42)
_(A, absmax, blocksize, quant_type, shape, dtype) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=339, limit=11)
_(A, absmax, blocksize, quant_type, shape, dtype, out) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=353, limit=12)
_dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=367, limit=41)
_(A, B, shapeB, absmax, code, blocksize) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=411, limit=7)
_(A, B, shapeB, absmax, code, blocksize, out) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=421, limit=15)
_gemv_4bit_impl(A, B, shapeB, absmax, code, blocksize, out) # Read(file_path=bitsandbytes/backends/cuda/ops.py, offset=438, limit=86)

bitsandbytes/backends/default/ops.py
====================================
_(A, row_stats, col_stats, dtype, bias) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=12, limit=20)
_(A, CA, CB, SCA, SCB, outlier_cols, bias) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=35, limit=36)
_(A, B, row_stats, col_stats, bias, dtype) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=74, limit=16)
_(A, B) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=93, limit=2)
_(A, B, out) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=98, limit=3)
_int8_linear_matmul_impl(A, B, out) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=103, limit=6)
_(A, threshold) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=112, limit=35)
_(A, code, blocksize) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=150, limit=23)
_(A, absmax, code, blocksize, dtype) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=176, limit=14)
_(A, blocksize, quant_type, quant_storage) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=193, limit=40)
_(A, absmax, blocksize, quant_type, shape, dtype) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=236, limit=48)
_(A, B, shapeB, absmax, code, blocksize) # Read(file_path=bitsandbytes/backends/default/ops.py, offset=287, limit=17)

bitsandbytes/backends/hpu/ops.py
================================
_(A, absmax, blocksize, quant_type, shape, dtype) # Read(file_path=bitsandbytes/backends/hpu/ops.py, offset=13, limit=41)

bitsandbytes/backends/triton/ops.py
===================================
quantize_blockwise(A, code, blocksize) # Read(file_path=bitsandbytes/backends/triton/ops.py, offset=14, limit=14)
dequantize_blockwise(A, absmax, code, blocksize, dtype) # Read(file_path=bitsandbytes/backends/triton/ops.py, offset=30, limit=17)
dequantize_blockwise_inplace(A, absmax, code, blocksize, dtype, out) # Read(file_path=bitsandbytes/backends/triton/ops.py, offset=49, limit=16)
quantize_4bit(A, blocksize, quant_type, quant_storage) # Read(file_path=bitsandbytes/backends/triton/ops.py, offset=67, limit=29)
dequantize_4bit(A, absmax, blocksize, quant_type, shape, dtype) # Read(file_path=bitsandbytes/backends/triton/ops.py, offset=98, limit=26)
dequantize_4bit_inplace(A, absmax, blocksize, quant_type, shape, dtype, out) # Read(file_path=bitsandbytes/backends/triton/ops.py, offset=126, limit=12)
gemv_4bit(A, B, shapeB, absmax, code, blocksize) # Read(file_path=bitsandbytes/backends/triton/ops.py, offset=140, limit=27)

bitsandbytes/backends/triton/triton_kernels.py
==============================================
dequant_8bit_kernel(a_ptr, c_ptr, quant_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK, SPLIT_SIZE) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=28, limit=31)
dequant_int8_blockwise(A_nf4, quant_state_code, absmax, out, quant_blocksize) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=61, limit=22)
quantize_blockwise_kernel(A_ptr, code_ptr, absmax_ptr, out_ptr, n_elements, BLOCK_SIZE, CODE_SIZE, SPLIT_NUM_BLOCKS) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=95, limit=51)
quantize_blockwise_triton(A, blocksize, code, blocks, absmax, quantized_out) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=148, limit=18)
quantize_fp4_blockwise_kernel(A_ptr, absmax_ptr, out_ptr, n_elements, BLOCK_SIZE, SPLIT_NUM_BLOCKS) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=181, limit=51)
quantize_nf4_blockwise_kernel(A_ptr, absmax_ptr, out_ptr, n_elements, BLOCK_SIZE, SPLIT_NUM_BLOCKS) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=247, limit=67)
quantize_4bit_blockwise_triton(A, blocksize, quant_type, blocks, absmax, num_elements, quantized_out) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=316, limit=23)
dequant_4bit_body_util(a, offsets, quant_ptr, absmax_ptr, n_elems, QUANT_BLOCK) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=342, limit=18)
dequantize_fp4_tree(val, absmax) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=364, limit=21)
dequant_fp4_body_util(a, offsets, absmax_ptr, n_elems, QUANT_BLOCK) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=388, limit=12)
dequantize_nf4_tree(val) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=404, limit=37)
dequant_nf4_body_util(a, offsets, absmax_ptr, n_elems, QUANT_BLOCK) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=444, limit=13)
dequant_4bit_kernel(a_ptr, c_ptr, quant_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK, SPLIT_SIZE) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=493, limit=23)
dequant_fp4_kernel(a_ptr, c_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK, SPLIT_SIZE) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=530, limit=22)
dequant_nf4_kernel(a_ptr, c_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK, SPLIT_SIZE) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=566, limit=22)
_dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=590, limit=22)
_dequantize_4bit_impl_passing_code(A, absmax, blocksize, code, dtype, out) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=614, limit=15)
quantize_4bit_blockwise_kernel(A_ptr, code_ptr, absmax_ptr, out_ptr, n_elements, BLOCK_SIZE, CODE_SIZE, SPLIT_NUM_BLOCKS) # Read(file_path=bitsandbytes/backends/triton/triton_kernels.py, offset=655, limit=59)

bitsandbytes/backends/utils.py
==============================
get_gaudi_sw_version() # Read(file_path=bitsandbytes/backends/utils.py, offset=76, limit=15)

bitsandbytes/backends/xpu/ops.py
================================
_(A, B) # Read(file_path=bitsandbytes/backends/xpu/ops.py, offset=14, limit=5)
_(A, absmax, blocksize, shape, dtype) # Read(file_path=bitsandbytes/backends/xpu/ops.py, offset=25, limit=8)
_(A, absmax, code, blocksize, dtype) # Read(file_path=bitsandbytes/backends/xpu/ops.py, offset=35, limit=21)

bitsandbytes/cextension.py
==========================
get_cuda_bnb_library_path(cuda_specs) # Read(file_path=bitsandbytes/cextension.py, offset=17, limit=21)
BNBNativeLibrary() # Read(file_path=bitsandbytes/cextension.py, offset=40, limit=24)
__init__(self, lib) # Read(file_path=bitsandbytes/cextension.py, offset=44, limit=2)
__getattr__(self, name) # Read(file_path=bitsandbytes/cextension.py, offset=48, limit=13)
throw_on_call(*args, **kwargs) # Read(file_path=bitsandbytes/cextension.py, offset=54, limit=5)
__getitem__(self, item) # Read(file_path=bitsandbytes/cextension.py, offset=62, limit=2)
CudaBNBNativeLibrary(BNBNativeLibrary) # Read(file_path=bitsandbytes/cextension.py, offset=66, limit=8)
__init__(self, lib) # Read(file_path=bitsandbytes/cextension.py, offset=69, limit=5)
get_available_cuda_binary_versions() # Read(file_path=bitsandbytes/cextension.py, offset=76, limit=12)
parse_cuda_version(version_str) # Read(file_path=bitsandbytes/cextension.py, offset=90, limit=5)
ErrorHandlerMockBNBNativeLibrary(BNBNativeLibrary) # Read(file_path=bitsandbytes/cextension.py, offset=97, limit=160)
__init__(self, error_msg) # Read(file_path=bitsandbytes/cextension.py, offset=118, limit=25)
_format_lib_error_message(self, available_versions, user_cuda_version, original_error, requested_version) # Read(file_path=bitsandbytes/cextension.py, offset=144, limit=75)
_format_dependency_error(self) # Read(file_path=bitsandbytes/cextension.py, offset=220, limit=26)
__getattr__(self, name) # Read(file_path=bitsandbytes/cextension.py, offset=247, limit=7)
throw_on_call(*args, **kwargs) # Read(file_path=bitsandbytes/cextension.py, offset=250, limit=2)
__getitem__(self, name) # Read(file_path=bitsandbytes/cextension.py, offset=255, limit=2)
get_native_library() # Read(file_path=bitsandbytes/cextension.py, offset=259, limit=28)

bitsandbytes/cuda_specs.py
==========================
CUDASpecs() # Read(file_path=bitsandbytes/cuda_specs.py, offset=9, limit=8)
has_imma(self) # Read(file_path=bitsandbytes/cuda_specs.py, offset=15, limit=2)
get_compute_capabilities() # Read(file_path=bitsandbytes/cuda_specs.py, offset=19, limit=2)
get_cuda_version_tuple() # Read(file_path=bitsandbytes/cuda_specs.py, offset=24, limit=16)
get_cuda_version_string() # Read(file_path=bitsandbytes/cuda_specs.py, offset=42, limit=7)
get_cuda_specs() # Read(file_path=bitsandbytes/cuda_specs.py, offset=51, limit=25)

bitsandbytes/diagnostics/cuda.py
================================
find_cuda_libraries_in_path_list(paths_list_candidate) # Read(file_path=bitsandbytes/diagnostics/cuda.py, offset=43, limit=20)
is_relevant_candidate_env_var(env_var, value) # Read(file_path=bitsandbytes/diagnostics/cuda.py, offset=65, limit=11)
get_potentially_lib_path_containing_env_vars() # Read(file_path=bitsandbytes/diagnostics/cuda.py, offset=78, limit=2)
find_cudart_libraries() # Read(file_path=bitsandbytes/diagnostics/cuda.py, offset=82, limit=22)
print_cuda_diagnostics(cuda_specs) # Read(file_path=bitsandbytes/diagnostics/cuda.py, offset=106, limit=23)
print_cuda_runtime_diagnostics() # Read(file_path=bitsandbytes/diagnostics/cuda.py, offset=131, limit=25)

bitsandbytes/diagnostics/main.py
================================
sanity_check() # Read(file_path=bitsandbytes/diagnostics/main.py, offset=29, limit=13)
get_package_version(name) # Read(file_path=bitsandbytes/diagnostics/main.py, offset=44, limit=6)
show_environment() # Read(file_path=bitsandbytes/diagnostics/main.py, offset=52, limit=18)
main() # Read(file_path=bitsandbytes/diagnostics/main.py, offset=72, limit=46)

bitsandbytes/diagnostics/utils.py
=================================
print_header(txt, width, filler) # Read(file_path=bitsandbytes/diagnostics/utils.py, offset=6, limit=3)
print_dedented(text) # Read(file_path=bitsandbytes/diagnostics/utils.py, offset=11, limit=2)

bitsandbytes/functional.py
==========================
GlobalPageManager() # Read(file_path=bitsandbytes/functional.py, offset=119, limit=22)
__init__(self) # Read(file_path=bitsandbytes/functional.py, offset=122, limit=2)
initialize(self) # Read(file_path=bitsandbytes/functional.py, offset=125, limit=2)
get_instance(cls) # Read(file_path=bitsandbytes/functional.py, offset=129, limit=5)
prefetch_all(self, to_cpu) # Read(file_path=bitsandbytes/functional.py, offset=135, limit=6)
CUBLAS_Context() # Read(file_path=bitsandbytes/functional.py, offset=143, limit=23)
__init__(self) # Read(file_path=bitsandbytes/functional.py, offset=146, limit=2)
initialize(self) # Read(file_path=bitsandbytes/functional.py, offset=149, limit=2)
get_instance(cls) # Read(file_path=bitsandbytes/functional.py, offset=153, limit=5)
get_context(self, device) # Read(file_path=bitsandbytes/functional.py, offset=159, limit=7)
Cusparse_Context() # Read(file_path=bitsandbytes/functional.py, offset=168, limit=15)
__init__(self) # Read(file_path=bitsandbytes/functional.py, offset=171, limit=2)
initialize(self) # Read(file_path=bitsandbytes/functional.py, offset=174, limit=2)
get_instance(cls) # Read(file_path=bitsandbytes/functional.py, offset=178, limit=5)
_cuda_device_of(a) # Read(file_path=bitsandbytes/functional.py, offset=193, limit=2)
_cuda_device_of(a) # Read(file_path=bitsandbytes/functional.py, offset=198, limit=2)
get_paged(*shape, dtype, device) # Read(file_path=bitsandbytes/functional.py, offset=202, limit=9)
prefetch_tensor(A, to_cpu) # Read(file_path=bitsandbytes/functional.py, offset=213, limit=8)
elementwise_func(func_name, A, B, value, prefetch) # Read(file_path=bitsandbytes/functional.py, offset=223, limit=25)
fill(A, value, device, prefetch) # Read(file_path=bitsandbytes/functional.py, offset=250, limit=2)
_mul(A, B, device) # Read(file_path=bitsandbytes/functional.py, offset=254, limit=2)
create_linear_map(signed, total_bits, add_zero) # Read(file_path=bitsandbytes/functional.py, offset=258, limit=17)
create_normal_map(offset, use_extra_value) # Read(file_path=bitsandbytes/functional.py, offset=277, limit=27)
create_fp8_map(signed, exponent_bits, precision_bits, total_bits) # Read(file_path=bitsandbytes/functional.py, offset=306, limit=41)
create_dynamic_map(signed, max_exponent_bits, total_bits) # Read(file_path=bitsandbytes/functional.py, offset=349, limit=53)
is_on_gpu(tensors) # Read(file_path=bitsandbytes/functional.py, offset=404, limit=34)
_get_tensor_stream(tensor) # Read(file_path=bitsandbytes/functional.py, offset=440, limit=3)
get_ptr(A) # Read(file_path=bitsandbytes/functional.py, offset=445, limit=13)
QuantState() # Read(file_path=bitsandbytes/functional.py, offset=460, limit=175)
__init__(self, absmax, shape, code, blocksize, quant_type, dtype, offset, state2) # Read(file_path=bitsandbytes/functional.py, offset=480, limit=20)
__getitem__(self, idx) # Read(file_path=bitsandbytes/functional.py, offset=501, limit=19)
from_dict(cls, qs_dict, device) # Read(file_path=bitsandbytes/functional.py, offset=522, limit=49)
as_dict(self, packed) # Read(file_path=bitsandbytes/functional.py, offset=572, limit=31)
to(self, device) # Read(file_path=bitsandbytes/functional.py, offset=604, limit=8)
__eq__(self, other) # Read(file_path=bitsandbytes/functional.py, offset=613, limit=22)
quantize_blockwise(A, code, absmax, out, blocksize, nested) # Read(file_path=bitsandbytes/functional.py, offset=637, limit=69)
dequantize_blockwise(A, quant_state, absmax, code, out, blocksize, nested) # Read(file_path=bitsandbytes/functional.py, offset=708, limit=75)
get_4bit_type(typename, device, blocksize) # Read(file_path=bitsandbytes/functional.py, offset=785, limit=80)
quantize_fp4(A, absmax, out, blocksize, compress_statistics, quant_storage) # Read(file_path=bitsandbytes/functional.py, offset=867, limit=9)
quantize_nf4(A, absmax, out, blocksize, compress_statistics, quant_storage) # Read(file_path=bitsandbytes/functional.py, offset=878, limit=9)
quantize_4bit(A, absmax, out, blocksize, compress_statistics, quant_type, quant_storage) # Read(file_path=bitsandbytes/functional.py, offset=889, limit=75)
dequantize_fp4(A, quant_state, absmax, out, blocksize) # Read(file_path=bitsandbytes/functional.py, offset=966, limit=8)
dequantize_nf4(A, quant_state, absmax, out, blocksize) # Read(file_path=bitsandbytes/functional.py, offset=976, limit=8)
dequantize_4bit(A, quant_state, absmax, out, blocksize, quant_type) # Read(file_path=bitsandbytes/functional.py, offset=986, limit=81)
quantize(A, code, out) # Read(file_path=bitsandbytes/functional.py, offset=1070, limit=17)
dequantize(A, state, absmax, code, out) # Read(file_path=bitsandbytes/functional.py, offset=1090, limit=18)
quantize_no_absmax(A, code, out) # Read(file_path=bitsandbytes/functional.py, offset=1111, limit=28)
dequantize_no_absmax(A, code, out) # Read(file_path=bitsandbytes/functional.py, offset=1142, limit=29)
optimizer_update_32bit(optimizer_name, g, p, state1, beta1, eps, step, lr, state2, beta2, beta3, alpha, weight_decay, gnorm_scale, unorm_vec, max_unorm, skip_zeros) # Read(file_path=bitsandbytes/functional.py, offset=1173, limit=101)
optimizer_update_8bit(optimizer_name, g, p, state1, state2, beta1, beta2, eps, step, lr, qmap1, qmap2, max1, max2, new_max1, new_max2, weight_decay, gnorm_scale, unorm_vec, max_unorm) # Read(file_path=bitsandbytes/functional.py, offset=1281, limit=130)
optimizer_update_8bit_blockwise(optimizer_name, g, p, state1, state2, beta1, beta2, beta3, alpha, eps, step, lr, qmap1, qmap2, absmax1, absmax2, weight_decay, gnorm_scale, skip_zeros) # Read(file_path=bitsandbytes/functional.py, offset=1413, limit=62)
percentile_clipping(grad, gnorm_vec, step, percentile) # Read(file_path=bitsandbytes/functional.py, offset=1478, limit=39)
check_matmul(A, B, out, transposed_A, transposed_B, expected_type) # Read(file_path=bitsandbytes/functional.py, offset=1519, limit=82)
gemv_4bit(A, B, out, transposed_A, transposed_B, state) # Read(file_path=bitsandbytes/functional.py, offset=1603, limit=54)
igemm(A, B, out, transposed_A, transposed_B) # Read(file_path=bitsandbytes/functional.py, offset=1659, limit=99)
batched_igemm(A, B, out, transposed_A, transposed_B) # Read(file_path=bitsandbytes/functional.py, offset=1760, limit=94)
int8_linear_matmul(A, B, out, dtype) # Read(file_path=bitsandbytes/functional.py, offset=1856, limit=24)
int8_mm_dequant(A, row_stats, col_stats, out, bias) # Read(file_path=bitsandbytes/functional.py, offset=1882, limit=26)
get_colrow_absmax(A, row_stats, col_stats, nnz_block_ptr, threshold) # Read(file_path=bitsandbytes/functional.py, offset=1911, limit=55)
get_row_absmax(A, threshold) # Read(file_path=bitsandbytes/functional.py, offset=1969, limit=35)
COOSparseTensor() # Read(file_path=bitsandbytes/functional.py, offset=2006, limit=17)
__init__(self, rows, cols, nnz, rowidx, colidx, values) # Read(file_path=bitsandbytes/functional.py, offset=2007, limit=16)
CSRSparseTensor() # Read(file_path=bitsandbytes/functional.py, offset=2025, limit=15)
__init__(self, rows, cols, nnz, rowptr, colidx, values) # Read(file_path=bitsandbytes/functional.py, offset=2026, limit=14)
CSCSparseTensor() # Read(file_path=bitsandbytes/functional.py, offset=2042, limit=15)
__init__(self, rows, cols, nnz, colptr, rowidx, values) # Read(file_path=bitsandbytes/functional.py, offset=2043, limit=14)
coo2csr(cooA) # Read(file_path=bitsandbytes/functional.py, offset=2059, limit=7)
coo2csc(cooA) # Read(file_path=bitsandbytes/functional.py, offset=2068, limit=10)
coo_zeros(rows, cols, nnz, device, dtype) # Read(file_path=bitsandbytes/functional.py, offset=2080, limit=5)
int8_double_quant(A, col_stats, row_stats, out_col, out_row, threshold) # Read(file_path=bitsandbytes/functional.py, offset=2087, limit=49)
int8_vectorwise_dequant(A, stats) # Read(file_path=bitsandbytes/functional.py, offset=2138, limit=12)
int8_vectorwise_quant(A, threshold) # Read(file_path=bitsandbytes/functional.py, offset=2152, limit=19)
spmm_coo(cooA, B, out) # Read(file_path=bitsandbytes/functional.py, offset=2173, limit=65)
spmm_coo_very_sparse(cooA, B, dequant_stats, out) # Read(file_path=bitsandbytes/functional.py, offset=2240, limit=80)
_enable_ipex_fusion(linear, x) # Read(file_path=bitsandbytes/functional.py, offset=2325, limit=44)

bitsandbytes/nn/modules.py
==========================
StableEmbedding(torch.nn.Embedding) # Read(file_path=bitsandbytes/nn/modules.py, offset=25, limit=104)
__init__(self, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, device, dtype) # Read(file_path=bitsandbytes/nn/modules.py, offset=51, limit=46)
reset_parameters(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=98, limit=3)
_fill_padding_idx_with_zero(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=109, limit=4)
forward(self, input) # Read(file_path=bitsandbytes/nn/modules.py, offset=114, limit=15)
Embedding(torch.nn.Embedding) # Read(file_path=bitsandbytes/nn/modules.py, offset=131, limit=77)
__init__(self, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=136, limit=43)
reset_parameters(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=180, limit=3)
_fill_padding_idx_with_zero(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=191, limit=4)
forward(self, input) # Read(file_path=bitsandbytes/nn/modules.py, offset=196, limit=12)
Params4bit(torch.nn.Parameter) # Read(file_path=bitsandbytes/nn/modules.py, offset=210, limit=144)
__new__(cls, data, requires_grad, quant_state, blocksize, compress_statistics, quant_type, quant_storage, module, bnb_quantized) # Read(file_path=bitsandbytes/nn/modules.py, offset=211, limit=25)
__getstate__(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=237, limit=5)
__setstate__(self, state) # Read(file_path=bitsandbytes/nn/modules.py, offset=243, limit=10)
__deepcopy__(self, memo) # Read(file_path=bitsandbytes/nn/modules.py, offset=254, limit=7)
__copy__(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=262, limit=5)
from_prequantized(cls, data, quantized_stats, requires_grad, device, module, **kwargs) # Read(file_path=bitsandbytes/nn/modules.py, offset=269, limit=24)
_quantize(self, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=294, limit=15)
cpu(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=310, limit=2)
cuda(self, device, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=313, limit=2)
xpu(self, device, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=316, limit=2)
to(self, device, dtype, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=320, limit=6)
to(self, dtype, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=328, limit=1)
to(self, tensor, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=331, limit=1)
to(self, *args, **kwargs) # Read(file_path=bitsandbytes/nn/modules.py, offset=333, limit=21)
fix_4bit_weight_quant_state_from_module(module) # Read(file_path=bitsandbytes/nn/modules.py, offset=356, limit=15)
Linear4bit(nn.Linear) # Read(file_path=bitsandbytes/nn/modules.py, offset=373, limit=149)
__init__(self, input_features, output_features, bias, compute_dtype, compress_statistics, quant_type, quant_storage, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=406, limit=37)
set_compute_type(self, x) # Read(file_path=bitsandbytes/nn/modules.py, offset=444, limit=19)
_save_to_state_dict(self, destination, prefix, keep_vars) # Read(file_path=bitsandbytes/nn/modules.py, offset=464, limit=22)
set_ipex_linear(self, x) # Read(file_path=bitsandbytes/nn/modules.py, offset=487, limit=9)
forward(self, x) # Read(file_path=bitsandbytes/nn/modules.py, offset=497, limit=25)
LinearFP4(Linear4bit) # Read(file_path=bitsandbytes/nn/modules.py, offset=524, limit=34)
__init__(self, input_features, output_features, bias, compute_dtype, compress_statistics, quant_storage, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=529, limit=29)
LinearNF4(Linear4bit) # Read(file_path=bitsandbytes/nn/modules.py, offset=560, limit=41)
__init__(self, input_features, output_features, bias, compute_dtype, compress_statistics, quant_storage, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=572, limit=29)
Int8Params(torch.nn.Parameter) # Read(file_path=bitsandbytes/nn/modules.py, offset=603, limit=83)
__new__(cls, data, requires_grad, has_fp16_weights, CB, SCB) # Read(file_path=bitsandbytes/nn/modules.py, offset=604, limit=15)
_quantize(self, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=620, limit=12)
cpu(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=633, limit=2)
cuda(self, device, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=636, limit=2)
xpu(self, device, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=639, limit=2)
__deepcopy__(self, memo) # Read(file_path=bitsandbytes/nn/modules.py, offset=642, limit=11)
to(self, device, dtype, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=655, limit=6)
to(self, dtype, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=663, limit=1)
to(self, tensor, non_blocking) # Read(file_path=bitsandbytes/nn/modules.py, offset=666, limit=1)
to(self, *args, **kwargs) # Read(file_path=bitsandbytes/nn/modules.py, offset=668, limit=18)
maybe_rearrange_weight(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) # Read(file_path=bitsandbytes/nn/modules.py, offset=688, limit=19)
Embedding8bit(nn.Embedding) # Read(file_path=bitsandbytes/nn/modules.py, offset=709, limit=45)
__init__(self, num_embeddings, embedding_dim, device, dtype) # Read(file_path=bitsandbytes/nn/modules.py, offset=729, limit=5)
_save_to_state_dict(self, destination, prefix, keep_vars) # Read(file_path=bitsandbytes/nn/modules.py, offset=735, limit=2)
forward(self, input) # Read(file_path=bitsandbytes/nn/modules.py, offset=738, limit=16)
Embedding4bit(nn.Embedding) # Read(file_path=bitsandbytes/nn/modules.py, offset=756, limit=98)
__init__(self, num_embeddings, embedding_dim, dtype, quant_type, quant_storage, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=777, limit=28)
_forward_with_partial_dequantize(self, input) # Read(file_path=bitsandbytes/nn/modules.py, offset=806, limit=32)
_save_to_state_dict(self, destination, prefix, keep_vars) # Read(file_path=bitsandbytes/nn/modules.py, offset=839, limit=2)
forward(self, input) # Read(file_path=bitsandbytes/nn/modules.py, offset=842, limit=12)
EmbeddingFP4(Embedding4bit) # Read(file_path=bitsandbytes/nn/modules.py, offset=856, limit=17)
__init__(self, num_embeddings, embedding_dim, dtype, quant_storage, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=857, limit=16)
EmbeddingNF4(Embedding4bit) # Read(file_path=bitsandbytes/nn/modules.py, offset=875, limit=17)
__init__(self, num_embeddings, embedding_dim, dtype, quant_storage, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=876, limit=16)
Linear8bitLt(nn.Linear) # Read(file_path=bitsandbytes/nn/modules.py, offset=894, limit=150)
__init__(self, input_features, output_features, bias, has_fp16_weights, threshold, index, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=926, limit=33)
_save_to_state_dict(self, destination, prefix, keep_vars) # Read(file_path=bitsandbytes/nn/modules.py, offset=960, limit=23)
_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) # Read(file_path=bitsandbytes/nn/modules.py, offset=984, limit=38)
init_8bit_state(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=1023, limit=5)
forward(self, x) # Read(file_path=bitsandbytes/nn/modules.py, offset=1029, limit=15)
OutlierAwareLinear(nn.Linear) # Read(file_path=bitsandbytes/nn/modules.py, offset=1046, limit=25)
__init__(self, input_features, output_features, bias, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=1047, limit=4)
forward_with_outliers(self, x, outlier_idx) # Read(file_path=bitsandbytes/nn/modules.py, offset=1052, limit=2)
quantize_weight(self, w, outlier_idx) # Read(file_path=bitsandbytes/nn/modules.py, offset=1055, limit=2)
forward(self, x) # Read(file_path=bitsandbytes/nn/modules.py, offset=1058, limit=13)
SwitchBackLinearBnb(nn.Linear) # Read(file_path=bitsandbytes/nn/modules.py, offset=1073, limit=37)
__init__(self, input_features, output_features, bias, has_fp16_weights, memory_efficient_backward, threshold, index, device) # Read(file_path=bitsandbytes/nn/modules.py, offset=1074, limit=22)
init_8bit_state(self) # Read(file_path=bitsandbytes/nn/modules.py, offset=1097, limit=5)
forward(self, x) # Read(file_path=bitsandbytes/nn/modules.py, offset=1103, limit=7)

bitsandbytes/nn/triton_based_modules.py
=======================================
_switchback_global(torch.autograd.Function) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=24, limit=42)
forward(ctx, X_3D, W, bias) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=26, limit=14)
backward(ctx, G_3D) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=42, limit=24)
_switchback_vectorrize(torch.autograd.Function) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=68, limit=40)
forward(ctx, X_3D, W, bias) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=70, limit=13)
backward(ctx, G_3D) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=85, limit=23)
_switchback_global_mem_efficient(torch.autograd.Function) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=110, limit=42)
forward(ctx, X_3D, W, bias) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=112, limit=16)
backward(ctx, G_3D) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=130, limit=22)
SwitchBackLinear(nn.Linear) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=154, limit=71)
__init__(self, in_features, out_features, bias, device, dtype, vector_wise_quantization, mem_efficient) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=155, limit=28)
prepare_for_eval(self) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=184, limit=18)
forward(self, x) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=203, limit=22)
StandardLinearFunction(torch.autograd.Function) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=233, limit=27)
forward(ctx, input, weight, bias) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=235, limit=8)
backward(ctx, grad_output_3D) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=245, limit=15)
StandardLinear(nn.Linear) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=262, limit=3)
forward(self, x) # Read(file_path=bitsandbytes/nn/triton_based_modules.py, offset=263, limit=2)

bitsandbytes/optim/adagrad.py
=============================
Adagrad(Optimizer1State) # Read(file_path=bitsandbytes/optim/adagrad.py, offset=8, limit=65)
__init__(self, params, lr, lr_decay, weight_decay, initial_accumulator_value, eps, optim_bits, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/adagrad.py, offset=9, limit=64)
Adagrad8bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/adagrad.py, offset=75, limit=66)
__init__(self, params, lr, lr_decay, weight_decay, initial_accumulator_value, eps, optim_bits, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/adagrad.py, offset=76, limit=65)
Adagrad32bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/adagrad.py, offset=143, limit=65)
__init__(self, params, lr, lr_decay, weight_decay, initial_accumulator_value, eps, optim_bits, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/adagrad.py, offset=144, limit=64)

bitsandbytes/optim/adam.py
==========================
Adam(Optimizer2State) # Read(file_path=bitsandbytes/optim/adam.py, offset=9, limit=59)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/adam.py, offset=10, limit=58)
Adam8bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/adam.py, offset=70, limit=70)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/adam.py, offset=71, limit=69)
Adam32bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/adam.py, offset=142, limit=59)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/adam.py, offset=143, limit=58)
PagedAdam(Optimizer2State) # Read(file_path=bitsandbytes/optim/adam.py, offset=203, limit=59)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/adam.py, offset=204, limit=58)
PagedAdam8bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/adam.py, offset=264, limit=70)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/adam.py, offset=265, limit=69)
PagedAdam32bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/adam.py, offset=336, limit=59)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/adam.py, offset=337, limit=58)

bitsandbytes/optim/adamw.py
===========================
AdamW(Optimizer2State) # Read(file_path=bitsandbytes/optim/adamw.py, offset=9, limit=59)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/adamw.py, offset=10, limit=58)
AdamW8bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/adamw.py, offset=70, limit=70)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/adamw.py, offset=71, limit=69)
AdamW32bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/adamw.py, offset=142, limit=59)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/adamw.py, offset=143, limit=58)
PagedAdamW(Optimizer2State) # Read(file_path=bitsandbytes/optim/adamw.py, offset=203, limit=58)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/adamw.py, offset=204, limit=57)
PagedAdamW8bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/adamw.py, offset=263, limit=69)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/adamw.py, offset=264, limit=68)
PagedAdamW32bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/adamw.py, offset=334, limit=58)
__init__(self, params, lr, betas, eps, weight_decay, amsgrad, optim_bits, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/adamw.py, offset=335, limit=57)

bitsandbytes/optim/ademamix.py
==============================
_ReferenceAdEMAMix(torch.optim.Optimizer) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=11, limit=94)
__init__(self, params, lr, betas, alpha, eps, weight_decay, t_beta3, t_alpha) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=16, limit=16)
step(self, closure) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=34, limit=71)
AdEMAMix(Optimizer2State) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=107, limit=165)
__init__(self, params, lr, betas, alpha, t_alpha, t_beta3, eps, weight_decay, optim_bits, min_8bit_size, is_paged) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=108, limit=31)
init_state(self, group, p, gindex, pindex) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=141, limit=37)
update_step(self, group, p, gindex, pindex) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=180, limit=83)
_get_state_double_buffer(self, p, dtype) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=264, limit=8)
AdEMAMix8bit(AdEMAMix) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=274, limit=27)
__init__(self, params, lr, betas, alpha, t_alpha, t_beta3, eps, weight_decay, min_8bit_size, is_paged) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=275, limit=26)
PagedAdEMAMix8bit(AdEMAMix8bit) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=303, limit=25)
__init__(self, params, lr, betas, alpha, t_alpha, t_beta3, eps, weight_decay, min_8bit_size) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=304, limit=24)
PagedAdEMAMix(AdEMAMix) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=330, limit=27)
__init__(self, params, lr, betas, alpha, t_alpha, t_beta3, eps, weight_decay, optim_bits, min_8bit_size) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=331, limit=26)
AdEMAMix32bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=359, limit=31)
__init__(self, params, lr, betas, alpha, t_alpha, t_beta3, eps, weight_decay, min_8bit_size, is_paged) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=360, limit=30)
PagedAdEMAMix32bit(AdEMAMix32bit) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=392, limit=25)
__init__(self, params, lr, betas, alpha, t_alpha, t_beta3, eps, weight_decay, min_8bit_size) # Read(file_path=bitsandbytes/optim/ademamix.py, offset=393, limit=24)

bitsandbytes/optim/lamb.py
==========================
LAMB(Optimizer2State) # Read(file_path=bitsandbytes/optim/lamb.py, offset=8, limit=65)
__init__(self, params, lr, bias_correction, betas, eps, weight_decay, amsgrad, adam_w_mode, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, max_unorm) # Read(file_path=bitsandbytes/optim/lamb.py, offset=9, limit=64)
LAMB8bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/lamb.py, offset=75, limit=62)
__init__(self, params, lr, bias_correction, betas, eps, weight_decay, amsgrad, adam_w_mode, args, min_8bit_size, percentile_clipping, block_wise, max_unorm) # Read(file_path=bitsandbytes/optim/lamb.py, offset=76, limit=61)
LAMB32bit(Optimizer2State) # Read(file_path=bitsandbytes/optim/lamb.py, offset=139, limit=62)
__init__(self, params, lr, bias_correction, betas, eps, weight_decay, amsgrad, adam_w_mode, args, min_8bit_size, percentile_clipping, block_wise, max_unorm) # Read(file_path=bitsandbytes/optim/lamb.py, offset=140, limit=61)

bitsandbytes/optim/lars.py
==========================
LARS(Optimizer1State) # Read(file_path=bitsandbytes/optim/lars.py, offset=11, limit=58)
__init__(self, params, lr, momentum, dampening, weight_decay, nesterov, optim_bits, args, min_8bit_size, percentile_clipping, max_unorm) # Read(file_path=bitsandbytes/optim/lars.py, offset=12, limit=57)
LARS8bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/lars.py, offset=71, limit=55)
__init__(self, params, lr, momentum, dampening, weight_decay, nesterov, args, min_8bit_size, percentile_clipping, max_unorm) # Read(file_path=bitsandbytes/optim/lars.py, offset=72, limit=54)
LARS32bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/lars.py, offset=128, limit=55)
__init__(self, params, lr, momentum, dampening, weight_decay, nesterov, args, min_8bit_size, percentile_clipping, max_unorm) # Read(file_path=bitsandbytes/optim/lars.py, offset=129, limit=54)
PytorchLARS(Optimizer) # Read(file_path=bitsandbytes/optim/lars.py, offset=185, limit=93)
__init__(self, params, lr, momentum, dampening, weight_decay, nesterov, max_unorm) # Read(file_path=bitsandbytes/optim/lars.py, offset=186, limit=28)
__setstate__(self, state) # Read(file_path=bitsandbytes/optim/lars.py, offset=215, limit=4)
step(self, closure) # Read(file_path=bitsandbytes/optim/lars.py, offset=221, limit=57)

bitsandbytes/optim/lion.py
==========================
Lion(Optimizer1State) # Read(file_path=bitsandbytes/optim/lion.py, offset=8, limit=53)
__init__(self, params, lr, betas, weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/lion.py, offset=9, limit=52)
Lion8bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/lion.py, offset=63, limit=50)
__init__(self, params, lr, betas, weight_decay, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/lion.py, offset=64, limit=49)
Lion32bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/lion.py, offset=115, limit=50)
__init__(self, params, lr, betas, weight_decay, args, min_8bit_size, percentile_clipping, block_wise, is_paged) # Read(file_path=bitsandbytes/optim/lion.py, offset=116, limit=49)
PagedLion(Optimizer1State) # Read(file_path=bitsandbytes/optim/lion.py, offset=167, limit=50)
__init__(self, params, lr, betas, weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/lion.py, offset=168, limit=49)
PagedLion8bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/lion.py, offset=219, limit=49)
__init__(self, params, lr, betas, weight_decay, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/lion.py, offset=220, limit=48)
PagedLion32bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/lion.py, offset=270, limit=49)
__init__(self, params, lr, betas, weight_decay, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/lion.py, offset=271, limit=48)

bitsandbytes/optim/optimizer.py
===============================
MockArgs() # Read(file_path=bitsandbytes/optim/optimizer.py, offset=15, limit=4)
__init__(self, initial_data) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=16, limit=3)
GlobalOptimManager() # Read(file_path=bitsandbytes/optim/optimizer.py, offset=21, limit=89)
__init__(self) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=28, limit=2)
initialize(self) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=31, limit=6)
get_instance(cls) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=39, limit=5)
register_parameters(self, params) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=45, limit=9)
override_config(self, parameters, key, value, key_value_dict) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=55, limit=52)
register_module_override(self, module, param_name, config) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=108, limit=2)
Optimizer8bit(torch.optim.Optimizer) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=112, limit=233)
__init__(self, params, defaults, optim_bits, is_paged) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=113, limit=36)
fill_qmap(self) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=150, limit=3)
__setstate__(self, state) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=154, limit=2)
load_state_dict(self, state_dict, move_to_device) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=157, limit=73)
cast(param, value) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=190, limit=21)
update_group(group, new_group) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=224, limit=3)
to_gpu(self) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=231, limit=10)
check_overrides(self) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=242, limit=18)
step(self, closure) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=262, limit=37)
get_config(self, gindex, pindex, group) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=300, limit=19)
init_state(self, group, p, gindex, pindex) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=320, limit=2)
update_step(self, group, p, gindex, pindex) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=323, limit=2)
get_state_buffer(self, p, dtype) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=326, limit=9)
prefetch_state(self, p) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=336, limit=9)
Optimizer2State(Optimizer8bit) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=347, limit=243)
__init__(self, optimizer_name, params, lr, betas, eps, weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, max_unorm, skip_zeros, is_paged, alpha, t_alpha, t_beta3) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=348, limit=95)
init_state(self, group, p, gindex, pindex) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=445, limit=50)
update_step(self, group, p, gindex, pindex) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=497, limit=93)
Optimizer1State(Optimizer8bit) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=592, limit=215)
__init__(self, optimizer_name, params, lr, betas, eps, weight_decay, optim_bits, args, min_8bit_size, percentile_clipping, block_wise, max_unorm, skip_zeros, is_paged) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=593, limit=77)
init_state(self, group, p, gindex, pindex) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=672, limit=42)
update_step(self, group, p, gindex, pindex) # Read(file_path=bitsandbytes/optim/optimizer.py, offset=716, limit=91)

bitsandbytes/optim/rmsprop.py
=============================
RMSprop(Optimizer1State) # Read(file_path=bitsandbytes/optim/rmsprop.py, offset=8, limit=62)
__init__(self, params, lr, alpha, eps, weight_decay, momentum, centered, optim_bits, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/rmsprop.py, offset=9, limit=61)
RMSprop8bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/rmsprop.py, offset=72, limit=61)
__init__(self, params, lr, alpha, eps, weight_decay, momentum, centered, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/rmsprop.py, offset=73, limit=60)
RMSprop32bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/rmsprop.py, offset=135, limit=62)
__init__(self, params, lr, alpha, eps, weight_decay, momentum, centered, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/rmsprop.py, offset=136, limit=61)

bitsandbytes/optim/sgd.py
=========================
SGD(Optimizer1State) # Read(file_path=bitsandbytes/optim/sgd.py, offset=8, limit=57)
__init__(self, params, lr, momentum, dampening, weight_decay, nesterov, optim_bits, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/sgd.py, offset=9, limit=56)
SGD8bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/sgd.py, offset=67, limit=54)
__init__(self, params, lr, momentum, dampening, weight_decay, nesterov, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/sgd.py, offset=68, limit=53)
SGD32bit(Optimizer1State) # Read(file_path=bitsandbytes/optim/sgd.py, offset=123, limit=54)
__init__(self, params, lr, momentum, dampening, weight_decay, nesterov, args, min_8bit_size, percentile_clipping, block_wise) # Read(file_path=bitsandbytes/optim/sgd.py, offset=124, limit=53)

bitsandbytes/research/autograd/_functions.py
============================================
prod(iterable) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=13, limit=2)
MatMulFP8Mixed(torch.autograd.Function) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=17, limit=82)
forward(ctx, A, B, out, fw_code, bw_code, bsz, bsz2) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=22, limit=40)
backward(ctx, grad_output) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=64, limit=35)
MatMulFP8Global(torch.autograd.Function) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=101, limit=82)
forward(ctx, A, B, out, fw_code, bw_code, bsz, bsz2) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=106, limit=40)
backward(ctx, grad_output) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=148, limit=35)
SwitchBackBnb(torch.autograd.Function) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=185, limit=153)
forward(ctx, A, B, out, bias, state) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=187, limit=115)
backward(ctx, grad_output) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=304, limit=34)
get_block_sizes(input_matrix, weight_matrix) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=340, limit=15)
matmul_fp8_global(A, B, fw_code, bw_code, out, bsz, bsz2) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=357, limit=12)
matmul_fp8_mixed(A, B, fw_code, bw_code, out, bsz, bsz2) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=371, limit=12)
switchback_bnb(A, B, out, state, threshold, bias) # Read(file_path=bitsandbytes/research/autograd/_functions.py, offset=385, limit=12)

bitsandbytes/research/nn/modules.py
===================================
LinearFP8Mixed(nn.Linear) # Read(file_path=bitsandbytes/research/nn/modules.py, offset=11, limit=32)
__init__(self, input_features, output_features, bias) # Read(file_path=bitsandbytes/research/nn/modules.py, offset=12, limit=13)
forward(self, x) # Read(file_path=bitsandbytes/research/nn/modules.py, offset=26, limit=17)
LinearFP8Global(nn.Linear) # Read(file_path=bitsandbytes/research/nn/modules.py, offset=45, limit=32)
__init__(self, input_features, output_features, bias) # Read(file_path=bitsandbytes/research/nn/modules.py, offset=46, limit=13)
forward(self, x) # Read(file_path=bitsandbytes/research/nn/modules.py, offset=60, limit=17)

bitsandbytes/triton/dequantize_rowwise.py
=========================================
dequantize_rowwise(x, state_x) # Read(file_path=bitsandbytes/triton/dequantize_rowwise.py, offset=9, limit=2)
_dequantize_rowwise(x_ptr, state_x, output_ptr, inv_127, n_elements, BLOCK_SIZE, P2) # Read(file_path=bitsandbytes/triton/dequantize_rowwise.py, offset=36, limit=18)
dequantize_rowwise(x, state_x) # Read(file_path=bitsandbytes/triton/dequantize_rowwise.py, offset=55, limit=10)

bitsandbytes/triton/int8_matmul_mixed_dequantize.py
===================================================
int8_matmul_mixed_dequantize(a, b, state_x, state_w, bias) # Read(file_path=bitsandbytes/triton/int8_matmul_mixed_dequantize.py, offset=7, limit=2)
init_to_zero(name) # Read(file_path=bitsandbytes/triton/int8_matmul_mixed_dequantize.py, offset=20, limit=2)
get_configs_io_bound() # Read(file_path=bitsandbytes/triton/int8_matmul_mixed_dequantize.py, offset=23, limit=25)
_int8_matmul_mixed_dequantize(A, B, C, bias, state_x_ptr, state_w_ptr, M, N, K, divfactor, has_bias, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, SPLIT_K, EVEN_K, ACC_TYPE) # Read(file_path=bitsandbytes/triton/int8_matmul_mixed_dequantize.py, offset=82, limit=83)
int8_matmul_mixed_dequantize(a, b, state_x, state_w, bias) # Read(file_path=bitsandbytes/triton/int8_matmul_mixed_dequantize.py, offset=166, limit=41)

bitsandbytes/triton/int8_matmul_rowwise_dequantize.py
=====================================================
int8_matmul_rowwise_dequantize(a, b, state_x, state_w, bias) # Read(file_path=bitsandbytes/triton/int8_matmul_rowwise_dequantize.py, offset=7, limit=2)
init_to_zero(name) # Read(file_path=bitsandbytes/triton/int8_matmul_rowwise_dequantize.py, offset=20, limit=2)
get_configs_io_bound() # Read(file_path=bitsandbytes/triton/int8_matmul_rowwise_dequantize.py, offset=23, limit=25)
_int8_matmul_rowwise_dequantize(A, B, C, bias, state_x_ptr, state_w_ptr, M, N, K, divfactor, has_bias, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, SPLIT_K, EVEN_K, ACC_TYPE) # Read(file_path=bitsandbytes/triton/int8_matmul_rowwise_dequantize.py, offset=82, limit=82)
int8_matmul_rowwise_dequantize(a, b, state_x, state_w, bias) # Read(file_path=bitsandbytes/triton/int8_matmul_rowwise_dequantize.py, offset=165, limit=43)

bitsandbytes/triton/matmul_perf_model.py
========================================
get_clock_rate_in_khz() # Read(file_path=bitsandbytes/triton/matmul_perf_model.py, offset=20, limit=9)
get_tensorcore_tflops(device, num_ctas, num_warps, dtype) # Read(file_path=bitsandbytes/triton/matmul_perf_model.py, offset=31, limit=10)
get_simd_tflops(device, num_ctas, num_warps, dtype) # Read(file_path=bitsandbytes/triton/matmul_perf_model.py, offset=43, limit=8)
get_tflops(device, num_ctas, num_warps, dtype) # Read(file_path=bitsandbytes/triton/matmul_perf_model.py, offset=53, limit=5)
estimate_matmul_time(num_warps, num_stages, A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, debug, **kwargs) # Read(file_path=bitsandbytes/triton/matmul_perf_model.py, offset=60, limit=74)
early_config_prune(configs, named_args, **kwargs) # Read(file_path=bitsandbytes/triton/matmul_perf_model.py, offset=136, limit=76)

bitsandbytes/triton/quantize_columnwise_and_transpose.py
========================================================
quantize_columnwise_and_transpose(x) # Read(file_path=bitsandbytes/triton/quantize_columnwise_and_transpose.py, offset=9, limit=2)
_quantize_columnwise_and_transpose(x_ptr, output_ptr, output_maxs, n_elements, M, N, BLOCK_SIZE, P2) # Read(file_path=bitsandbytes/triton/quantize_columnwise_and_transpose.py, offset=38, limit=25)
quantize_columnwise_and_transpose(x) # Read(file_path=bitsandbytes/triton/quantize_columnwise_and_transpose.py, offset=64, limit=12)

bitsandbytes/triton/quantize_global.py
======================================
quantize_global_transpose(input) # Read(file_path=bitsandbytes/triton/quantize_global.py, offset=7, limit=2)
quantize_global(x) # Read(file_path=bitsandbytes/triton/quantize_global.py, offset=10, limit=2)
_quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE) # Read(file_path=bitsandbytes/triton/quantize_global.py, offset=25, limit=15)
quantize_global(x) # Read(file_path=bitsandbytes/triton/quantize_global.py, offset=41, limit=9)
_quantize_global_transpose(A, absmax_inv_ptr, B, stride_am, stride_an, stride_bn, stride_bm, M, N, BLOCK_M, BLOCK_N, GROUP_M) # Read(file_path=bitsandbytes/triton/quantize_global.py, offset=61, limit=40)
quantize_global_transpose(input) # Read(file_path=bitsandbytes/triton/quantize_global.py, offset=102, limit=23)

bitsandbytes/triton/quantize_rowwise.py
=======================================
quantize_rowwise(x) # Read(file_path=bitsandbytes/triton/quantize_rowwise.py, offset=9, limit=2)
_quantize_rowwise(x_ptr, output_ptr, output_maxs, n_elements, BLOCK_SIZE, P2) # Read(file_path=bitsandbytes/triton/quantize_rowwise.py, offset=36, limit=20)
quantize_rowwise(x) # Read(file_path=bitsandbytes/triton/quantize_rowwise.py, offset=57, limit=11)

bitsandbytes/triton/triton_utils.py
===================================
is_triton_available() # Read(file_path=bitsandbytes/triton/triton_utils.py, offset=5, limit=10)

bitsandbytes/utils.py
=====================
outlier_hook(module, input) # Read(file_path=bitsandbytes/utils.py, offset=8, limit=31)
_reverse_4bit_compress_format(weight) # Read(file_path=bitsandbytes/utils.py, offset=42, limit=5)
OutlierTracer() # Read(file_path=bitsandbytes/utils.py, offset=49, limit=40)
__init__(self) # Read(file_path=bitsandbytes/utils.py, offset=52, limit=2)
initialize(self, model) # Read(file_path=bitsandbytes/utils.py, offset=55, limit=12)
is_initialized(self) # Read(file_path=bitsandbytes/utils.py, offset=68, limit=2)
get_hvalue(self, weight) # Read(file_path=bitsandbytes/utils.py, offset=71, limit=2)
get_outliers(self, weight) # Read(file_path=bitsandbytes/utils.py, offset=74, limit=9)
get_instance(cls) # Read(file_path=bitsandbytes/utils.py, offset=85, limit=4)
find_outlier_dims(weight, reduction_dim, zscore, topk, rdm) # Read(file_path=bitsandbytes/utils.py, offset=91, limit=21)
execute_and_return(command_string) # Read(file_path=bitsandbytes/utils.py, offset=114, limit=15)
_decode(subprocess_err_out_tuple) # Read(file_path=bitsandbytes/utils.py, offset=115, limit=2)
execute_and_return_decoded_std_streams(command_string) # Read(file_path=bitsandbytes/utils.py, offset=118, limit=8)
replace_linear(model, linear_replacement, skip_modules, copy_weights, post_processing_function) # Read(file_path=bitsandbytes/utils.py, offset=131, limit=43)
pack_dict_to_tensor(source_dict) # Read(file_path=bitsandbytes/utils.py, offset=176, limit=15)
unpack_tensor_to_dict(tensor_data) # Read(file_path=bitsandbytes/utils.py, offset=193, limit=15)

install_cuda.py
===============
install_cuda(version, base_path, download_path) # Read(file_path=install_cuda.py, offset=18, limit=47)
main() # Read(file_path=install_cuda.py, offset=67, limit=30)

names.py
========
show_info(functionNode, isclass) # Read(file_path=names.py, offset=5, limit=7)

scripts/stale.py
================
main() # Read(file_path=scripts/stale.py, offset=30, limit=26)

setup.py
========
BinaryDistribution(Distribution) # Read(file_path=setup.py, offset=10, limit=3)
has_ext_modules(self) # Read(file_path=setup.py, offset=11, limit=2)

test.py
=======
GrepExtractor() # Read(file_path=test.py, offset=17, limit=159)
get_indent_level(line) # Read(file_path=test.py, offset=21, limit=11)
find_definitions_with_grep(file_path) # Read(file_path=test.py, offset=34, limit=29)
extract_signature(line) # Read(file_path=test.py, offset=65, limit=30)
find_end_line(lines, start_idx, start_indent) # Read(file_path=test.py, offset=97, limit=40)
process_file(file_path) # Read(file_path=test.py, offset=139, limit=37)
ASTExtractor() # Read(file_path=test.py, offset=178, limit=92)
CodeVisitor(ast.NodeVisitor) # Read(file_path=test.py, offset=181, limit=67)
__init__(self) # Read(file_path=test.py, offset=184, limit=2)
visit_ClassDef(self, node) # Read(file_path=test.py, offset=187, limit=21)
visit_FunctionDef(self, node) # Read(file_path=test.py, offset=209, limit=3)
visit_AsyncFunctionDef(self, node) # Read(file_path=test.py, offset=213, limit=3)
_process_function(self, node, is_async) # Read(file_path=test.py, offset=217, limit=31)
process_file(file_path) # Read(file_path=test.py, offset=250, limit=20)
find_python_files(root_path, recursive) # Read(file_path=test.py, offset=272, limit=6)
format_output(file_path, elements, root_path) # Read(file_path=test.py, offset=280, limit=22)
benchmark_methods(file_path) # Read(file_path=test.py, offset=304, limit=17)
main() # Read(file_path=test.py, offset=323, limit=74)

test2.py
========
CodeVisitor(ast.NodeVisitor) # Read(file_path=test2.py, offset=15, limit=68)
__init__(self) # Read(file_path=test2.py, offset=18, limit=2)
visit_ClassDef(self, node) # Read(file_path=test2.py, offset=21, limit=22)
visit_FunctionDef(self, node) # Read(file_path=test2.py, offset=44, limit=3)
visit_AsyncFunctionDef(self, node) # Read(file_path=test2.py, offset=48, limit=3)
_process_function(self, node, is_async) # Read(file_path=test2.py, offset=52, limit=31)
process_file(file_path) # Read(file_path=test2.py, offset=85, limit=24)
find_python_files(root_path) # Read(file_path=test2.py, offset=111, limit=17)
format_file_section(file_path, elements, root_path) # Read(file_path=test2.py, offset=130, limit=44)
generate_docs_brief(root_path, output_path) # Read(file_path=test2.py, offset=176, limit=66)
main() # Read(file_path=test2.py, offset=244, limit=37)

test3.py
========
CodeVisitor(ast.NodeVisitor) # Read(file_path=test3.py, offset=14, limit=68)
__init__(self) # Read(file_path=test3.py, offset=17, limit=2)
visit_ClassDef(self, node) # Read(file_path=test3.py, offset=20, limit=22)
visit_FunctionDef(self, node) # Read(file_path=test3.py, offset=43, limit=3)
visit_AsyncFunctionDef(self, node) # Read(file_path=test3.py, offset=47, limit=3)
_process_function(self, node, is_async) # Read(file_path=test3.py, offset=51, limit=31)
process_file(file_path) # Read(file_path=test3.py, offset=84, limit=24)
find_python_files(root_path) # Read(file_path=test3.py, offset=110, limit=17)
format_file_section(file_path, elements, root_path) # Read(file_path=test3.py, offset=129, limit=19)
generate_docs_brief(root_path, output_path) # Read(file_path=test3.py, offset=150, limit=40)
main() # Read(file_path=test3.py, offset=192, limit=37)

tests/conftest.py
=================
_set_seed() # Read(file_path=tests/conftest.py, offset=9, limit=6)
pytest_runtest_call(item) # Read(file_path=tests/conftest.py, offset=17, limit=13)
pytest_runtest_teardown(item, nextitem) # Read(file_path=tests/conftest.py, offset=33, limit=4)
requires_cuda() # Read(file_path=tests/conftest.py, offset=40, limit=5)

tests/helpers.py
================
get_available_devices() # Read(file_path=tests/helpers.py, offset=19, limit=29)
torch_save_to_buffer(obj) # Read(file_path=tests/helpers.py, offset=50, limit=5)
torch_load_from_buffer(buffer) # Read(file_path=tests/helpers.py, offset=57, limit=5)
get_test_dims(min, max, n) # Read(file_path=tests/helpers.py, offset=64, limit=2)
format_with_label(label, value) # Read(file_path=tests/helpers.py, offset=68, limit=10)
id_formatter(label) # Read(file_path=tests/helpers.py, offset=80, limit=5)
describe_dtype(dtype) # Read(file_path=tests/helpers.py, offset=99, limit=2)

tests/test_autograd.py
======================
test_matmullt(device, dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose, decomp, has_fp16_weights, has_bias) # Read(file_path=tests/test_autograd.py, offset=32, limit=123)
test_matmul_4bit(device, dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose, has_bias, compress_statistics, quant_type) # Read(file_path=tests/test_autograd.py, offset=169, limit=89)

tests/test_cuda_setup_evaluator.py
==================================
cuda120_spec() # Read(file_path=tests/test_cuda_setup_evaluator.py, offset=8, limit=6)
test_get_cuda_bnb_library_path(monkeypatch, cuda120_spec) # Read(file_path=tests/test_cuda_setup_evaluator.py, offset=16, limit=3)
test_get_cuda_bnb_library_path_override(monkeypatch, cuda120_spec, caplog) # Read(file_path=tests/test_cuda_setup_evaluator.py, offset=21, limit=4)

tests/test_deprecated.py
========================
test_dynamic_quantization() # Read(file_path=tests/test_deprecated.py, offset=11, limit=22)
test_percentile_clipping(gtype) # Read(file_path=tests/test_deprecated.py, offset=37, limit=24)
test_matmul_fp8(dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose) # Read(file_path=tests/test_deprecated.py, offset=77, limit=68)
test_fp8linear() # Read(file_path=tests/test_deprecated.py, offset=148, limit=28)

tests/test_functional.py
========================
assert_all_approx_close(a, b, rtol, atol, count, throw) # Read(file_path=tests/test_functional.py, offset=25, limit=9)
FFN(torch.nn.Module) # Read(file_path=tests/test_functional.py, offset=36, limit=14)
__init__(self, input_features, hidden_size, bias) # Read(file_path=tests/test_functional.py, offset=37, limit=8)
forward(self, x) # Read(file_path=tests/test_functional.py, offset=46, limit=4)
Timer() # Read(file_path=tests/test_functional.py, offset=52, limit=36)
__init__(self) # Read(file_path=tests/test_functional.py, offset=53, limit=4)
tick(self, name) # Read(file_path=tests/test_functional.py, offset=58, limit=7)
tock(self, name, evict, print_ms) # Read(file_path=tests/test_functional.py, offset=66, limit=16)
reset(self) # Read(file_path=tests/test_functional.py, offset=83, limit=5)
Test8BitBlockwiseQuantizeFunctional() # Read(file_path=tests/test_functional.py, offset=90, limit=202)
test_dynamic_blockwise_quantization(self, device, dtype, nested, blocksize, signed) # Read(file_path=tests/test_functional.py, offset=96, limit=51)
test_blockwise_cpu_large(self, hidden, blocksize) # Read(file_path=tests/test_functional.py, offset=151, limit=17)
test_few_bit_quant(self, device, bits, method) # Read(file_path=tests/test_functional.py, offset=174, limit=50)
test_fp8_quant(self, device) # Read(file_path=tests/test_functional.py, offset=226, limit=47)
test_bench_dequantization(self) # Read(file_path=tests/test_functional.py, offset=278, limit=14)
test_stable_embedding() # Read(file_path=tests/test_functional.py, offset=295, limit=3)
quant(x) # Read(file_path=tests/test_functional.py, offset=300, limit=4)
dequant(c, maxC) # Read(file_path=tests/test_functional.py, offset=306, limit=2)
mm_dequant(maxA, maxB, C) # Read(file_path=tests/test_functional.py, offset=310, limit=2)
quant_multi(x, dim) # Read(file_path=tests/test_functional.py, offset=314, limit=5)
quant_multi_chunk(x, dim, chunk_size) # Read(file_path=tests/test_functional.py, offset=321, limit=14)
mean(xx) # Read(file_path=tests/test_functional.py, offset=337, limit=2)
TestIGEMMFunctional() # Read(file_path=tests/test_functional.py, offset=354, limit=206)
test_approx_igemm(self, dim1, dim2, quant_methods, batched) # Read(file_path=tests/test_functional.py, offset=359, limit=32)
test_igemm(self, hidden_dim, batch_dim, transpose, seq_dim) # Read(file_path=tests/test_functional.py, offset=398, limit=41)
test_dim3_igemm(self, seq_dim, hidden_dim, batch_dim) # Read(file_path=tests/test_functional.py, offset=443, limit=12)
test_minmax_igemm(self, seq_dim, hidden_dim, batch_dim, transpose) # Read(file_path=tests/test_functional.py, offset=460, limit=71)
min_max(x) # Read(file_path=tests/test_functional.py, offset=461, limit=5)
test_ibmm(self, dim1, dim2, dim3, dim4, transpose) # Read(file_path=tests/test_functional.py, offset=537, limit=23)
TestLLMInt8Functional() # Read(file_path=tests/test_functional.py, offset=562, limit=261)
vectorwise_mm_dequant(xq, S1, S2, dtype) # Read(file_path=tests/test_functional.py, offset=564, limit=15)
vectorwise_quant(x, dim) # Read(file_path=tests/test_functional.py, offset=581, limit=5)
test_int8_linear_matmul(self, device, dim1, dim2, dim3, dim4, dims, ldb) # Read(file_path=tests/test_functional.py, offset=594, limit=11)
test_int8_linear_matmul_half(self, device, dim1, dim2, dim3, dim4, dims) # Read(file_path=tests/test_functional.py, offset=612, limit=17)
test_dequant_mm(self, device, dim1, dim4, dims, has_bias) # Read(file_path=tests/test_functional.py, offset=635, limit=36)
test_colrow_absmax(self, dim1, dim2, dims, threshold) # Read(file_path=tests/test_functional.py, offset=677, limit=33)
test_int8_double_quant(self, dim1, dim2) # Read(file_path=tests/test_functional.py, offset=714, limit=31)
test_integrated_int8_linear_matmul(self, device, dim1, dim4, inner) # Read(file_path=tests/test_functional.py, offset=758, limit=29)
test_coo_double_quant(self, device, dim1, dim2) # Read(file_path=tests/test_functional.py, offset=791, limit=16)
test_coo_int8_vectorwise_quant(self, device, dim1, dim2) # Read(file_path=tests/test_functional.py, offset=811, limit=12)
TestSpMMFunctional() # Read(file_path=tests/test_functional.py, offset=826, limit=232)
test_spmm_coo(self, dim1, dim2, transposed_B) # Read(file_path=tests/test_functional.py, offset=830, limit=26)
test_spmm_bench(self) # Read(file_path=tests/test_functional.py, offset=858, limit=39)
test_spmm_coo_very_sparse(self, dim1, dim2, dtype, out_func) # Read(file_path=tests/test_functional.py, offset=902, limit=41)
test_spmm_coo_dequant(self, dim1, dim2, dtype) # Read(file_path=tests/test_functional.py, offset=962, limit=96)
TestSparseTensorFunctional() # Read(file_path=tests/test_functional.py, offset=1061, limit=35)
test_coo2csr(self) # Read(file_path=tests/test_functional.py, offset=1062, limit=16)
test_coo2csc(self) # Read(file_path=tests/test_functional.py, offset=1079, limit=17)
TestQuantize4BitFunctional() # Read(file_path=tests/test_functional.py, offset=1098, limit=277)
test_4bit_quant(self, device, dtype, quant_type, blocksize) # Read(file_path=tests/test_functional.py, offset=1103, limit=28)
test_4bit_compressed_stats(self, device, quant_type, blocksize) # Read(file_path=tests/test_functional.py, offset=1135, limit=27)
test_bench_4bit_dequant(self, quant_type) # Read(file_path=tests/test_functional.py, offset=1167, limit=20)
test_gemv_4bit(self, device, dim, dtype, storage_type, quant_storage, double_quant, kind) # Read(file_path=tests/test_functional.py, offset=1207, limit=141)
test_gemv_eye_4bit(self, device, storage_type, dtype, double_quant) # Read(file_path=tests/test_functional.py, offset=1353, limit=22)
test_normal_map_tree() # Read(file_path=tests/test_functional.py, offset=1379, limit=12)

tests/test_generation.py
========================
get_4bit_config() # Read(file_path=tests/test_generation.py, offset=12, limit=10)
get_model_and_tokenizer(config) # Read(file_path=tests/test_generation.py, offset=24, limit=18)
get_prompt_for_generation_eval(text, add_roles) # Read(file_path=tests/test_generation.py, offset=44, limit=10)
generate(model, tokenizer, text, generation_config, prompt_func) # Read(file_path=tests/test_generation.py, offset=56, limit=5)
model_and_tokenizer(request) # Read(file_path=tests/test_generation.py, offset=68, limit=4)
test_pi(requires_cuda, model_and_tokenizer, inference_kernel, DQ, dtype) # Read(file_path=tests/test_generation.py, offset=78, limit=45)

tests/test_linear4bit.py
========================
test_linear_serialization(device, quant_type, compress_statistics, bias, quant_storage, save_before_forward) # Read(file_path=tests/test_linear4bit.py, offset=34, limit=150)
test_copy_param(device, quant_type, blocksize, compress_statistics) # Read(file_path=tests/test_linear4bit.py, offset=190, limit=13)
test_deepcopy_param(device, quant_type, blocksize, compress_statistics) # Read(file_path=tests/test_linear4bit.py, offset=209, limit=20)
test_params4bit_real_serialization(device, quant_type, blocksize, compress_statistics) # Read(file_path=tests/test_linear4bit.py, offset=235, limit=27)
test_linear4bit_torch_compile(device, quant_type, compute_dtype, compress_statistics, bias, fullgraph, mode) # Read(file_path=tests/test_linear4bit.py, offset=272, limit=69)

tests/test_linear8bitlt.py
==========================
test_linear_no_igemmlt(device) # Read(file_path=tests/test_linear8bitlt.py, offset=25, limit=39)
test_linear_serialization(device, has_fp16_weights, threshold, serialize_before_forward, deserialize_before_cuda, save_before_forward, load_before_cuda) # Read(file_path=tests/test_linear8bitlt.py, offset=73, limit=99)
linear8bit(requires_cuda) # Read(file_path=tests/test_linear8bitlt.py, offset=175, limit=17)
test_linear8bit_copy_param(linear8bit) # Read(file_path=tests/test_linear8bitlt.py, offset=194, limit=5)
test_linear8bit_deepcopy_param(linear8bit) # Read(file_path=tests/test_linear8bitlt.py, offset=201, limit=13)
test_linear8bit_serialization(linear8bit) # Read(file_path=tests/test_linear8bitlt.py, offset=216, limit=12)
test_linear8bitlt_torch_compile(device, threshold, bias, fullgraph, mode) # Read(file_path=tests/test_linear8bitlt.py, offset=236, limit=60)

tests/test_modules.py
=====================
MockArgs() # Read(file_path=tests/test_modules.py, offset=11, limit=4)
__init__(self, initial_data) # Read(file_path=tests/test_modules.py, offset=12, limit=3)
MLP8bit(torch.nn.Module) # Read(file_path=tests/test_modules.py, offset=17, limit=20)
__init__(self, dim1, dim2, has_fp16_weights, threshold) # Read(file_path=tests/test_modules.py, offset=18, limit=14)
forward(self, x) # Read(file_path=tests/test_modules.py, offset=33, limit=4)
get_args() # Read(file_path=tests/test_modules.py, offset=39, limit=6)
assert_all_approx_close(a, b, atol, rtol, count) # Read(file_path=tests/test_modules.py, offset=47, limit=6)
test_linear8bitlt_inference(device, threshold) # Read(file_path=tests/test_modules.py, offset=57, limit=11)
test_linear8bitlt_accumulated_gradient(device) # Read(file_path=tests/test_modules.py, offset=72, limit=43)
test_linear8bitlt_no_fp16_weights(device, threshold) # Read(file_path=tests/test_modules.py, offset=119, limit=119)
test_linear_kbit_fp32_bias(device, module) # Read(file_path=tests/test_modules.py, offset=249, limit=21)
test_kbit_backprop(device, module) # Read(file_path=tests/test_modules.py, offset=287, limit=57)
test_embedding_lossless(device, embedding_class, input_shape, embedding_dim, quant_storage) # Read(file_path=tests/test_modules.py, offset=360, limit=29)
test_embedding_error(device, embedding_class, input_shape, embedding_dim, quant_storage) # Read(file_path=tests/test_modules.py, offset=405, limit=31)
test_4bit_linear_warnings(device) # Read(file_path=tests/test_modules.py, offset=439, limit=26)
test_4bit_embedding_warnings(device) # Read(file_path=tests/test_modules.py, offset=468, limit=11)
test_4bit_embedding_weight_fsdp_fix(requires_cuda) # Read(file_path=tests/test_modules.py, offset=481, limit=15)
test_4bit_linear_weight_fsdp_fix(requires_cuda) # Read(file_path=tests/test_modules.py, offset=498, limit=15)
test_embedding_not_implemented_error() # Read(file_path=tests/test_modules.py, offset=515, limit=8)

tests/test_ops.py
=================
TestLLMInt8Ops() # Read(file_path=tests/test_ops.py, offset=18, limit=82)
test_int8_linear_matmul(self, device) # Read(file_path=tests/test_ops.py, offset=20, limit=10)
test_int8_linear_matmul_out(self, device) # Read(file_path=tests/test_ops.py, offset=32, limit=12)
test_int8_vectorwise_quant(self, threshold, device) # Read(file_path=tests/test_ops.py, offset=47, limit=23)
test_int8_mm_dequant(self, device) # Read(file_path=tests/test_ops.py, offset=72, limit=11)
test_int8_scaled_mm(self, device, dtype, has_bias) # Read(file_path=tests/test_ops.py, offset=87, limit=13)
TestInt8BlockwiseQuantOps() # Read(file_path=tests/test_ops.py, offset=102, limit=50)
test_quantize_blockwise(self, device, dtype, blocksize) # Read(file_path=tests/test_ops.py, offset=106, limit=20)
test_dequantize_blockwise(self, device, dtype, blocksize) # Read(file_path=tests/test_ops.py, offset=130, limit=22)
Test4bitBlockwiseQuantOps() # Read(file_path=tests/test_ops.py, offset=154, limit=75)
test_quantize_4bit(self, device, dtype, storage_dtype, quant_type, blocksize) # Read(file_path=tests/test_ops.py, offset=160, limit=15)
test_dequantize_4bit(self, device, dtype, storage_dtype, quant_type, blocksize) # Read(file_path=tests/test_ops.py, offset=181, limit=25)
test_gemv_4bit(self, device, dtype, storage_dtype, quant_type, blocksize) # Read(file_path=tests/test_ops.py, offset=212, limit=17)

tests/test_optim.py
===================
assert_most_approx_close(a, b, rtol, atol, max_error_count) # Read(file_path=tests/test_optim.py, offset=21, limit=6)
get_temp_dir() # Read(file_path=tests/test_optim.py, offset=29, limit=4)
rm_path(path) # Read(file_path=tests/test_optim.py, offset=35, limit=2)
test_optimizer32bit(requires_cuda, dim1, dim2, gtype, optim_name) # Read(file_path=tests/test_optim.py, offset=171, limit=74)
test_global_config(requires_cuda, dim1, dim2, gtype) # Read(file_path=tests/test_optim.py, offset=250, limit=39)
test_optimizer8bit(requires_cuda, dim1, dim2, gtype, optim_name) # Read(file_path=tests/test_optim.py, offset=305, limit=139)
test_adam_percentile_clipping(requires_cuda, dim1, dim2, gtype, optim_bits) # Read(file_path=tests/test_optim.py, offset=451, limit=81)
test_benchmark_blockwise(dim1, dim2, gtype, optim_name) # Read(file_path=tests/test_optim.py, offset=552, limit=23)

tests/test_triton.py
====================
test_switchback(vector_wise_quantization) # Read(file_path=tests/test_triton.py, offset=16, limit=49)