# CMakeLists.txt Changes for Unified Kernels

## Summary of changes

Replace separate `CUDA_FILES` and `HIP_FILES` with a single `GPU_FILES` list.
For HIP builds, tell CMake to compile `.cu` files using the HIP language.

## Diff

```diff
 # Define included source files
 set(CPP_FILES csrc/cpu_ops.cpp csrc/pythonInterface.cpp)
-set(CUDA_FILES csrc/ops.cu csrc/kernels.cu)
-set(HIP_FILES csrc/ops.hip csrc/kernels.hip)
+set(GPU_FILES csrc/ops.cu csrc/kernels.cu)
 set(MPS_FILES csrc/mps_ops.mm)
 set(METAL_FILES csrc/mps_kernels.metal)
 set(XPU_FILES csrc/xpu_ops.cpp csrc/xpu_kernels.cpp)
```

```diff
 if(BUILD_CUDA)
     # ... (CUDA setup unchanged)
-    list(APPEND SRC_FILES ${CUDA_FILES})
+    list(APPEND SRC_FILES ${GPU_FILES})
     string(APPEND BNB_OUTPUT_NAME "_cuda${CUDA_VERSION_SHORT}")
     add_compile_definitions(BUILD_CUDA)
 elseif(BUILD_HIP)
     # ... (HIP setup unchanged)
-    list(APPEND SRC_FILES ${HIP_FILES})
+    list(APPEND SRC_FILES ${GPU_FILES})
     string(APPEND BNB_OUTPUT_NAME "_rocm")
     # ...
```

```diff
 if(BUILD_HIP)
     # ...
-    set_source_files_properties(${HIP_FILES} PROPERTIES LANGUAGE HIP)
+    set_source_files_properties(${GPU_FILES} PROPERTIES LANGUAGE HIP)
     set_target_properties(bitsandbytes PROPERTIES LINKER_LANGUAGE CXX)
     # ...
 endif()
```

## Files to delete after migration

- `csrc/common_hip.cuh`
- `csrc/kernels.hip`
- `csrc/kernels_hip.cuh`
- `csrc/ops.hip`
- `csrc/ops_hip.cuh`
