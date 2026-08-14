# This CMake config hopefully makes it easier to compile.
# Ensure the CUDA Toolkit is available on your path. Then run:
#   For  GCC: `cmake -B build . && cmake --build build`
#   For MSVC: `cmake -B build . && cmake --build build --config Release`
# You can also use the following options and variables
#  - COMPUTE_BACKEND: Set to `cpu`, `cuda`, `hip`, or `xpu` to select the backend
#  - CUDA_VERSION: The expected CUDA version, for sanity checking. The actual version
#                  is whatever CMake finds on your path.
#  - COMPUTE_CAPABILITY: Which GPU Arch/Compute codes to provide to NVCC.
#                        Separate by semicolons, i.e. `-DCOMPUTE_CAPABILITY=89;90;100;120`
#                        Check your compute capability here: https://developer.nvidia.com/cuda-gpus
#  - PTXAS_VERBOSE: Pass the `-v` option to the PTX Assembler
#  - ROCM_VERSION: Override the ROCm release used in the output library name. Accepts a
#                  dotted release or shortcode. For example, `7.14.0` and `714`
#                  both produce libbitsandbytes_rocm714.so.
cmake_minimum_required(VERSION 3.22.1)

# On Windows with HIP backend, auto-detect compilers from ROCM_PATH before project()
if(WIN32 AND COMPUTE_BACKEND STREQUAL "hip")
    if(NOT DEFINED ENV{ROCM_PATH} OR "$ENV{ROCM_PATH}" STREQUAL "")
        message(FATAL_ERROR
            "ROCM_PATH must be set for HIP builds on Windows. "
            "After 'rocm-sdk init', set it from 'rocm-sdk path --root'. "
            "PowerShell: $env:ROCM_PATH = (rocm-sdk path --root)"
        )
    endif()
    file(TO_CMAKE_PATH "$ENV{ROCM_PATH}" ROCM_PATH)
    if(ROCM_PATH AND NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++.exe")
    endif()
    if(ROCM_PATH AND NOT DEFINED CMAKE_HIP_COMPILER)
        set(CMAKE_HIP_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++.exe")
    endif()
    if(NOT DEFINED HIP_PLATFORM)
        set(HIP_PLATFORM "amd" CACHE STRING "HIP Platform")
    endif()
    # On Windows, the HIP compiler needs explicit paths to find device libraries.
    if(ROCM_PATH)
        find_path(ROCM_DEVICE_LIB_PATH
            NAMES oclc_abi_version_400.bc ocml.bc
            PATHS "${ROCM_PATH}/amdgcn/bitcode"
                  "${ROCM_PATH}/lib/llvm/amdgcn/bitcode"
            NO_DEFAULT_PATH
        )
        string(APPEND CMAKE_HIP_FLAGS " --rocm-path=${ROCM_PATH}")
        if(ROCM_DEVICE_LIB_PATH)
            string(APPEND CMAKE_HIP_FLAGS " --rocm-device-lib-path=${ROCM_DEVICE_LIB_PATH}")
        endif()
    endif()
endif()

project(bitsandbytes LANGUAGES CXX)

# If run without specifying a build type, default to using the Release configuration:
#    optimizing the generated binaries for performance and also adds the `-DNDEBUG` flag,
#    which turns off a bunch of asserts which seem to link to new symbols in libstdc++,
#    worsening our many_linux compliance..
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define included source files
set(CPP_FILES csrc/cpu_ops.cpp csrc/pythonInterface.cpp)
set(GPU_FILES csrc/ops.cu csrc/kernels.cu)
set(XPU_FILES csrc/xpu_ops.cpp csrc/xpu_kernels.cpp)
# C++ sources are always included
list(APPEND SRC_FILES ${CPP_FILES})

set(COMPUTE_BACKEND "cpu" CACHE STRING "The compute backend to use (cpu, cuda, hip, xpu)")
set_property(CACHE COMPUTE_BACKEND PROPERTY STRINGS cpu cuda hip xpu)
option(PTXAS_VERBOSE "Pass through -v flag to PTX Assembler" OFF)

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 14.0)
endif()

set(BNB_OUTPUT_NAME "bitsandbytes")

message(STATUS "Configuring ${PROJECT_NAME} (Backend: ${COMPUTE_BACKEND})")

if(${COMPUTE_BACKEND} STREQUAL "cuda")
    if(APPLE)
        message(FATAL_ERROR "CUDA is not supported on macOS" )
    endif()
    set(BUILD_CUDA ON)
    set(BUILD_HIP OFF)
elseif(${COMPUTE_BACKEND} STREQUAL "hip")
    if(APPLE)
        message(FATAL_ERROR "HIP is not supported on macOS" )
    endif()
    set(BUILD_CUDA OFF)
    set(BUILD_HIP ON)
elseif(${COMPUTE_BACKEND} STREQUAL "xpu")
    if(APPLE)
        message(FATAL_ERROR "XPU is not supported on macOS" )
    endif()
    set(BUILD_CUDA OFF)
    set(BUILD_HIP OFF)
    set(BUILD_XPU ON)
else()
    set(BUILD_CUDA OFF)
    set(BUILD_HIP OFF)
    set(BUILD_XPU OFF)
    set(BUILD_CPU ON)
endif()


if (BUILD_CPU)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" HOST_ARCH)
    if(MSVC)
        # Use the experimental OpenMP runtime for persistent thread pool support.
        # Requires CMake 3.30+; silently ignored on older CMake versions.
        set(OpenMP_RUNTIME_MSVC "experimental")
    endif()
    find_package(OpenMP)
endif()

if(BUILD_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # NVCC normally will only work with MSVC up to 1939. VS2022 17.10+ starts using versions 1940+.
    # Workaround: use --allow-unsupported-compiler
    # This needs to be added *before* we try to enable the CUDA language so CMake's compiler check passes.
    if(MSVC AND MSVC_VERSION VERSION_GREATER_EQUAL 1940)
        string(APPEND CMAKE_CUDA_FLAGS " --allow-unsupported-compiler")

        # Suppress MSVC STL version mismatch errors when using a newer compiler than CUDA officially supports.
        # This is needed to build with VS2022 17.11+ and CUDA < 12.4.
        if (MSVC_VERSION VERSION_GREATER_EQUAL 1941)
            string(APPEND CMAKE_CUDA_FLAGS " -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
        endif()

    endif()

    enable_language(CUDA) # This will fail if CUDA is not found
    find_package(CUDAToolkit REQUIRED)

    # Convert the CUDA version from X.Y.z to XY. There's probably a shorter way of doing this
    string(REGEX MATCH "^[0-9]+.[0-9]+" _CUDA_VERSION_FIRST_TWO "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REPLACE "." "" CUDA_VERSION_SHORT "${_CUDA_VERSION_FIRST_TWO}")

    # Expose a cache variable that the user can set to ensure the correct version of CUDA is found
    set(CUDA_VERSION "${CUDA_VERSION_SHORT}" CACHE STRING "Expected CUDA Version Shortcode")

    message(STATUS "CUDA Version: ${CUDA_VERSION_SHORT} (${CMAKE_CUDA_COMPILER_VERSION})")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

    # It should match the discovered version
    if(NOT CUDA_VERSION STREQUAL "${CUDA_VERSION_SHORT}")
        message(FATAL_ERROR "You've specified CUDA version ${CUDA_VERSION} however the CUDA compiler found is ${CUDA_VERSION_SHORT}."
            " Ensure the desired CUDA compiler is the first one available on your PATH."
        )
    endif()

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.8")
        message(FATAL_ERROR "CUDA Version < 11.8 is not supported")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "14.0")
        message(FATAL_ERROR "CUDA Version > 13 is not supported")
    endif()

    # CMake < 3.23.0 does not define CMAKE_CUDA_ARCHITECTURES_ALL.
    if(CMAKE_VERSION VERSION_LESS "3.23.0")
        message(STATUS "CMake < 3.23.0; determining CUDA architectures supported...")

        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
            # Starting in CUDA 13.0, Thor Blackwell is renamed to SM110.
            # Support for architectures older than Turing (SM75) is removed.
            list(APPEND CMAKE_CUDA_ARCHITECTURES_ALL 75 80 86 87 88 89 90 100 103 110 120 121)
            list(APPEND CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR 80 90 100 110 120)
        else()
            # 11.8-12.9 supports these at a minimum.
            set(CMAKE_CUDA_ARCHITECTURES_ALL 50 52 53 60 61 62 70 72 75 80 86 87 89 90)
            set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR 50 60 70 80 90)

            # CUDA 12.8 adds support for Blackwell.
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES_ALL 100 101 120 121)
                list(APPEND CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR 100 120)
            endif()

            # CUDA 12.9 adds SM103 (Blackwell B300).
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
                list(APPEND CMAKE_CUDA_ARCHITECTURES_ALL 103)
            endif()
        endif()
    endif()

    string(APPEND CMAKE_CUDA_FLAGS " --use_fast_math")

    # It's safe for us to enable more aggressive compression for 13.0+
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        string(APPEND CMAKE_CUDA_FLAGS " --compress-mode=size")
    endif()

    # CUDA 13.2+ CCCL headers require MSVC's standard conforming preprocessor.
    if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.2")
        string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/Zc:preprocessor")
    endif()

    if(PTXAS_VERBOSE)
        string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
    endif()

    foreach(capability ${CMAKE_CUDA_ARCHITECTURES_ALL})
        # Most of the items here are like: `xx-real`, so we just extract the `xx` portion
        string(REGEX MATCH "[0-9]+" capability_id "${capability}")
        if(capability_id GREATER 0)
            list(APPEND POSSIBLE_CAPABILITIES ${capability_id})
        endif()
    endforeach()

    # This can be changed via -D argument to CMake
    # By default all possible capabilities are compiled
    set(COMPUTE_CAPABILITY "${POSSIBLE_CAPABILITIES}" CACHE STRING "Compute Capabilities Targeted")

    message(STATUS "CUDA Capabilities Available: ${POSSIBLE_CAPABILITIES}")
    message(STATUS "CUDA Capabilities  Selected: ${COMPUTE_CAPABILITY}")

    # Use the "real" option to build native cubin for all selections.
    # Ensure we build the PTX for the latest version.
    # This behavior of adding a PTX (virtual) target for the highest architecture
    # is similar to how the "all" and "all-major" options would behave in CMake >= 3.23.
    # TODO: Consider bumping CMake requirement and using CMAKE_CUDA_ARCHITECTURES=[all | native] by default
    list(REMOVE_DUPLICATES COMPUTE_CAPABILITY)
    list(SORT COMPUTE_CAPABILITY COMPARE NATURAL)
    list(POP_BACK COMPUTE_CAPABILITY _LATEST_CAPABILITY)
    list(TRANSFORM COMPUTE_CAPABILITY APPEND "-real" OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES)
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${_LATEST_CAPABILITY})

    message(STATUS "CUDA Targets: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA NVCC Flags: ${CMAKE_CUDA_FLAGS}")

    list(APPEND SRC_FILES ${GPU_FILES})

    # 4-bit GEMM SIMT and dispatch compile for all selected archs.
    # sm75/sm80+ MMA files only compile if those archs are selected.
    set(_cc_all ${COMPUTE_CAPABILITY} ${_LATEST_CAPABILITY})
    list(APPEND SRC_FILES csrc/gemm_4bit_simt.cu csrc/gemm_4bit.cu)

    if(75 IN_LIST _cc_all)
        # Builds only on sm75
        list(APPEND SRC_FILES csrc/gemm_4bit_sm75.cu)
        add_compile_definitions(BNB_HAS_GEMM4BIT_SM75)
    endif()

    set(_cc_sm80plus)
    foreach(_cc IN LISTS _cc_all)
        if(_cc GREATER_EQUAL 80)
            list(APPEND _cc_sm80plus ${_cc})
        endif()
    endforeach()
    if(_cc_sm80plus)
        # Builds only on sm80+
        list(APPEND SRC_FILES csrc/gemm_4bit_sm80.cu)
        add_compile_definitions(BNB_HAS_GEMM4BIT_SM80)
    endif()

    string(APPEND BNB_OUTPUT_NAME "_cuda${CUDA_VERSION_SHORT}")
    add_compile_definitions(BUILD_CUDA)
elseif(BUILD_HIP)
    # Set target architectures before enable_language(HIP), which would otherwise
    # auto-detect a single GPU and override the defaults.
    if(DEFINED BNB_ROCM_ARCH)
      set(CMAKE_HIP_ARCHITECTURES ${BNB_ROCM_ARCH})
    elseif(AMDGPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
      set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_TARGETS})
    elseif(NOT CMAKE_HIP_ARCHITECTURES)
      set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")
    endif()

    enable_language(HIP)
    message(STATUS "HIP Compiler: ${CMAKE_HIP_COMPILER}")
    message(STATUS "HIP Targets: ${CMAKE_HIP_ARCHITECTURES}")

    list(APPEND SRC_FILES ${GPU_FILES})

    # 4-bit GEMM: build the SIMT kernel + C dispatch on ROCm.
    set_source_files_properties(csrc/gemm_4bit_simt.cu csrc/gemm_4bit.cu PROPERTIES LANGUAGE HIP)
    list(APPEND SRC_FILES csrc/gemm_4bit_simt.cu csrc/gemm_4bit.cu)

    string(APPEND BNB_OUTPUT_NAME "_rocm")

    # HIP and ROCm releases can diverge, so use the ROCm release for the filename.
    set(ROCM_VERSION "" CACHE STRING "ROCm release used in the output library name")
    set(_ROCM_VERSION_RAW "${ROCM_VERSION}")
    set(_ROCM_VERSION_SOURCE "-DROCM_VERSION")

    if(NOT _ROCM_VERSION_RAW)
        if(NOT ROCM_PATH)
            if(DEFINED ENV{ROCM_PATH})
                file(TO_CMAKE_PATH "$ENV{ROCM_PATH}" ROCM_PATH)
            elseif(CMAKE_HIP_COMPILER_ROCM_ROOT)
                set(ROCM_PATH "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
            elseif(WIN32)
                message(FATAL_ERROR "ROCM_PATH must be set for HIP builds on Windows")
            else()
                message(WARNING "ROCM_PATH is not set; falling back to /opt/rocm")
                set(ROCM_PATH "/opt/rocm")
            endif()
        endif()

        foreach(_VERSION_FILE "${ROCM_PATH}/.info/version" "${ROCM_PATH}/core/.info/version")
            if(NOT _ROCM_VERSION_RAW AND EXISTS "${_VERSION_FILE}")
                file(READ "${_VERSION_FILE}" _ROCM_VERSION_RAW)
                set(_ROCM_VERSION_SOURCE "${_VERSION_FILE}")
            endif()
        endforeach()

        if(NOT _ROCM_VERSION_RAW)
            execute_process(
                COMMAND rocm-sdk version
                OUTPUT_VARIABLE _ROCM_VERSION_RAW
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            set(_ROCM_VERSION_SOURCE "rocm-sdk version")
        endif()

        if(NOT _ROCM_VERSION_RAW)
            message(FATAL_ERROR
                "Could not determine the ROCm version from ${ROCM_PATH}/.info/version, "
                "${ROCM_PATH}/core/.info/version, or 'rocm-sdk version'. "
                "Set ROCM_PATH correctly or pass -DROCM_VERSION=<version>."
            )
        endif()
    endif()

    string(STRIP "${_ROCM_VERSION_RAW}" _ROCM_VERSION_RAW)
    if(_ROCM_VERSION_RAW MATCHES "^([0-9]+)\\.([0-9]+)")
        set(_ROCM_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(_ROCM_VERSION_MINOR "${CMAKE_MATCH_2}")
    elseif(_ROCM_VERSION_RAW MATCHES "^([6-9])([0-9]+)$")
        set(_ROCM_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(_ROCM_VERSION_MINOR "${CMAKE_MATCH_2}")
    elseif(_ROCM_VERSION_RAW MATCHES "^([1-5][0-9])([0-9]+)$")
        set(_ROCM_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(_ROCM_VERSION_MINOR "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR
            "Could not parse ROCm version '${_ROCM_VERSION_RAW}'. "
            "Pass -DROCM_VERSION=7.14.0 (or 714)."
        )
    endif()

    set(ROCM_VERSION "${_ROCM_VERSION_MAJOR}.${_ROCM_VERSION_MINOR}")
    set(_ROCM_VERSION_TAG "${_ROCM_VERSION_MAJOR}${_ROCM_VERSION_MINOR}")
    message(STATUS
        "ROCm Release: ${_ROCM_VERSION_MAJOR}.${_ROCM_VERSION_MINOR}; "
        "library suffix: rocm${_ROCM_VERSION_TAG} (from ${_ROCM_VERSION_SOURCE})"
    )
    string(APPEND BNB_OUTPUT_NAME "${_ROCM_VERSION_TAG}")
    add_compile_definitions(__HIP_PLATFORM_AMD__)
    add_compile_definitions(__HIP_PLATFORM_HCC__)
    add_compile_definitions(BUILD_HIP)
elseif(BUILD_XPU)
    list(APPEND SRC_FILES ${XPU_FILES})
    add_compile_definitions(BUILD_XPU)
    set(CMAKE_C_COMPILER icx)
    set(CMAKE_CXX_COMPILER icpx)
    if(WIN32)
        set(CMAKE_CXX_COMPILER icx)
    endif()

    # Get the oneAPI year (2025, 2026, ...) so we can name the library with it.
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE _XPU_COMPILER_VERSION_OUTPUT
        ERROR_VARIABLE _XPU_COMPILER_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "20[0-9][0-9]" XPU_VERSION "${_XPU_COMPILER_VERSION_OUTPUT}")

    if(XPU_VERSION)
        message(STATUS "XPU oneAPI Version: ${XPU_VERSION}")
        string(APPEND BNB_OUTPUT_NAME "_xpu${XPU_VERSION}")
    else()
        message(WARNING "Could not detect oneAPI version from icpx; building unversioned libbitsandbytes_xpu.")
        string(APPEND BNB_OUTPUT_NAME "_xpu")
    endif()
else()
    string(APPEND BNB_OUTPUT_NAME "_cpu")
    set(GPU_SOURCES)
endif()


if(WIN32)
    # Export all symbols
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    # Prevent Windows SDK min/max macros from conflicting with std::min/std::max
    add_compile_definitions(NOMINMAX)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")

    # Only real cl.exe needs (and accepts) the conformant preprocessor switch;
    # clang-cl / icx are already conformant and warn on it.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")
    endif()

    # /arch:AVX2 is only valid for x86/x64 targets, not ARM64
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _msvc_arch)
    if(_msvc_arch MATCHES "x86|x64|amd64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    endif()
endif()

set_source_files_properties(${CPP_FILES} PROPERTIES LANGUAGE CXX)
add_library(bitsandbytes SHARED ${SRC_FILES})
target_compile_features(bitsandbytes PUBLIC cxx_std_17)
target_include_directories(bitsandbytes PUBLIC csrc)
set_target_properties(bitsandbytes PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

if (BUILD_CPU)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
    if (ipo_supported AND NOT MSVC)
        set_property(TARGET bitsandbytes PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

    if (OpenMP_CXX_FOUND)
        target_link_libraries(bitsandbytes PRIVATE OpenMP::OpenMP_CXX)
        add_definitions(-DHAS_OPENMP)
    endif()

    if (NOT MSVC)
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_compile_options(bitsandbytes PRIVATE -fno-semantic-interposition)
        endif()

        if (HOST_ARCH MATCHES "x86_64|amd64")
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag(-mavx512f HAS_AVX512F_FLAG)
            check_cxx_compiler_flag(-mavx512bf16 HAS_AVX512BF16_FLAG)
            if (HAS_AVX512F_FLAG)
                target_compile_options(
                    bitsandbytes PRIVATE
                    -mavx512f
                    -mavx512bw
                    -mavx512dq
                    -mavx512vl
                )
            endif()
            if (HAS_AVX512BF16_FLAG)
                target_compile_options(bitsandbytes PRIVATE -mavx512bf16)
            endif()
            target_compile_options(
                bitsandbytes PRIVATE
                -mprefer-vector-width=256
                -mfma
                -mavx2
                -mf16c
                -mlzcnt
                -mbmi
                -mbmi2
            )
        endif()
    endif()
endif()


if(BUILD_CUDA)
    target_include_directories(bitsandbytes PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(bitsandbytes PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
    set_target_properties(bitsandbytes
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
endif()
if(BUILD_HIP)
    # ROCM_PATH was resolved during version detection; retain it for package discovery.
    if(NOT ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            file(TO_CMAKE_PATH "$ENV{ROCM_PATH}" ROCM_PATH)
        elseif(WIN32)
            message(FATAL_ERROR "ROCM_PATH must be set for ROCm builds on Windows")
        else()
            message(WARNING "ROCM_PATH is not set; falling back to /opt/rocm")
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
    macro(find_package_and_print_version PACKAGE_NAME)
      find_package("${PACKAGE_NAME}" ${ARGN})
      message("${PACKAGE_NAME} VERSION: ${${PACKAGE_NAME}_VERSION}")
    endmacro()
    find_package_and_print_version(hipblas REQUIRED)
    find_package_and_print_version(hiprand REQUIRED)

    ## hacky way of excluding hip::amdhip64 (with it linked many tests unexpectedly fail e.g. adam8bit because of inaccuracies)
    ## On Windows, we need to link amdhip64 explicitly
    if(NOT WIN32)
        set_target_properties(hip::host PROPERTIES INTERFACE_LINK_LIBRARIES "")
        set_target_properties(hip-lang::host PROPERTIES INTERFACE_LINK_LIBRARIES "")
        set(CMAKE_HIP_IMPLICIT_LINK_LIBRARIES "")
    endif()

    target_include_directories(bitsandbytes PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include ${ROCM_PATH}/include /include)
    target_link_directories(bitsandbytes PRIVATE ${ROCM_PATH}/lib /lib)
    target_link_libraries(bitsandbytes PUBLIC roc::hipblas hip::hiprand)

    # On Windows, rocblas is not pulled in transitively by roc::hipblas
    # and is needed because ops_hip.cuh uses rocblas_handle directly.
    if(WIN32)
        target_link_libraries(bitsandbytes PUBLIC rocblas)
    endif()

    target_compile_definitions(bitsandbytes PUBLIC BNB_USE_HIP)
    set_source_files_properties(${GPU_FILES} PROPERTIES LANGUAGE HIP)
    set_target_properties(bitsandbytes PROPERTIES LINKER_LANGUAGE CXX)

    find_package(hipblaslt REQUIRED)
    target_link_libraries(bitsandbytes PUBLIC roc::hipblaslt)
endif()
if(BUILD_XPU)
    set(SYCL_LINK_FLAGS "-fsycl;--offload-compress;-fsycl-targets=spir64_gen,spir64;-Xs;-device pvc,xe-lpg,ats-m150 -options ' -cl-intel-enable-auto-large-GRF-mode -cl-poison-unsupported-fp64-kernels -cl-intel-greater-than-4GB-buffer-required'")
    set(SYCL_COMPILE_FLAGS "-fsycl;-fhonor-nans;-fhonor-infinities;-fno-associative-math;-fno-approx-func;-fno-sycl-instrument-device-code;--offload-compress;-fsycl-targets=spir64_gen,spir64;")

    set_property(TARGET bitsandbytes PROPERTY CXX_STANDARD 20)
    target_compile_options(bitsandbytes PRIVATE ${SYCL_COMPILE_FLAGS})
    target_link_options(bitsandbytes PRIVATE ${SYCL_LINK_FLAGS})

endif()

if(WIN32)
    set_target_properties(bitsandbytes PROPERTIES PREFIX "lib")
endif()
set_target_properties(bitsandbytes PROPERTIES OUTPUT_NAME ${BNB_OUTPUT_NAME})
if(WIN32)
    set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bitsandbytes")
    set_target_properties(bitsandbytes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bitsandbytes")
    if(MSVC)
        set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bitsandbytes")
        set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bitsandbytes")
        set_target_properties(bitsandbytes PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bitsandbytes")
        set_target_properties(bitsandbytes PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bitsandbytes")
    endif()
endif()

set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bitsandbytes")
