# This CMake config hopefully makes it easier to compile.
# Ensure the CUDA Toolkit is available on your path. Then run:
#   For  GCC: `cmake -B build . && cmake --build build`
#   For MSVC: `cmake -B build . && cmake --build build --config Release`
# You can also use the following options
#  - BUILD_CUDA: Default ON, will build with CUDA
#  - NO_CUBLASLT: Default OFF, will skip building/linking CUBLASLT support
#  - CUDA_VERSION: The expected CUDA version, for sanity checking. The actual version
#                  is whatever CMake finds on your path.
#  - COMPUTE_CAPABILITY: Which GPU Arch/Compute codes to provide to NVCC.
#                        Separate by semicolons, i.e. `-DCOMPUTE_CAPABILITY=89;90`
#                        Check your compute capability here: https://developer.nvidia.com/cuda-gpus
#  - PTXAS_VERBOSE: Pass the `-v` option to the PTX Assembler
cmake_minimum_required(VERSION 3.18)

project(bitsandbytes LANGUAGES C CXX)

option(BUILD_CUDA "Build bitsandbytes with CUDA support" ON)
option(NO_CUBLASLT "Disable CUBLAS" OFF)
option(PTXAS_VERBOSE "Pass through -v flag to PTX Assembler" OFF)

set(CPP_FILES csrc/common.cpp csrc/cpu_ops.cpp csrc/pythonInterface.c)
list(APPEND CUDA_FILES csrc/ops.cu csrc/kernels.cu)
list(APPEND SRC_FILES ${CPP_FILES})

message(STATUS "BUILD_CUDA := ${BUILD_CUDA}")
message(STATUS "NO_CUBLASLT := ${NO_CUBLASLT}")

set(BNB_OUTPUT_NAME "bitsandbytes")

if(BUILD_CUDA)
    enable_language(CUDA) # This will fail if CUDA is not found

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

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.0")
        message(FATAL_ERROR "CUDA Version < 11 is not supported")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        message(FATAL_ERROR "CUDA Version > 12 is not supported")
    endif()

    string(APPEND CMAKE_CUDA_FLAGS " --use_fast_math")
    if(PTXAS_VERBOSE)
        # Verbose? Outputs register usage information, and other things...
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

    foreach(capability ${COMPUTE_CAPABILITY})
        string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${capability},code=sm_${capability}")
    endforeach()

    message(STATUS "CUDA NVCC Flags: ${CMAKE_CUDA_FLAGS}")

    list(APPEND SRC_FILES ${CUDA_FILES})

    string(APPEND BNB_OUTPUT_NAME "_cuda${CUDA_VERSION_SHORT}")
    if(NO_CUBLASLT)
        string(APPEND BNB_OUTPUT_NAME "_nocublaslt")
    endif()
else()
    message(STATUS "Building CPU Only")
    string(APPEND BNB_OUTPUT_NAME "_cpu")
    if(NO_CUBLASLT)
        message(WARNING "We're building in CPU only mode but NO_CUBLASLT is enabled. It will have no effect.")
    endif()
endif()

set_source_files_properties(${CPP_FILES} PROPERTIES LANGUAGE CXX)
add_library(bitsandbytes SHARED ${SRC_FILES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(bitsandbytes PUBLIC csrc include)
target_compile_features(bitsandbytes PUBLIC cxx_std_14)


if(BUILD_CUDA)
    target_compile_definitions(bitsandbytes PUBLIC BUILD_CUDA)
    target_link_libraries(bitsandbytes PUBLIC cudart cublas cusparse)
    if(NO_CUBLASLT)
        target_compile_definitions(bitsandbytes PUBLIC NO_CUBLASLT)
    else()
        target_link_libraries(bitsandbytes PUBLIC cublasLt)
    endif()

    set_target_properties(bitsandbytes
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(WIN32)
    set_target_properties(bitsandbytes PROPERTIES PREFIX "lib")
endif()

set_target_properties(bitsandbytes
    PROPERTIES
        OUTPUT_NAME ${BNB_OUTPUT_NAME}
        # We have to use a generator expression to prevent MSVC Debug/Release subdirs being made
        RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_SOURCE_DIR}/bitsandbytes>"
        LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_SOURCE_DIR}/bitsandbytes>"
        POSITION_INDEPENDENT_CODE ON # The `-fPIC` commands for non-windows compilers
        WINDOWS_EXPORT_ALL_SYMBOLS ON # On Windows, export all c methods as DLL exports
)
