cmake_minimum_required(VERSION 3.22.1)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(APPLE)
        # No CUDA on macOS
        set(CMAKE_CUDA_ARCHITECTURES "")
    else()
        # Set cuda12x supported archs only for now
        # This has to be set before project()
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    endif()
endif()

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    message("Building without CUDA support")
    set(LIBSUFFIX "cpu")

    project(bitsandbytes LANGUAGES CXX)
else()
    message("Building CUDA support for ${CMAKE_CUDA_ARCHITECTURES}")
    # Find CUDA tools if we are compiling with CUDA
    find_package(CUDAToolkit REQUIRED)
    set(LIBSUFFIX "cuda${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}")

    project(bitsandbytes LANGUAGES CXX CUDA)
    add_compile_definitions(BUILD_CUDA)
endif()

# Set global flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if(WIN32)
    # Mute warnings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=177")

    # Export all symbols
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Weird MSVC hacks
if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  /NODEFAULTLIB:msvcprtd /NODEFAULTLIB:MSVCRTD /NODEFAULTLIB:LIBCMT")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
endif()

# Add csrc files
add_library(bitsandbytes SHARED
	csrc/ops.cu
	csrc/kernels.cu
	csrc/common.cpp
	csrc/cpu_ops.cpp
	csrc/pythonInterface.cpp)

target_include_directories(bitsandbytes PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    target_include_directories(bitsandbytes PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include)

    # Pass options to NVCC
    target_compile_options(bitsandbytes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xptxas=-v
        -dc
        >)

    set_target_properties(
        bitsandbytes
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(bitsandbytes CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusparse)
endif()

# Set the output name of the CUDA library
set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY ./bitsandbytes)
set_target_properties(bitsandbytes PROPERTIES OUTPUT_NAME "bitsandbytes_${LIBSUFFIX}")
