cmake_minimum_required(VERSION 3.26.0)

# Set cuda12x supported archs only for now
# This has to be set before project()
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

project(bitsandbytes LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Set global flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
add_compile_definitions(BUILD_CUDA)

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

# pthread does not exist on Windows
if(WIN32)
    include(ExternalProject)
    ExternalProject_Add(pthread-win32
        GIT_REPOSITORY https://github.com/GerHobbelt/pthread-win32
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    )

    include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
    link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include)

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

if(WIN32)
    # pthread does not exist on Windows
    add_dependencies(bitsandbytes pthread-win32)
    target_link_libraries(bitsandbytes CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusparse pthreadVC3)
else()
    target_link_libraries(bitsandbytes CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusparse)
endif()

# Set the output name of the CUDA library
set_target_properties(bitsandbytes PROPERTIES OUTPUT_NAME "bitsandbytes_cuda120")
