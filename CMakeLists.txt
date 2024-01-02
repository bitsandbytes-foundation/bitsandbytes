cmake_minimum_required(VERSION 3.22.1)

option(ENABLE_CUDA "Build for CUDA (Nvidia)" OFF)
option(ENABLE_MPS "Build for Metal Performance Shaders (Apple)" OFF)

if(ENABLE_CUDA)
    if(APPLE)
        message(FATAL_ERROR "CUDA is not supported on macOS" )
    endif()
    option(NO_CUBLASLT "Don't use CUBLAST" OFF)
    if(NO_CUBLASLT)
        set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()
endif()

if(ENABLE_CUDA)
    message("Building CUDA support for ${CMAKE_CUDA_ARCHITECTURES}")
    # Find CUDA tools if we are compiling with CUDA
    find_package(CUDAToolkit REQUIRED)
    if(NO_CUBLASLT)
        set(LIBSUFFIX "cuda${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}_nocublaslt")
    else()
        set(LIBSUFFIX "cuda${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}")
    endif()

    project(bitsandbytes LANGUAGES CXX CUDA)
    add_compile_definitions(BUILD_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(GPU_SOURCES csrc/ops.cu csrc/kernels.cu)
elseif(ENABLE_MPS)
    if(NOT APPLE)
        message(FATAL_ERROR "MPS is only supported on macOS" )
    endif()
    message("Building MPS support")
    set(LIBSUFFIX "mps")
    project(bitsandbytes LANGUAGES CXX OBJCXX)
    add_compile_definitions(BUILD_MPS)
    set(METAL_SOURCES csrc/mps_kernels.metal)
    file(MAKE_DIRECTORY "build")
    add_custom_command(OUTPUT "bitsandbytes/bitsandbytes.metallib"  
                   COMMAND xcrun metal -c -o "build/bitsandbytes.air" ${METAL_SOURCES}
                   COMMAND xcrun metallib "build/bitsandbytes.air" -o "bitsandbytes/bitsandbytes.metallib"
                   DEPENDS "${METAL_SOURCES}"  
                   COMMENT "Compiling Metal kernels"  
                   VERBATIM)
    add_custom_target(metallib DEPENDS "bitsandbytes/bitsandbytes.metallib")
    set(GPU_SOURCES csrc/mps_ops.mm)
else()
    message("Building with CPU only")
    set(LIBSUFFIX "cpu")

    project(bitsandbytes LANGUAGES CXX)
    set(GPU_SOURCES)
endif()

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 13.1)
endif()
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED C++14)

if(WIN32)
    # Mute warnings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=177")

    # Enable fast math on VC++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")

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
	${GPU_SOURCES}
	csrc/common.cpp
	csrc/cpu_ops.cpp
	csrc/pythonInterface.cpp)

target_include_directories(bitsandbytes PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(ENABLE_CUDA)
    target_include_directories(bitsandbytes PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/include)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

    set_target_properties(
        bitsandbytes
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(bitsandbytes CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusparse)
endif()
if(ENABLE_MPS)
    add_dependencies(bitsandbytes metallib)
    target_link_libraries(bitsandbytes objc "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph")
endif()

set_target_properties(bitsandbytes PROPERTIES OUTPUT_NAME "bitsandbytes_${LIBSUFFIX}")
# Set the output name of the CUDA library
if(MSVC)
set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE bitsandbytes)
set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG bitsandbytes)
set_target_properties(bitsandbytes PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE bitsandbytes)
set_target_properties(bitsandbytes PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG bitsandbytes)
endif()

set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY bitsandbytes)
