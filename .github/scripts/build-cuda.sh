#!/bin/bash
set -xeuo pipefail

: "${RUNNER_OS:?RUNNER_OS must be set (Linux/Windows/macOS)}"
: "${RUNNER_ARCH:?RUNNER_ARCH must be set (X64/ARM64)}"
: "${CUDA_VERSION:?CUDA_VERSION must be set}"

if [[ -v CUDA_TARGETS ]]; then
    build_capability="${CUDA_TARGETS}"
elif [ "${RUNNER_ARCH}" = "ARM64" ]; then
    build_capability="75;80;90"

    # CUDA 12.8-12.9: Add sm100/sm120
    [[ "${CUDA_VERSION}" == 12.8.* || "${CUDA_VERSION}" == 12.9.* ]] && build_capability="75;80;90;100;120"

    # CUDA 13.0+: Add sm100/sm110/sm120
    [[ "${CUDA_VERSION}" == 13.*.* ]] && build_capability="75;80;90;100;110;120;121"
else
    # By default, target Pascal through Hopper.
    build_capability="60;70;75;80;86;89;90"

    # CUDA 12.8+: Add sm100 and sm120; remove < sm70 to align with PyTorch 2.8+cu128 minimum
    [[ "${CUDA_VERSION}" == 12.8.* || "${CUDA_VERSION}" == 12.9.* ]] && build_capability="70;75;80;86;89;90;100;120"

    # CUDA 13.0+: Remove < sm75 to align with PyTorch 2.9+cu130 minimum
    [[ "${CUDA_VERSION}" == 13.*.* ]] && build_capability="75;80;86;89;90;100;120"
fi

if [ "${RUNNER_OS}" == "Linux" ]; then
    # We'll use Rocky Linux 8 in order to maintain manylinux 2.24 compatibility.
    image="nvidia/cuda:${CUDA_VERSION}-devel-rockylinux8"
    echo "Using image $image"

    docker run -i -w /src -v "$PWD:/src" "$image" bash -c \
        "dnf -y --refresh update --security \
        && dnf -y install cmake gcc-toolset-11-toolchain --setopt=install_weak_deps=False --setopt=tsflags=nodocs \
        && source scl_source enable gcc-toolset-11 \
        && cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=\"${build_capability}\" . \
        && cmake --build . --config Release --parallel"
else
    cmake -G Ninja -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="${build_capability}" -DCMAKE_BUILD_TYPE=Release -S .
    cmake --build . --config Release
fi

output_dir="output/${RUNNER_OS}/${RUNNER_ARCH}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
