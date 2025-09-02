#!/bin/bash
declare build_arch
declare build_os
declare cuda_version
declare cuda_targets

set -xeuo pipefail

if [[ -v cuda_targets ]]; then
    build_capability="${cuda_targets}"
elif [ "${build_arch}" = "aarch64" ]; then
    build_capability="75;80;90"

    # CUDA 12.8+: Add sm100/sm120
    [[ "${cuda_version}" == 12.8.* || "${cuda_version}" == 12.9.* ]] && build_capability="75;80;90;100;120"
else
    # By default, target Maxwell through Hopper.
    build_capability="50;60;70;75;80;86;89;90"

    # CUDA 12.8+: Add sm100 and sm120; remove < sm70 to align with PyTorch 2.8+cu128 minimum
    [[ "${cuda_version}" == 12.8.* || "${cuda_version}" == 12.9.* ]] && build_capability="70;75;80;86;89;90;100;120"
fi

[[ "${build_os}" = windows-* ]] && python3 -m pip install ninja

if [ "${build_os:0:6}" == ubuntu ]; then
    # We'll use Rocky Linux 8 in order to maintain manylinux 2.24 compatibility.
    image="nvidia/cuda:${cuda_version}-devel-rockylinux8"
    echo "Using image $image"

    docker run -i -w /src -v "$PWD:/src" "$image" bash -c \
        "dnf update -y \
        && dnf install cmake gcc-toolset-11 -y \
        && source scl_source enable gcc-toolset-11 \
        && cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=\"${build_capability}\" . \
        && cmake --build . --config Release"
else
    pip install cmake==3.28.3
    cmake -G Ninja -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="${build_capability}" -DCMAKE_BUILD_TYPE=Release -S .
    cmake --build . --config Release
fi


output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
