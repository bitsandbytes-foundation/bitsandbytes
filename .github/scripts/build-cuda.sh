#!/bin/bash
declare build_arch
declare build_os
declare cuda_version
declare cuda_targets

set -xeuo pipefail

if [[ -v cuda_targets ]]; then
    build_capability="${cuda_targets}"
else
    # By default, target Maxwell through Hopper.
    build_capability="50;52;60;61;70;75;80;86;89;90"

    # CUDA 12.8: Add sm100 and sm120; remove < sm75 to align with PyTorch 2.7+cu128 minimum
    [[ "${cuda_version}" == 12.8.* ]] && build_capability="75;80;86;89;90;100;120"
fi

[[ "${build_os}" = windows-* ]] && python3 -m pip install ninja

if [ "${build_os:0:6}" == ubuntu ]; then
    image=nvidia/cuda:${cuda_version}-devel-ubuntu22.04
    echo "Using image $image"
    docker run --platform "linux/$build_arch" -i -w /src -v "$PWD:/src" "$image" sh -c \
        "apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake \
    && cmake -DPTXAS_VERBOSE=1 -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=\"${build_capability}\" . \
    && cmake --build ."
else
    pip install cmake==3.28.3
    cmake -G Ninja -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="${build_capability}" -DCMAKE_BUILD_TYPE=Release -S .
    cmake --build . --config Release
fi


output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
