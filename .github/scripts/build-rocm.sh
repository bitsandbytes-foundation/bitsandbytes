#!/bin/bash
declare build_arch
declare build_os
declare rocm_version

set -xeuo pipefail
bnb_rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1103"

# ROCm 6.4+ - Add RDNA4 and RDNA3.5 targets. Note we assume >=6.4.4.
[[ "${rocm_version}" == 6.4.* || "${rocm_version}" == 7.* ]] && bnb_rocm_arch="${bnb_rocm_arch};gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"

# ROCm 7.0+ - Add gfx950
[[ "${rocm_version}" == 7.* ]] && bnb_rocm_arch="${bnb_rocm_arch};gfx950"

if [ "${build_os:0:6}" == ubuntu ]; then
    image=rocm/dev-ubuntu-22.04:${rocm_version}-complete
    echo "Using image $image"
    docker run --rm --platform "linux/$build_arch" -i \
        -w /src -v "$PWD:/src" "$image" sh -c \
        "apt-get update \
      && pip install cmake==3.31.6 \
      && cmake -DCOMPUTE_BACKEND=hip -DCMAKE_BUILD_TYPE=MinSizeRel -DCMAKE_HIP_FLAGS=\"--offload-compress\" -DBNB_ROCM_ARCH=\"${bnb_rocm_arch}\" . \
      && cmake --build ."
else
    bnb_rocm_arch="gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

    pip install ninja cmake==3.31.6

    # Install ROCm SDK wheels from repo.radeon.com.
    rocm_base_url="https://repo.radeon.com/rocm/windows/rocm-rel-${rocm_version}"
    pip install \
        "${rocm_base_url}/rocm_sdk_core-${rocm_version}-py3-none-win_amd64.whl" \
        "${rocm_base_url}/rocm_sdk_devel-${rocm_version}-py3-none-win_amd64.whl" \
        "${rocm_base_url}/rocm_sdk_libraries_custom-${rocm_version}-py3-none-win_amd64.whl" \
        "${rocm_base_url}/rocm-${rocm_version}.tar.gz"

    # Expand the devel tarball
    rocm-sdk init

    ROCM_PATH="$(rocm-sdk path --root)"
    export ROCM_PATH
    export PATH="${ROCM_PATH}/bin:${PATH}"

    cmake -G Ninja \
        -DCOMPUTE_BACKEND=hip \
        -DBNB_ROCM_ARCH="${bnb_rocm_arch}" \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DCMAKE_HIP_FLAGS="--offload-compress" \
        -S .
    cmake --build .
fi

output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
