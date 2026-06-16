#!/bin/bash
set -xeuo pipefail

: "${RUNNER_OS:?RUNNER_OS must be set (Linux/Windows)}"
: "${ROCM_VERSION:?ROCM_VERSION must be set}"

bnb_rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1103"

# ROCm 6.4+ - Add RDNA4 and RDNA3.5 targets. Note we assume >=6.4.4.
[[ "${ROCM_VERSION}" == 6.4.* || "${ROCM_VERSION}" == 7.* ]] && bnb_rocm_arch="${bnb_rocm_arch};gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"

# ROCm 7.0+ - Add gfx950
[[ "${ROCM_VERSION}" == 7.* ]] && bnb_rocm_arch="${bnb_rocm_arch};gfx950"

if [ "${RUNNER_OS}" == "Linux" ]; then
    image=rocm/dev-ubuntu-22.04:${ROCM_VERSION}-complete
    echo "Using image $image"
    docker run --rm -i \
        -w /src -v "$PWD:/src" "$image" sh -c \
        "apt-get update \
      && pip install cmake==3.31.6 \
      && cmake -DCOMPUTE_BACKEND=hip -DCMAKE_BUILD_TYPE=MinSizeRel -DCMAKE_HIP_FLAGS=\"--offload-compress\" -DBNB_ROCM_ARCH=\"${bnb_rocm_arch}\" . \
      && cmake --build ."
else
    bnb_rocm_arch="gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

    # Install ROCm SDK wheels from repo.radeon.com.
    rocm_base_url="https://repo.radeon.com/rocm/windows/rocm-rel-${ROCM_VERSION}"
    pip install \
        "${rocm_base_url}/rocm_sdk_core-${ROCM_VERSION}-py3-none-win_amd64.whl" \
        "${rocm_base_url}/rocm_sdk_devel-${ROCM_VERSION}-py3-none-win_amd64.whl" \
        "${rocm_base_url}/rocm_sdk_libraries_custom-${ROCM_VERSION}-py3-none-win_amd64.whl" \
        "${rocm_base_url}/rocm-${ROCM_VERSION}.tar.gz"

    # Expand the devel tarball
    rocm-sdk init

    ROCM_PATH="$(rocm-sdk path --root | tr '\\' '/')"
    export ROCM_PATH PATH="${ROCM_PATH}/bin:${PATH}"

    cmake -G Ninja \
        -DCOMPUTE_BACKEND=hip \
        -DBNB_ROCM_ARCH="${bnb_rocm_arch}" \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DCMAKE_HIP_FLAGS="--offload-compress" \
        -DCMAKE_HIP_COMPILER_ROCM_ROOT="${ROCM_PATH}" \
        -S .
    cmake --build .
fi

output_dir="output/${RUNNER_OS}/X64"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
