#!/bin/bash
set -xeuo pipefail

: "${RUNNER_OS:?RUNNER_OS must be set (Linux/Windows)}"
: "${ROCM_VERSION:?ROCM_VERSION must be set}"

rocm_version_at_least() {
    local required_version="$1"
    local current_major current_minor required_major required_minor

    IFS=. read -r current_major current_minor _ <<< "${ROCM_VERSION}"
    IFS=. read -r required_major required_minor _ <<< "${required_version}"

    if ((current_major > required_major)); then
        return 0
    fi
    if ((current_major < required_major)); then
        return 1
    fi
    if ((current_minor >= required_minor)); then
        return 0
    fi
    return 1
}

# Baseline: GCN 5.1, CDNA2, CDNA3, RDNA2, RDNA3.
bnb_rocm_arch="gfx906;gfx906:sramecc-;gfx90a;gfx90a:sramecc-;gfx942;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103"

# ROCm 6.4+ - Add CDNA1, RDNA1, RDNA3.5, and RDNA4 and targets. Note we assume >=6.4.4.
if rocm_version_at_least "6.4"; then
    bnb_rocm_arch="${bnb_rocm_arch};gfx908;gfx1010;gfx1011;gfx1012;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
fi

# ROCm 7.0+ - Add gfx950 (CDNA4)
if rocm_version_at_least "7.0"; then
    bnb_rocm_arch="${bnb_rocm_arch};gfx950"
fi

if [ "${RUNNER_OS}" == "Linux" ]; then
    image_suffix="complete"
    if rocm_version_at_least "7.14"; then
        image_suffix="full"
    fi
    image=rocm/dev-ubuntu-22.04:${ROCM_VERSION}-${image_suffix}
    echo "Using image $image"
    docker run --rm -i \
        -w /src -v "$PWD:/src" "$image" sh -c \
        "pip install cmake==3.31.6 \
      && cmake -DCOMPUTE_BACKEND=hip -DCMAKE_BUILD_TYPE=MinSizeRel -DCMAKE_HIP_FLAGS=\"--offload-compress\" -DBNB_ROCM_ARCH=\"${bnb_rocm_arch}\" . \
      && cmake --build . --parallel"
else
    bnb_rocm_arch="gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

    if rocm_version_at_least "7.14"; then
        # Additional RDNA3.5 targets.
        bnb_rocm_arch="${bnb_rocm_arch};gfx1152;gfx1153"

        pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,devel]==${ROCM_VERSION}"
    else
        # Install ROCm SDK wheels from repo.radeon.com.
        rocm_base_url="https://repo.radeon.com/rocm/windows/rocm-rel-${ROCM_VERSION}"
        pip install \
            "${rocm_base_url}/rocm_sdk_core-${ROCM_VERSION}-py3-none-win_amd64.whl" \
            "${rocm_base_url}/rocm_sdk_devel-${ROCM_VERSION}-py3-none-win_amd64.whl" \
            "${rocm_base_url}/rocm_sdk_libraries_custom-${ROCM_VERSION}-py3-none-win_amd64.whl" \
            "${rocm_base_url}/rocm-${ROCM_VERSION}.tar.gz"

    fi

    # Expand the devel tarball.
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
