#!/bin/bash
set -xeuo pipefail

rocm_version_at_least() {
    local required_major required_minor

    IFS=. read -r required_major required_minor <<< "$1"
    ((rocm_version_major > required_major ||
        (rocm_version_major == required_major && rocm_version_minor >= required_minor)))
}

if [[ "${RUNNER_OS:-}" != "Linux" && "${RUNNER_OS:-}" != "Windows" ]]; then
    echo "Invalid RUNNER_OS '${RUNNER_OS:-<unset>}'; expected Linux or Windows." >&2
    exit 1
fi

if [[ ! "${ROCM_VERSION:-}" =~ ^([0-9]+)\.([0-9]+)(\.[0-9]+)?$ ]]; then
    echo "Invalid ROCM_VERSION '${ROCM_VERSION:-<unset>}'; expected a dotted ROCm release." >&2
    exit 1
fi

rocm_version_major="$((10#${BASH_REMATCH[1]}))"
rocm_version_minor="$((10#${BASH_REMATCH[2]}))"
rocm_version_tag="${rocm_version_major}${rocm_version_minor}"

bnb_rocm_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1103"

# ROCm 6.4+ - Add RDNA4 and RDNA3.5 targets. Note we assume >=6.4.4.
if rocm_version_at_least "6.4"; then
    bnb_rocm_arch="${bnb_rocm_arch};gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
fi

# ROCm 7.0+ - Add gfx950
if rocm_version_at_least "7.0"; then
    bnb_rocm_arch="${bnb_rocm_arch};gfx950"
fi

# ROCm 7.14+ - Add CDNA1, CDNA5, and RDNA2 targets.
if rocm_version_at_least "7.14"; then
    bnb_rocm_arch="${bnb_rocm_arch};gfx908;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1250"
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
      && cmake -DCOMPUTE_BACKEND=hip -DROCM_VERSION=\"${ROCM_VERSION}\" -DCMAKE_BUILD_TYPE=MinSizeRel -DCMAKE_HIP_FLAGS=\"--offload-compress\" -DBNB_ROCM_ARCH=\"${bnb_rocm_arch}\" . \
      && cmake --build . --parallel"
else
    bnb_rocm_arch="gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

    if rocm_version_at_least "7.14"; then
        # Add RDNA2 and additional RDNA3.5 targets.
        bnb_rocm_arch="${bnb_rocm_arch};gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1152;gfx1153"

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
        -DROCM_VERSION="${ROCM_VERSION}" \
        -DBNB_ROCM_ARCH="${bnb_rocm_arch}" \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DCMAKE_HIP_FLAGS="--offload-compress" \
        -DCMAKE_HIP_COMPILER_ROCM_ROOT="${ROCM_PATH}" \
        -S .
    cmake --build .
fi

output_dir="output/${RUNNER_OS}/X64"
mkdir -p "${output_dir}"

shopt -s nullglob
libraries=(bitsandbytes/libbitsandbytes_rocm${rocm_version_tag}.{so,dylib,dll})
shopt -u nullglob

if [ "${#libraries[@]}" -eq 0 ]; then
    expected_pattern="bitsandbytes/libbitsandbytes_rocm${rocm_version_tag}.{so,dylib,dll}"
    echo "Expected ROCm ${ROCM_VERSION} library was not built: ${expected_pattern}" >&2
    exit 1
fi

cp "${libraries[@]}" "${output_dir}/"
