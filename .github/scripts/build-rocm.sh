#!/bin/bash
declare build_arch
declare build_os
declare rocm_version

set -xeuo pipefail
bnb_rocm_arch="gfx90a;gfx942;gfx1100;gfx1101"

# ROCm 6.4+ - Add gfx1150/gfx1151/gfx1200/gfx1201. Note we assume >=6.4.4.
[[ "${rocm_version}" == 6.4.* || "${rocm_version}" == 7.* ]] && bnb_rocm_arch="${bnb_rocm_arch};gfx1150;gfx1151;gfx1200;gfx1201"

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
fi

output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
