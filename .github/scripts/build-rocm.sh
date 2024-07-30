#!/bin/bash
declare build_arch
declare build_os
declare rocm_version

set -xeuo pipefail
bnb_rocm_arch="gfx90a;gfx942;gfx1100"
if [ "${build_os:0:6}" == ubuntu ]; then
	image=rocm/dev-ubuntu-22.04:${rocm_version}-complete
	echo "Using image $image"
	docker run --rm --platform "linux/$build_arch" -i \
		-w /src -v "$PWD:/src" "$image" sh -c \
		"apt-get update \
      && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake \
      && cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=\"${bnb_rocm_arch}\" . \
      && cmake --build ."
fi

output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
