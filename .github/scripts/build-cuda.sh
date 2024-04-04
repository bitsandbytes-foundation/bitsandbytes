#!/bin/bash
declare build_arch
declare build_os
declare cuda_version

set -xeuo pipefail
build_capability="50;52;60;61;70;75;80;86;89;90"
[[ "${cuda_version}" == 11.7.* ]] && build_capability=${build_capability%??????}
[[ "${cuda_version}" == 11.8.* ]] && build_capability=${build_capability%???}
[[ "${build_os}" = windows-* ]] && python3 -m pip install ninja
for NO_CUBLASLT in ON OFF; do
	if [ "${build_os:0:6}" == ubuntu ]; then
		image=nvidia/cuda:${cuda_version}-devel-ubuntu22.04
		echo "Using image $image"
		docker run --platform "linux/$build_arch" -i -w /src -v "$PWD:/src" "$image" sh -c \
			"apt-get update \
      && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake \
      && cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=\"${build_capability}\" -DNO_CUBLASLT=${NO_CUBLASLT} . \
      && cmake --build ."
	else
		pip install cmake==3.28.3
		cmake -G Ninja -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="${build_capability}" -DNO_CUBLASLT=${NO_CUBLASLT} -DCMAKE_BUILD_TYPE=Release -S .
		cmake --build . --config Release
	fi
done

output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
