#!/bin/bash
declare build_arch
declare build_os

set -xeuo pipefail

if [[ "${build_os}" == windows* ]]; then
    pip install cmake==3.30.9
else
    pip install cmake==3.28.3
fi

if [ "${build_os:0:5}" == macos ] && [ "${build_arch}" == aarch64 ]; then
	cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DCOMPUTE_BACKEND=cpu .
else
	cmake -DCOMPUTE_BACKEND=cpu .
fi
cmake --build . --config Release

output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
