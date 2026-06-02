#!/bin/bash
declare build_arch
declare build_os

set -xeuo pipefail

if [[ "${build_os}" == windows* ]]; then
    pip install cmake==3.30.9
else
    pip install cmake==3.28.3
fi

# Temporary: vectorization reporting
if [[ "${build_os}" == windows* ]]; then
    EXTRA_CXX_FLAGS="/Qvec-report:2 /Qpar-report:1"
elif [[ "${build_os:0:5}" == macos ]]; then
    EXTRA_CXX_FLAGS="-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize"
else
    EXTRA_CXX_FLAGS="-fopt-info-vec-missed -fopt-info-vec -fopt-info-loop-optimized"
fi

if [ "${build_os:0:5}" == macos ] && [ "${build_arch}" == aarch64 ]; then
	cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DCOMPUTE_BACKEND=cpu -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" .
else
	cmake -DCOMPUTE_BACKEND=cpu -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" .
fi
cmake --build . --config Release

output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
