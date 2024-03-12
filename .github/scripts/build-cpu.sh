#!/bin/bash
declare build_arch
declare build_os

set -xeuo pipefail

pip install cmake==3.28.3

if [ "${build_os:0:6}" == ubuntu ] && [ "${build_arch}" == aarch64 ]; then
	# Allow cross-compile on aarch64
	sudo apt-get update
	sudo apt-get install -y gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu g++-aarch64-linux-gnu
	cmake -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCOMPUTE_BACKEND=cpu .
elif [ "${build_os:0:5}" == macos ] && [ "${build_arch}" == aarch64 ]; then
	cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DCOMPUTE_BACKEND=cpu .
else
	cmake -DCOMPUTE_BACKEND=cpu .
fi
cmake --build . --config Release

output_dir="output/${build_os}/${build_arch}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
