#!/bin/bash
set -xeuo pipefail

: "${RUNNER_OS:?RUNNER_OS must be set (Linux/Windows/macOS)}"
: "${RUNNER_ARCH:?RUNNER_ARCH must be set (X64/ARM64)}"

if [ "${RUNNER_OS}" == "macOS" ] && [ "${RUNNER_ARCH}" == "ARM64" ]; then
	cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DCOMPUTE_BACKEND=cpu .
else
	cmake -DCOMPUTE_BACKEND=cpu .
fi
cmake --build . --config Release

output_dir="output/${RUNNER_OS}/${RUNNER_ARCH}"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
