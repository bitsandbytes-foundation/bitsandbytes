#!/bin/bash
set -xeuo pipefail

: "${RUNNER_OS:?RUNNER_OS must be set (Linux/Windows)}"

# We currently only build XPU on Linux x64 and Windows x64.
if [ "${RUNNER_OS}" == "Linux" ]; then
    # TODO: We might want to pre-build this as our own customized image in the future.
    image=intel/deep-learning-essentials:2025.1.3-0-devel-ubuntu22.04
    echo "Using image $image"
    docker run --rm -i \
        -w /src -v "$PWD:/src" "$image" sh -c \
        "apt-get update \
      && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake bison intel-fw-gpu intel-ocloc \
      && cmake -DCOMPUTE_BACKEND=xpu . \
      && cmake --build . --config Release"
fi

output_dir="output/${RUNNER_OS}/X64"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
