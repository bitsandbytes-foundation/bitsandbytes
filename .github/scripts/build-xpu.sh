#!/bin/bash
set -xeuo pipefail

: "${RUNNER_OS:?RUNNER_OS must be set (Linux/Windows)}"
: "${ONEAPI_VERSION:?ONEAPI_VERSION must be set (2025/2026)}"

case "${ONEAPI_VERSION}" in
    2025) image=intel/deep-learning-essentials:2025.1.3-0-devel-ubuntu22.04 ;;
    2026) image=intel/deep-learning-essentials:2026.0.0-devel-ubuntu22.04 ;;
    *) echo "Unsupported ONEAPI_VERSION: ${ONEAPI_VERSION}"; exit 1 ;;
esac

# We currently only build XPU on Linux x64 and Windows x64.
if [ "${RUNNER_OS}" == "Linux" ]; then
    # TODO: We might want to pre-build this as our own customized image in the future.
    echo "Using image $image"
    docker run --rm -i \
        -w /src -v "$PWD:/src" "$image" sh -c \
        "apt-get update \
      && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake bison intel-fw-gpu intel-ocloc \
      && cmake -DCOMPUTE_BACKEND=xpu . \
      && cmake --build . --config Release --parallel"
fi

output_dir="output/${RUNNER_OS}/X64"
mkdir -p "${output_dir}"
(shopt -s nullglob && cp bitsandbytes/*.{so,dylib,dll} "${output_dir}")
