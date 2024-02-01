#!/bin/bash
BASE_PATH=$1

echo "MAKE SURE LD_LIBRARY_PATH IS EMPTY!"
echo $LD_LIBRARY_PATH

if [[ ! -z "${LD_LIBRARY_PATH}" ]]; then
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi


module unload cuda && echo "no module function available. Probably not on a slurm cluster."
module unload gcc && echo "no module function available. Probably not on a slurm cluster."

rm -rf dist build
make cleaneggs
make cleanlibs

rm -rf build/*
export CUDA_HOME=
export CUDA_VERSION=
make cpuonly CUDA_VERSION="CPU"

if [ ! -f "./bitsandbytes/libbitsandbytes_cpu.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.0
make cuda110 CUDA_VERSION=110

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda110.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.1
make cuda11x CUDA_VERSION=111

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda111.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.4
make cuda11x CUDA_VERSION=114

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda114.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.5
make cuda11x CUDA_VERSION=115

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda115.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.7
make cuda11x CUDA_VERSION=117

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda117.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.8
make cuda118 CUDA_VERSION=118

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda118.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-12.0
make cuda12x CUDA_VERSION=120

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda120.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-12.1
make cuda12x CUDA_VERSION=121

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda121.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-12.2
make cuda12x CUDA_VERSION=122

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda122.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-12.3
make cuda12x CUDA_VERSION=123

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda123.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

############################# START NO CUBLASLT #############################################
# binaries without 8-bit matmul support START HERE
# ###########################################################################################

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.0
make cuda110_nomatmul CUDA_VERSION=110

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda110_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi


rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.1
make cuda11x_nomatmul CUDA_VERSION=111

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda111_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.4
make cuda11x_nomatmul CUDA_VERSION=114

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda114_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.5
make cuda11x_nomatmul CUDA_VERSION=115

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda115_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.7
make cuda11x_nomatmul CUDA_VERSION=117

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda117_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-11.8
make cuda118_nomatmul CUDA_VERSION=118

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda118_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-12.0
make cuda12x_nomatmul CUDA_VERSION=120

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda120_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-12.1
make cuda12x_nomatmul CUDA_VERSION=121

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda121_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-12.2
make cuda12x_nomatmul CUDA_VERSION=122

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda122_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

rm -rf build/*
export CUDA_HOME=$BASE_PATH/cuda-12.3
make cuda12x_nomatmul CUDA_VERSION=123

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda123_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessful!" 1>&2
  exit 64
fi

python -m build
python -m twine upload dist/* --verbose
