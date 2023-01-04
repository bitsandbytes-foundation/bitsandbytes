#!/bin/bash
BASE_PATH=$1

echo "MAKE SURE LD_LIBRARY_PATH IS EMPTY!"
echo $LD_LIBRARY_PATH

if [[ ! -z "${LD_LIBRARY_PATH}" ]]; then
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi


module unload cuda
module unload gcc

rm -rf dist build
make cleaneggs
make cleanlibs

make clean
export CUDA_HOME=
export CUDA_VERSION=
make cpuonly CUDA_VERSION="CPU"

if [ ! -f "./bitsandbytes/libbitsandbytes_cpu.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.0
make cuda110 CUDA_VERSION=110

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda110.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.1
make cuda11x CUDA_VERSION=111

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda111.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.2
make cuda11x CUDA_VERSION=112

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda112.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.3
make cuda11x CUDA_VERSION=113

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda113.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.4
make cuda11x CUDA_VERSION=114

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda114.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.5
make cuda11x CUDA_VERSION=115

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda115.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.6

make cuda11x CUDA_VERSION=116
if [ ! -f "./bitsandbytes/libbitsandbytes_cuda116.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.7
make cuda11x CUDA_VERSION=117

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda117.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.8
make cuda12x CUDA_VERSION=118

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda118.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-12.0
make cuda12x CUDA_VERSION=120

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda120.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi


make clean
export CUDA_HOME=$BASE_PATH/cuda-10.2
make cuda10x_nomatmul CUDA_VERSION=102

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda102_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi


make clean
export CUDA_HOME=$BASE_PATH/cuda-11.0
make cuda110_nomatmul CUDA_VERSION=110

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda110_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi


make clean
export CUDA_HOME=$BASE_PATH/cuda-11.1
make cuda11x_nomatmul CUDA_VERSION=111

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda111_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.2
make cuda11x_nomatmul CUDA_VERSION=112

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda112_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.3
make cuda11x_nomatmul CUDA_VERSION=113

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda113_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.4
make cuda11x_nomatmul CUDA_VERSION=114

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda114_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.5
make cuda11x_nomatmul CUDA_VERSION=115

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda115_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.6

make cuda11x_nomatmul CUDA_VERSION=116
if [ ! -f "./bitsandbytes/libbitsandbytes_cuda116_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.7
make cuda11x_nomatmul CUDA_VERSION=117

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda117_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-11.8
make cuda12x_nomatmul CUDA_VERSION=118

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda118_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

make clean
export CUDA_HOME=$BASE_PATH/cuda-12.0
make cuda12x_nomatmul CUDA_VERSION=120

if [ ! -f "./bitsandbytes/libbitsandbytes_cuda120_nocublaslt.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi

python -m build
python -m twine upload dist/* --verbose
