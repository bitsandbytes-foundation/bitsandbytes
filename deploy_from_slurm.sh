#!/bin/bash
BASE_PATH=$1

module unload cuda
module unload gcc

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=
make cpuonly

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=cpu python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.0
make cuda110

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=110 python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.1
make cuda11x

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=111 python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.2
make cuda11x

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=112 python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.3
make cuda11x

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=113 python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.4
make cuda11x

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=114 python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.5
make cuda11x

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=115 python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.6

make cuda11x
if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=116 python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.7
make cuda11x

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=117 python -m build
python -m twine upload dist/* --verbose


rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-10.2
make cuda10x_nomatmul

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=102-nomatmul python -m build
python -m twine upload dist/* --verbose


rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.0
make cuda110_nomatmul

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=110-nomatmul python -m build
python -m twine upload dist/* --verbose


rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.1
make cuda11x_nomatmul

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=111-nomatmul python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.2
make cuda11x_nomatmul

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=112-nomatmul python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.3
make cuda11x_nomatmul

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=113-nomatmul python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.4
make cuda11x_nomatmul

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=114-nomatmul python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.5
make cuda11x_nomatmul

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=115-nomatmul python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.6

make cuda11x_nomatmul
if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=116-nomatmul python -m build
python -m twine upload dist/* --verbose

rm -rf dist build
make clean
make cleaneggs
export CUDA_HOME=$BASE_PATH/cuda-11.7
make cuda11x_nomatmul

if [ ! -f "./bitsandbytes/libbitsandbytes.so" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  echo "Compilation unsuccessul!" 1>&2
  exit 64
fi
CUDA_VERSION=117-nomatmul python -m build
python -m twine upload dist/* --verbose
