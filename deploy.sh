#!/bin/bash

rm -rf dist build
make clean
CUDA_HOME=/usr/local/cuda-10.2 make
CUDA_VERSION=102 python -m build
python -m twine upload --repository testpypi dist/* --verbose

rm -rf dist build
make clean
CUDA_HOME=/usr/local/cuda-11.1 make
CUDA_VERSION=111 python -m build
python -m twine upload --repository testpypi dist/* --verbose
