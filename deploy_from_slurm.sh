#!/bin/bash
module unload cuda
module unload gcc

rm -rf dist build
make clean
make cleaneggs
module load cuda/9.2
module load gcc/7.3.0
CUDA_HOME=/public/apps/cuda/9.2
make
CUDA_VERSION=92 python -m build
python -m twine upload dist/* --verbose
module unload cuda


rm -rf dist build
make clean
make cleaneggs
module load cuda/10.0
CUDA_HOME=/public/apps/cuda/10.0
make cuda10x
CUDA_VERSION=100 python -m build
python -m twine upload dist/* --verbose
module unload cuda
module unload gcc
module load gcc/8.4

rm -rf dist build
make clean
make cleaneggs
module load cuda/10.1
CUDA_HOME=/public/apps/cuda/10.1
make cuda10x
CUDA_VERSION=101 python -m build
python -m twine upload dist/* --verbose
module unload cuda

rm -rf dist build
make clean
make cleaneggs
module load cuda/10.2
CUDA_HOME=/public/apps/cuda/10.2/
make cuda10x
CUDA_VERSION=102 python -m build
python -m twine upload dist/* --verbose
module unload cuda


rm -rf dist build
make clean
make cleaneggs
module load cuda/11.0
CUDA_HOME=/public/apps/cuda/11.0
make cuda110
CUDA_VERSION=110 python -m build
python -m twine upload dist/* --verbose
module unload cuda

rm -rf dist build
make clean
make cleaneggs
module load cuda/11.1
CUDA_HOME=/public/apps/cuda/11.1
make cuda11x
CUDA_VERSION=111 python -m build
python -m twine upload dist/* --verbose
module unload cuda

rm -rf dist build
make clean
make cleaneggs
module load cuda/11.2
CUDA_HOME=/public/apps/cuda/11.2
make cuda11x
CUDA_VERSION=112 python -m build
python -m twine upload dist/* --verbose
module unload cuda

rm -rf dist build
make clean
make cleaneggs
CUDA_HOME=/private/home/timdettmers/git/autoswap/local/cuda-11.3 make cuda11x
CUDA_VERSION=113 python -m build
python -m twine upload dist/* --verbose
module unload cuda
