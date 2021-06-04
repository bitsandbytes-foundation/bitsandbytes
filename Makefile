MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))
NPROCS:=$(shell grep -c ^processor /proc/cpuinfo)


GPP:= /usr/bin/g++
###########################################
# SPECIFY ANACONDA PATH IF THIS PATH DIFFERS FROM YOUR CONFIG
ANACONDA_HOME = $(HOME)/anaconda3
###########################################

ROOT_DIR_CCP := $(ROOT_DIR)/cpp_source
ROOT_DIR_CU := $(ROOT_DIR)/cuda_source
BUILD_DIR:= $(ROOT_DIR)/cuda_build

FILES := $(ROOT_DIR_CU)/basicOps.cu $(ROOT_DIR_CU)/clusterKernels.cu
FILES_CPP := $(ROOT_DIR_CCP)/pythonInterface.c

# General compilation flags

INCLUDE :=  -I /usr/local/cuda/include -I $(ROOT_DIR)/include -I $(ANACONDA_HOME)/include -I $(ROOT_DIR)/dependencies/cub
LIB := -L /usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lcurand -lcusparse -lhdf5 -L $(ANACONDA_HOME)/lib

# NVIDIA NVCC compilation flags
#COMPUTE_CAPABILITY := -gencode arch=compute_52,code=sm_52 # Maxwell
COMPUTE_CAPABILITY := -gencode arch=compute_75,code=sm_75 # Turing

all: $(ROOT_DIR)/dependencies/cub $(BUILD_DIR) $(HOME)/anaconda3
	nvcc $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dc $(FILES) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	nvcc $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++11 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libClusterNet.so $(LIB)

$(BUILD_DIR):
	mkdir -p cuda_build
	mkdir -p dependencies

$(ROOT_DIR)/dependencies/cub:
	git clone https://github.com/NVlabs/cub $(ROOT_DIR)/dependencies/cub

clean:
	rm cuda_build/* ./bitsandbytes/libClusterNet.so
