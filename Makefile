MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))
NPROCS:=$(shell grep -c ^processor /proc/cpuinfo)


GPP:= /usr/bin/g++
NVCC := $(CUDA_HOME)/bin/nvcc
###########################################
# SPECIFY ANACONDA PATH IF THIS PATH DIFFERS FROM YOUR CONFIG
ANACONDA_HOME = $(HOME)/anaconda3
###########################################

ROOT_DIR_CCP := $(ROOT_DIR)/cpp_source
ROOT_DIR_CU := $(ROOT_DIR)/cuda_source
BUILD_DIR:= $(ROOT_DIR)/cuda_build

FILES := $(ROOT_DIR_CU)/basicOps.cu $(ROOT_DIR_CU)/clusterKernels.cu
FILES_CPP := $(ROOT_DIR_CCP)/pythonInterface.c

INCLUDE :=  -I $(CUDA_HOME)/include -I $(ROOT_DIR)/include -I $(ANACONDA_HOME)/include -I $(ROOT_DIR)/dependencies/cub
LIB := -L $(CUDA_HOME)/lib64 -lcudart -lcuda -lcublas -lcurand -lcusparse -L $(ANACONDA_HOME)/lib

# NVIDIA NVCC compilation flags
#COMPUTE_CAPABILITY := -gencode arch=compute_50,code=sm_50 # Maxwell
#COMPUTE_CAPABILITY += -gencode arch=compute_52,code=sm_52 # Maxwell
#COMPUTE_CAPABILITY += -gencode arch=compute_61,code=sm_61 # Pascal
#COMPUTE_CAPABILITY += -gencode arch=compute_70,code=sm_70 # Volta
#COMPUTE_CAPABILITY += -gencode arch=compute_72,code=sm_72 # Volta 
COMPUTE_CAPABILITY := -gencode arch=compute_75,code=sm_75 # Volta 

all: $(ROOT_DIR)/dependencies/cub $(BUILD_DIR) $(HOME)/anaconda3
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++11 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libBitsNBytes.so $(LIB)

cuda92: $(ROOT_DIR)/dependencies/cub $(BUILD_DIR) $(HOME)/anaconda3
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++11 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libBitsNBytes.so $(LIB)

cuda10x: $(ROOT_DIR)/dependencies/cub $(BUILD_DIR) $(HOME)/anaconda3
	$(NVCC) $(COMPUTE_CAPABILITY) -gencode arch=compute_75,code=sm_75 -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++11 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libBitsNBytes.so $(LIB)

cuda110: $(ROOT_DIR)/dependencies/cub $(BUILD_DIR) $(HOME)/anaconda3
	$(NVCC) $(COMPUTE_CAPABILITY) -gencode arch=compute_80,code=sm_80 -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++11 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libBitsNBytes.so $(LIB)

cuda111: $(ROOT_DIR)/dependencies/cub $(BUILD_DIR) $(HOME)/anaconda3
	$(NVCC) $(COMPUTE_CAPABILITY) -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++11 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libBitsNBytes.so $(LIB)

cuda113: $(BUILD_DIR) $(HOME)/anaconda3
	$(NVCC) $(COMPUTE_CAPABILITY) -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES) -I $(CUDA_HOME)/include -I $(ROOT_DIR)/include -I $(ANACONDA_HOME)/include $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++11 -shared -fPIC -I $(CUDA_HOME)/include -I $(ROOT_DIR)/include -I $(ANACONDA_HOME)/include $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libBitsNBytes.so $(LIB)

$(BUILD_DIR):
	mkdir -p cuda_build
	mkdir -p dependencies

$(ROOT_DIR)/dependencies/cub:
	git clone https://github.com/NVlabs/cub $(ROOT_DIR)/dependencies/cub

clean:
	rm cuda_build/* ./bitsandbytes/libBitsNBytes.so
