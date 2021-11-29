MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

GPP:= /usr/bin/g++
ifeq ($(CUDA_HOME),)
	CUDA_HOME:= $(shell which nvcc | rev | cut -d'/' -f3- | rev)
endif
NVCC := $(CUDA_HOME)/bin/nvcc

###########################################

CSRC := $(ROOT_DIR)/csrc
BUILD_DIR:= $(ROOT_DIR)/cuda_build

FILES_CUDA := $(CSRC)/ops.cu $(CSRC)/kernels.cu
FILES_CPP := $(CSRC)/pythonInterface.c

INCLUDE :=  -I $(CUDA_HOME)/include -I $(ROOT_DIR)/csrc -I $(CONDA_PREFIX)/include -I $(ROOT_DIR)/dependencies/cub -I $(ROOT_DIR)/include
LIB := -L $(CUDA_HOME)/lib64 -lcudart -lcuda -lcublas -lcurand -lcusparse -L $(CONDA_PREFIX)/lib

# NVIDIA NVCC compilation flags
COMPUTE_CAPABILITY := -gencode arch=compute_35,code=sm_35 # Kepler 
COMPUTE_CAPABILITY += -gencode arch=compute_37,code=sm_37 # Kepler 
COMPUTE_CAPABILITY += -gencode arch=compute_50,code=sm_50 # Maxwell
COMPUTE_CAPABILITY += -gencode arch=compute_52,code=sm_52 # Maxwell
COMPUTE_CAPABILITY += -gencode arch=compute_60,code=sm_60 # Pascal
COMPUTE_CAPABILITY += -gencode arch=compute_61,code=sm_61 # Pascal
COMPUTE_CAPABILITY += -gencode arch=compute_70,code=sm_70 # Volta
COMPUTE_CAPABILITY += -gencode arch=compute_72,code=sm_72 # Volta 
COMPUTE_CAPABILITY += -gencode arch=compute_72,code=sm_72 # Volta 

# CUDA 9.2 supports CC 3.0, but CUDA >= 11.0 does not
CC_CUDA92 := -gencode arch=compute_30,code=sm_30

# Later versions of CUDA support the new architectures
CC_CUDA10x := -gencode arch=compute_30,code=sm_30
CC_CUDA10x += -gencode arch=compute_75,code=sm_75

CC_CUDA110 := -gencode arch=compute_75,code=sm_75
CC_CUDA110 += -gencode arch=compute_80,code=sm_80

CC_CUDA11x := -gencode arch=compute_75,code=sm_75
CC_CUDA11x += -gencode arch=compute_80,code=sm_80
CC_CUDA11x += -gencode arch=compute_86,code=sm_86

all: $(ROOT_DIR)/dependencies/cub $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes.so $(LIB)

cuda92: $(ROOT_DIR)/dependencies/cub $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA92) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA92) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes.so $(LIB)

cuda10x: $(ROOT_DIR)/dependencies/cub $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA10x) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA10x) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes.so $(LIB)

cuda110: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA110) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA110) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes.so $(LIB)

cuda11x: $(BUILD_DIR) env
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA11x) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
	$(NVCC) $(COMPUTE_CAPABILITY) $(CC_CUDA11x) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++14 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes.so $(LIB)

env:
	@echo "ENVIRONMENT"
	@echo "============================"
	@echo "NVCC path: $(NVCC)"
	@echo "GPP path: $(GPP) VERSION: `$(GPP) --version | head -n 1`"
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "CONDA_PREFIX: $(CONDA_PREFIX)"
	@echo "PATH: $(PATH)"
	@echo "LD_LIBRARY_PATH: $(LD_LIBRARY_PATH)"
	@echo "============================"

$(BUILD_DIR):
	mkdir -p cuda_build
	mkdir -p dependencies

$(ROOT_DIR)/dependencies/cub:
	git clone https://github.com/NVlabs/cub $(ROOT_DIR)/dependencies/cub
	cd dependencies/cub; git checkout 1.11.0

clean:
	rm cuda_build/* ./bitsandbytes/libbitsandbytes.so

cleaneggs:
	rm -rf *.egg*
