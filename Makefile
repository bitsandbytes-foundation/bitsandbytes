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
BUILD_DIR:= $(ROOT_DIR)/build

FILES := $(ROOT_DIR_CU)/basicOps.cu $(ROOT_DIR_CU)/clusterKernels.cu
FILES_CPP := $(ROOT_DIR_CCP)/pythonInterface.c

# General compilation flags

INCLUDE :=  -I /usr/local/cuda/include -I $(ROOT_DIR)/include -I $(ANACONDA_HOME)/include -I $(ROOT_DIR)/dependencies/cub
LIB := -L /usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lcurand -lcusparse -lhdf5 -L $(ANACONDA_HOME)/lib

# NVIDIA NVCC compilation flags
#COMPUTE_CAPABILITY := -gencode arch=compute_52,code=sm_52 # Maxwell
COMPUTE_CAPABILITY := -gencode arch=compute_75,code=sm_75 # Turing

all: $(ROOT_DIR)/dependencies/cub $(ROOT_DIR)/build $(HOME)/anaconda3
	nvcc $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dc $(FILES) $(INCLUDE) $(LIB) --output-directory $(ROOT_DIR)/build
	nvcc $(COMPUTE_CAPABILITY) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o -o $(BUILD_DIR)/link.o 
	$(GPP) -std=c++11 -shared -fPIC $(INCLUDE) $(BUILD_DIR)/basicOps.o $(BUILD_DIR)/clusterKernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libClusterNet.so $(LIB)

$(ROOT_DIR)/build:
	mkdir -p build
	mkdir -p dependencies

$(ROOT_DIR)/dependencies/cub:
	git clone https://github.com/NVlabs/cub $(ROOT_DIR)/dependencies/cub

test:
	$(GPP) -DHDF5 -std=c++11 $(INCLUDE) -L $(ROOT_DIR)/lib $(ROOT_DIR)/main.cpp -o main $(LIB) $(FLAGS_GPU) -lClusterNet

########
#WIKIPEDIA
########
wiki: $(ROOT_DIR)/data/wiki/enwiki-20170820-pages-articles.xml.bz2 $(ROOT_DIR)/dependencies/wikiextractor $(ROOT_DIR)/data/wiki/raw/AA

$(ROOT_DIR)/data/wiki/raw/AA:
	python dependencies/wikiextractor/WikiExtractor.py \
		 data/wiki/enwiki-20170820-pages-articles.xml.bz2 \
		-o $(ROOT_DIR)/data/wiki/raw/ \
		-b 500M \
		--min_text_length 300 \
		--processes $(NPROCS) \
		--filter_disambig_pages \
		--json

$(ROOT_DIR)/data/wiki/enwiki-20170820-pages-articles.xml.bz2:
	echo "Please save the wikipedia dump in the following path"
	echo $(ROOT_DIR)/data/wiki/
	echo "Close Transmission when the download is completed"
	mkdir -p data/wiki
	#wget http://rover.ms.mff.cuni.cz/~pasky/brmson/enwiki-20150112-pages-articles.xml.bz2 -O data/wiki/enwiki-20150112-pages-articles.xml.bz2
	#transmission-gtk http://itorrents.org/torrent/D567CE8E2EC4792A99197FB61DEAEBD70ADD97C0.torrent
	#if [ $(HOME)/Downloads/enwiki-20170820-pages-articles.xml.bz2 ] then \
	#	mv $(HOME)/Downloads/enwiki-20170820-pages-articles.xml.bz2 $(ROOT_DIR)/data/wiki/; \
	#fi

$(ROOT_DIR)/dependencies/wikiextractor: $(ROOT_DIR)/dependencies
	git clone https://github.com/attardi/wikiextractor $(ROOT_DIR)/dependencies/wikiextractor

clean:
	rm build/* ./bitsandbytes/libClusterNet.so
