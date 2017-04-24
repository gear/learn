CUDA_ROOT ?= /usr/local/cuda-8.0
NVCC ?= $(CUDA_ROOT)/bin/nvcc
CC := /usr/bin/g++
LD_CUDA_FLAGS := -lcudart
CUDA_INCLUDE := $(CUDA_ROOT)/include
CUDA_LIBRARY := $(CUDA_ROOT)/lib64

TARGETS = print_threads
all: $(TARGETS)

print_threads: print_threads.cpp print_kernel.o
	$(CC) $< -o $@ print_kernel.o -O3 -Wall -I$(CUDA_INCLUDE) -L$(CUDA_LIBRARY) $(LD_CUDA_FLAGS)

print_kernel.o: print_kernel.cu
	$(NVCC) -O3 -I$(CUDA_INC_PATH) -o $@ -L$(CUDA_LIBRARY) $(LD_CUDA_FLAGS) -c $<

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
