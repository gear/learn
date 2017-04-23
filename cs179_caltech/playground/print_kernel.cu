#include <cstdio>
#include <cuda_runtime.h>
#include "print_kernel.cuh"


__global__ void cudaKernelFunc() {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Index: %d; thread: %d; block: %d; blockDim: %d\n",
				 index, threadIdx.x, blockIdx.x, blockDim.x);
}


void cudaCallKernel() {
	cudaKernelFunc<<<10,10>>>();
	cudaDeviceSynchronize();
}
