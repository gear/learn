/*
 * CUDA blur
 * Kevin Yuh, 2014
 * Revised by Nailen Matschke, 2016
 */

#include <cstdio>

#include <cuda_runtime.h>

#include "blur_device.cuh"

#define GTX_580_SM = 16

__global__
void cudaBlurKernel(const float *raw_data, const float *blur_v, float *out_data,
    int n_frames, int blur_v_size) {
	int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
	int indices_per_thread = ceilf(n_frames / (blockDim.x*gridDim.x))+1;
  int offset = global_thread_index * indices_per_thread;
  for(int i = 0; i < indices_per_thread; ++i) {
    int index = i + offset;
    if (index > n_frames) return;
    out_data[index] = 0;
    if (index < blur_v_size) {
  		for (int j = 0; j <= index; j++)
  			out_data[index] += raw_data[index - j] * blur_v[j];
  	} else {
  		for (int j = 0; j < blur_v_size; j++)
  			out_data[index] += raw_data[index - j] * blur_v[j];
  	}
  }
}

void cudaCallBlurKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *raw_data,
        const float *blur_v,
        float *out_data,
        const unsigned int n_frames,
        const unsigned int blur_v_size) {
	cudaBlurKernel<<<blocks,threadsPerBlock>>>(raw_data, blur_v, out_data, n_frames, blur_v_size);
	cudaDeviceSynchronize();
}
