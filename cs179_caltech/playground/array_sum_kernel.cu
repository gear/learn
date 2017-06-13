#include <cassert>
#include <cuda_runtime.h>
#include "array_sum_kernel.cuh"

__global__
void gpu_divergence(const int* input, int* output, n) {
    extern __shared__ int s[];

    // Prepare the indices and data
    unsigned tid = threadIdx.x;
    unsigned g_id = blockIdx.x * blockDim.x + tid;
    s[tid] = input[g_id];
    __syncthreads();

    for (int d=0; d < blockDim.x; d *= 2) {
        if (tid % (2*d) == 0) {
            s[tid] += s[tid+d];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = s[0];
}
