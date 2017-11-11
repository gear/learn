#include <cassert>
#include <cuda_runtime.h>
#include "array_sum_kernel.cuh"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}


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

__global__
void gpu_divergence_main(const int* input, int* output,
                         int n, int gridSize, int blockSize) {
    assert(n % blockSize == 0)
    // First reduce - result: array of length gridSize
    gpu_divergence<<<gridSize,blockSize>>>(input, output, n);
    while ()
    gpuErrCh;

}
