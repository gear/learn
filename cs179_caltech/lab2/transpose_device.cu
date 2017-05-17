#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304 matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
    /* Since the naive kernel doesn't use shared memory, we do not have
     * bank conflict. The remaining problem is data alignment.
     * Each warp handles 32*4 = 128 4-bytes elements, hence there is minimum
     * of 4 cache reads. However, in here, due to the fact that n >= 512,
     * each thread in a warp reads from 5 cache lines. A warp reads 160 cache
     * lines.
     */
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {

    __shared__ float data[64*64];

    int s_i = threadIdx.x;
    int s_j = threadIdx.y * 4;

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_k = 4;

    /* I would like to move all non-linear transformation
     * into the shared memory, therefore we need to specify
     * the indices for the transposed matrix instead of just
     * swapping i and j.
     */
    const int i_t = threadIdx.x + 64 * blockIdx.y;
    int j_t = 4 * threadIdx.y + 64 * blockIdx.x;


    for (int k = 0; k < end_k; k++)
        data[s_j + k + s_i*64] = input[i + n * (j + k)];
    __syncthreads();

    for (int k = 0; k < end_k; k++)
        output[i_t + n * (j_t + k)] = data[s_i + (s_j+k)*64];
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
  /* Zero-padding for shared memory to avoid bank conflicts */
  __shared__ float data[64*65];

  int s_i = threadIdx.x;
  int s_j = threadIdx.y * 4;

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;

  /* I would like to move all non-linear transformation
   * into the shared memory, therefore we need to specify
   * the indices for the transposed matrix instead of just
   * swapping i and j.
   */
  const int i_t = threadIdx.x + 64 * blockIdx.y;
  int j_t = 4 * threadIdx.y + 64 * blockIdx.x;

  /* Unroll the loop too */
  data[s_j + 0 + s_i*65] = input[i + n * (j + 0)];
  data[s_j + 1 + s_i*65] = input[i + n * (j + 1)];
  data[s_j + 2 + s_i*65] = input[i + n * (j + 2)];
  data[s_j + 3 + s_i*65] = input[i + n * (j + 3)];
  __syncthreads();

  output[i_t + n * (j_t + 0)] = data[s_i + (s_j+0)*65];
  output[i_t + n * (j_t + 1)] = data[s_i + (s_j+1)*65];
  output[i_t + n * (j_t + 2)] = data[s_i + (s_j+2)*65];
  output[i_t + n * (j_t + 3)] = data[s_i + (s_j+3)*65];
}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
