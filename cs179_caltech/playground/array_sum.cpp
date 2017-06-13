#include <cassert>
#include <cstdio>
#include <cstdlib> /* rand() srand() and RAND_MAX */
#include <cstring>
#include <string>
#include <math.h> /* fabs(), pow() */

#include <cuda_runtime.h>

#include "array_sum_kernel.cuh"

/*
 * NOTE: You can use this macro to easily check cuda error codes
 * and get more information.
 *
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *         what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
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

#define BLOCKSIZE 128;

/* Naive CPU array sum. */
void cpuSum(const int *input, int *output, int n) {
    long result = 0;
    for (int i = 0; i < n; i++) {
        result += input[i];
    }
    *output = result;
}

/*
 * Fills fill with random numbers [0, max). Size is number of elements to
 * assign.
 */
void randomFill(int *fill, int size, unsigned max) {
    for (int i = 0; i < size; i++) {
        int r = rand() % max;
        fill[i] = r;
    }
}

int main(int argc, char *argv[]) {

    // Seed random number generator
    srand(2017);

    std::string kernel = "all";
    int size_to_run = -1;

    // Check arguments
    assert(argc <= 3);
    if (argc >= 2)
        size_to_run = atoi(argv[1]);
    if (argc == 3)
        kernel = argv[2];

    // 2^20, 2^21, 2^22, 2^23
    if (!(size_to_run == -1     ||
         size_to_run == 1048576 ||
         size_to_run == 2097152 ||
         size_to_run == 4194304 ||
         size_to_run == 8388608))
    {
        fprintf(stderr,
            "Program only designed to run sizes 1048576, 2097152, 4194304, 8388608\n");
    }

    assert(kernel == "all"              ||
        kernel == "cpu"                 ||
        kernel == "gpu_divergence"      ||
        kernel == "gpu_bank_conflict"   ||
        kernel == "gpu_sequential_addr" ||
        kernel == "gpu_global_add"      ||
        kernel == "gpu_unroll_last"     ||
        kernel == "gpu_unroll_all"      ||
        kernel == "optimal");

    // Run the transpose implementations for all desired sizes (2^9 = 512,
    // 2^12 = 4096)
    for (int _i = 20; _i < 24; _i++) {
        int n = 1 << _i;
        if (size_to_run != -1 && size_to_run != n)
            continue;

        assert(n % 64 == 0);

        cudaEvent_t start;
        cudaEvent_t stop;

#define START_TIMER() {                                                        \
            gpuErrChk(cudaEventCreate(&start));                                \
            gpuErrChk(cudaEventCreate(&stop));                                 \
            gpuErrChk(cudaEventRecord(start));                                 \
        }

#define STOP_RECORD_TIMER(name) {                                              \
            gpuErrChk(cudaEventRecord(stop));                                  \
            gpuErrChk(cudaEventSynchronize(stop));                             \
            gpuErrChk(cudaEventElapsedTime(&name, start, stop));               \
            gpuErrChk(cudaEventDestroy(start));                                \
            gpuErrChk(cudaEventDestroy(stop));                                 \
        }

        // Initialize timers
        float cpu_ms = -1;
        float gpu_divergence_ms = -1;
        float gpu_bank_conflict_ms = -1;
        float gpu_sequential_addr_ms = -1;
        float gpu_global_add_ms = -1;
        float gpu_unroll_last = -1;
        float gpu_unroll_all = -1;
        float gpu_optimal = -1;

        // Allocate host memory
        int *input = new int[n];
        int *output = new int[n];
        int ref_output = 0.0; /* Correct sum */

        // Allocate device memory
        int *d_input;
        int *d_output;
        gpuErrChk(cudaMalloc(&d_input, n * sizeof(int)));
        gpuErrChk(cudaMalloc(&d_output, n * sizeof(int)));

        // Initialize input data to random integers in [0, 10000)
        randomFill(input, n, 10000);

        // Copy input to GPU
        gpuErrChk(cudaMemcpy(d_input, input, n * sizeof(int),
            cudaMemcpyHostToDevice));

        // CPU implementation
        if (kernel == "cpu" || kernel == "all") {
            START_TIMER();
            cpuSum(input, output, n);
            STOP_RECORD_TIMER(cpu_ms);

            ref_output = *output;
            memset(output, 0, n * n * sizeof(int));

            printf("Size %d naive CPU: %f ms\n", n, cpu_ms);
        }

        // Naive GPU implementation of reduce sum (warp divergence)
        if (kernel == "gpu_divergence" || kernel == "all") {
            START_TIMER();
            cudaReduceSum(d_input, d_output, n, DIVERGENCE);
            STOP_RECORD_TIMER(gpu_divergence_ms);

            gpuErrChk(cudaMemcpy(output, d_output, n * sizeof(int),
                cudaMemcpyDeviceToHost));
            checkTransposed(input, output, n);

            memset(output, 0, n * sizeof(int));
            gpuErrChk(cudaMemset(d_output, 0, n * sizeof(int)));

            printf("Size %d naive GPU: %f ms\n", n, naive_gpu_ms);
        }

        // shmem GPU implementation
        if (kernel == "shmem" || kernel == "all") {
            START_TIMER();
            cudaTranspose(d_input, d_output, n, SHMEM);
            STOP_RECORD_TIMER(shmem_gpu_ms);

            gpuErrChk(cudaMemcpy(output, d_output, n * n * sizeof(float),
                cudaMemcpyDeviceToHost));
            checkTransposed(input, output, n);

            memset(output, 0, n * n * sizeof(float));
            gpuErrChk(cudaMemset(d_output, 0, n * n * sizeof(float)));

            printf("Size %d shmem GPU: %f ms\n", n, shmem_gpu_ms);
        }

        // Optimal GPU implementation
        if (kernel == "optimal"    || kernel == "all") {
            START_TIMER();
            cudaTranspose(d_input, d_output, n, OPTIMAL);
            STOP_RECORD_TIMER(optimal_gpu_ms);

            gpuErrChk(cudaMemcpy(output, d_output, n * n * sizeof(float),
                cudaMemcpyDeviceToHost));
            checkTransposed(input, output, n);

            memset(output, 0, n * n * sizeof(float));
            gpuErrChk(cudaMemset(d_output, 0, n * n * sizeof(float)));

            printf("Size %d optimal GPU: %f ms\n", n, optimal_gpu_ms);
        }

        // Free host memory
        delete[] input;
        delete[] output;

        // Free device memory
        gpuErrChk(cudaFree(d_input));
        gpuErrChk(cudaFree(d_output));

        printf("\n");
    }
}
