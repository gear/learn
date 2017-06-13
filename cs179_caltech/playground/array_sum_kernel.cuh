#ifndef ARRAY_SUM_KERNEL_CUH
#define ARRAY_SUM_KERNEL_CUH

enum SumImplementation {  DIVERGENCE, BANKCONFLICT, SEQ, \
                          GLOBAL_ADD, UNROLL_LAST, UNROLL_ALL, \
                          OPTIMAL  };

void cudaReduceSum(
    const int *d_input,
    int *d_output,
    int n,
    SumImplementation type);
)

#endif
