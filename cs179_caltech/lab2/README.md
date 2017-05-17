# CS179 - GPU Programming Lab2

## PART 1

### Question 1.1: Latency Hiding

Approximately how many arithmetic instructions does it take to hide the latency
of a single arithmetic instruction on a GK110?

Assume all of the arithmetic instructions are independent (ie have no
instruction dependencies). You do not need to consider the number of execution
cores on the chip.

_Answer_:

According to the slides, an arithmetic instruction on a GK110 takes ~10ns to
execute. In one GK110's SM, there are 4 schedulers, each can start instruction
in 4 warps and 2 instructions each warps in 1 clock. Therefore, the maximum
number of instructions per clock a SM can start is 8. Assuming the frequency of
GK110 is 1GHz, hence there are 10 clocks in 10ns; we can start 80 arithmetic
instructions in 10 clocks. If we ignore the instruction dependencies, it takes
80 arithmetic instructions to hide the latency of a single arithmetic instruction.


### Question 1.2: Thread Divergence

Let the block shape be (32, 32, 1).

(a)
```cpp
int idx = threadIdx.y + blockDim.y * threadIdx.x;
if (idx % 32 < 16)
    foo();
else
    bar();
```
Does this code diverge? Why or why not?

_Answer_:

This code does not diverge. Since the block shape is (32,32,1), we have:
- `blockDim.y == 32`
- `threadIdx.y` is the warp id (32 threads along threadIdx.x is a warp)
Since `threadIdx.x = [0..31]` in a warp and `blockDim.y == 32`, we always
have `idx % 32 == threadIdx.y` in a warp. Therefore, a warp will execute
`foo()` xor `bar()`. Hence, there is no divergence.

(b)
```cpp
const float pi = 3.14;
float result = 1.0;
for (int i = 0; i < threadIdx.x; i++)
    result *= pi; //*
```
Does this code diverge? Why or why not? (This is a bit of a trick question,
either "yes" or "no can be a correct answer with appropriate explanation.)

_Answer_:

I think this code diverges. Although all threads in a warp executes the same
instruction, but the number of executions are different among each threads.
For this reason, some threads will wait for other threads when it reached
the end of their execution.

### Question 1.3: Coalesced Memory Access

Let the block shape be (32, 32, 1). Let data be a `(float *)` pointing to global
memory and let data be 128 byte aligned (so data % 128 == 0).

Consider each of the following access patterns.

(a)
```cpp
data[threadIdx.x + blockDim.x * threadIdx.y] = 1.0;
```
Is this write coalesced? How many 128 byte cache lines does this write to?

_Answer_:

Memory access in this code is coalesced. Each warp of 32 threads handles exactly
32 consecutive floats in `data`. Each float is 32-bit or 4-bytes, hence each
warp writes to exactly one 128-bytes cache line. Since the block has 32 warps,
this code writes to 32 128-bytes cache lines.

(b)
```cpp
data[threadIdx.y + blockDim.y * threadIdx.x] = 1.0;
```

Is this write coalesced? How many 128 byte cache lines does this write to?

_Answer_:

This code is non-coalesced due to each thread in a warp writes to different
128-bytes cache lines. Each warp writes to 32 128-bytes cache lines.


(c)
```cpp
data[1 + threadIdx.x + blockDim.x * threadIdx.y] = 1.0;
```

Is this write coalesced? How many 128 byte cache lines does this write to?

_Answer_:

This code is non-coalesced and possibly error prone. Since the data is aligned,
each thread in a warp will read 31 values from one cache line and read one more
float (4 bytes) from the next cache line, causing total of 2 128-bytes cache
lines writes per warp. Since the minimum number of cache line access is 1, this
code is non-coalesced.

### Question 1.4: Bank Conflicts and Instruction Dependencies

Let's consider multiplying a 32 x 128 matrix with a 128 x 32 element matrix.
This outputs a 32 x 32 matrix. We'll use 32 ** 2 = 1024 threads and each thread
will compute 1 output element. Although its not optimal, for the sake of
simplicity let's use a single block, so grid shape = (1, 1, 1),
block shape = (32, 32, 1).

For the sake of this problem, let's assume both the left and right matrices have
already been stored in shared memory are in column major format. This means the
element in the ith row and jth column is accessible at lhs[i + 32 * j] for the
left hand side and rhs[i + 128 * j] for the right hand side.

This kernel will write to a variable called output stored in shared memory.

Consider the following kernel code:

```cpp
int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    // Loop unrolling technique
    output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}
```

(a)
Are there bank conflicts in this code? If so, how many ways is the bank conflict
(2-way, 4-way, etc)?

_Answer_:

Since the number of elements in `output`, `lsh`, and `rhs` is a multiple of 32,
it is safe to assume that the first element of these matrices belongs to bank 0.  
Consider the first warp: `threadIdx.x = [0..31]`, `threadIdx.y = 0`. For each
for loop, the memory bank access of threads in the first warp is given as:

| Mem \ Thread | 0 | 1 | 2 | ... | i |
| :------------- |:-:|:-:|:-:|:---:|:-:|
| `lhs[i + k * 32]` | 0 | 1 | 2 | ... | i |
| `rhs[k + j * 128]` | k | k | k | ... | k |
| `output[i + 32 * j]` | 0 | 1 | 2 | ... | i |

We can see here each thread access different memory bank for `lhs` and `output`.
All threads accesses the same memory address of bank `k` for `rhs`. For
example, if `k = 5`, all thread accesses `rhs[5]` (belongs to bank 5). Here,
bank 5 has also been accessed by thread 5 to read `lhs` and `output`. However,
since the same element is accessed by all thread from bank 5, bank conflict does
not occur. The behavior of the second line is exactly the same as the first line
of code. Hence, no bank conflict.


(b)
Expand the inner part of the loop (below)

```cpp
output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
```

into "psuedo-assembly" as was done in the coordinate addition example in lecture
4.

There's no need to expand the indexing math, only to expand the loads, stores,
and math. Notably, the operation a += b * c can be computed by a single
instruction called a fused multiply add (FMA), so this can be a single
instruction in your "psuedo-assembly".

Hint: Each line should expand to 5 instructions.

_Answer_:

```cpp
 1| lhs_i_k = lsh[i + 32 * k];              // LOAD
 2| rhs_k_j = rhs[k + 128 * j];             // LOAD
 3| output_i_j = output[i + 32 * j];        // LOAD
 4| r = lhs_i_k * rhs_k_j + output_j_j;     // FMA
 5| output[i + 32 * j] = r;                 // STORE
 6|
 7| lhs_i_k_1 = lhs[i + 32 * (k+1)];        // LOAD
 8| rhs_k_1_j = rhs[(k+1) + 128 * j];       // LOAD
 9| output_i_j = output[i + 32 * j];        // LOAD
10| r = lsh_i_k_1 * rhs_k_1_j + output_i_j; // FMA
11| output[i + 32 * j] = r;                 // STORE
```

(c)
Identify pairs of dependent instructions in your answer to part b.

_Answer_:

Pairs of dependent instructions: (4,3), (4,2), (4,1), (5,4), (9,5), (10,9),
(10,8), (10,7), (11,10).

(d)
Rewrite the code given at the beginning of this problem to minimize instruction
dependencies. You can add or delete instructions (deleting an instruction is a
valid way to get rid of a dependency!) but each iteration of the loop must still
process 2 values of k.

_Answer_:

```cpp
int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    // Loop unrolling technique
    x = lhs[i + 32 * k] * rhs[k + 128 * j];
    y = lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
    output[i + 32 * j] += x + y;
}
```

(e)
Can you think of any other anything else you can do that might make this code
run faster?

_Answer_:

We can explicitly write each load instruction so they are stored in the
registers, but this task might already be done by the compiler. On the other
hand, we can unroll the loop by 4 or 8 to avoid instruction dependencies.

### PART 2 - Matrix transpose optimization

Optimize the CUDA matrix transpose implementations in transpose_cuda.cu. Read
ALL of the TODO comments. Matrix transpose is a common exercise in GPU
optimization, so do not search for existing GPU matrix transpose code on the
Internet.

Your transpose code only need to be able to transpose square matrices where the
side length is a multiple of 64.

The initial implementation has each block of 1024 threads handle a 64x64 block
of the matrix, but you can change anything about the kernel if it helps obtain
better performance.

The main method of transpose.cc already checks for correctness for all transpose
results, so there should be an assertion failure if your kernel produces incorrect
output.

The purpose of the `shmemTransposeKernel` is to demonstrate proper usage of global
and shared memory. The `optimalTransposeKernel` should be built on top of
`shmemTransposeKernel` and should incorporate any "tricks" such as ILP, loop
unrolling, vectorized IO, etc that have been discussed in class.

You can compile and run the code by running

make transpose
./transpose

and the build process was tested on minuteman. If this does not work on haru for
you, be sure to add the lines

export PATH=/usr/local/cuda-6.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH

to your ~/.profile file (and then exit and ssh back in to restart your shell).

On OS X, you may have to run or add to your .bash_profile the command

export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/cuda/lib/

in order to get dynamic library linkage to work correctly.

The transpose program takes 2 optional arguments: input size and method. Input
size must be one of -1, 512, 1024, 2048, 4096, and method must be one all,
cpu, gpu_memcpy, naive, shmem, optimal. Input size is the first argument and
defaults to -1. Method is the second argument and defaults to all. You can pass
input size without passing method, but you cannot pass method without passing an
input size.

Examples:
./transpose
./transpose 512
./transpose 4096 naive
./transpose -1 optimal

Copy paste the output of ./transpose.cc into README.txt once you are done.
Describe the strategies used for performance in either block comments over the
kernel (as done for naiveTransposeKernel) or in README.txt.

_Answer_:

Output from the console: (GTX 580)
```
Index of the GPU with the lowest temperature: 0 (44 C)
Time limit for this program set to 10 seconds
Size 512 naive CPU: 0.443040 ms
Size 512 GPU memcpy: 0.032032 ms
Size 512 naive GPU: 0.062720 ms
Size 512 shmem GPU: 0.026016 ms
Size 512 optimal GPU: 0.017600 ms

Size 1024 naive CPU: 2.640128 ms
Size 1024 GPU memcpy: 0.052864 ms
Size 1024 naive GPU: 0.253152 ms
Size 1024 shmem GPU: 0.088224 ms
Size 1024 optimal GPU: 0.072128 ms

Size 2048 naive CPU: 33.025375 ms
Size 2048 GPU memcpy: 0.217152 ms
Size 2048 naive GPU: 1.079072 ms
Size 2048 shmem GPU: 0.306272 ms
Size 2048 optimal GPU: 0.245728 ms

Size 4096 naive CPU: 155.485855 ms
Size 4096 GPU memcpy: 0.797472 ms
Size 4096 naive GPU: 4.658048 ms
Size 4096 shmem GPU: 1.159136 ms
Size 4096 optimal GPU: 0.980992 ms
```

### BONUS

Mathematical scripting environments such as Matlab or Python + Numpy often
encourage expressing algorithms in terms of vector operations because they offer
a convenient and performant interface. For instance, one can add 2 n-component
vectors (a and b) in Numpy with c = a + b.

This is often implemented with something like the following code:

```cpp
void vec_add(float *left, float *right, float *out, int size) {
    for (int i = 0; i < size; i++)
        out[i] = left[i] + right[i];
}
```

Consider the code

a = x + y + z

where x, y, z are n-component vectors.

One way this could be computed would be

vec_add(x, y, a, n);
vec_add(a, z, a, n);

In what ways is this code (2 calls to vec_add) worse than the following?

for (int i = 0; i < n; i++)
    a[i] = x[i] + y[i] + z[i];

List at least 2 ways (you don't need more than a sentence or two for each way).
