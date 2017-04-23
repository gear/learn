#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "print_kernel.cuh"


using std::cerr;
using std::cout;
using std::endl;

const float PI = 3.14159265358979;

int main() {
    cudaCallKernel();
    printf("Hello, I ran kernel, me good?\n");
    return 1;
}
