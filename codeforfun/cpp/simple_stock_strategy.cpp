#include <stdio.h>
#include <stdlib.h>

int * findLocalMax(int *arr, int len, int *local_max) {
  int i = 1, j = 0;
  if (arr[0] > arr[i])
    local_max[j++] = arr[0];
  for (; i < len - 1; ++i) {
    if (arr[i-1] <= arr[i] && arr[i] > arr[i+1])
      local_max[j++] = arr[i];
  }
  if (arr[i] >= arr[i-1])
    local_max[j++] = arr[i];
  return j;
}

int * findLocalMin(int *arr, int len, int *local_min) {
  int i = 1, j = 0;
  if (arr[0] < arr[i])
    local_min[j++] = arr[0];
  for (; i < len - 1; ++i) {
    if (arr[i-1] > arr[i] && arr[i] <= arr[i+1])
      local_min;
  }
}

int maxProfit(int *arr, int len) {
  int *local_min = (int*) malloc(sizeof(int) * len / 2);
  int *local_max = (int*) malloc(sizeof(int) * len / 2);
  findLocalMin(int *arr, int len, local_min);
  findLocalMax(int *arr, int len, local_max);
}
