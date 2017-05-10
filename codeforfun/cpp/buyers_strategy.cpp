#include <iostream>
#include <stdio.h>

int *temp = new int[10000];

int sum(int *arr, int len) {
  int s = 0;
  for (int i = 0; i < len; i++) s+= arr[i];
  return s;
}

int min(int a, int b) {
  return a < b ? a : b;
}

int min3(int *arr) {
  int a = arr[0], b = arr[1], c = arr[2];
  return min(a, min(b,c));
}

int minCost(int *arr, int len) {
  if (len < 3) return sum(arr, len);
  if (len == 3) return sum(arr, len) - min3(arr);
  if (temp[len] != -1) return temp[len];
  else {
    int b1 = sum(arr, 1) + minCost(arr+1, len-1);
    int b2 = sum(arr, 2) + minCost(arr+2, len-2);
    int b3 = sum(arr, 3) + minCost(arr+3, len-3) - min3(arr);
    temp[len] = min(b1, min(b2,b3));
  }
  return temp[len];
}

int main() {
  int test[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
  for (int i = 0; i < 10000; ++i) temp[i] = -1;
  printf("%d\n", sum(test, 14));
  printf("%d\n", minCost(test, 14));
  return 1;
}
