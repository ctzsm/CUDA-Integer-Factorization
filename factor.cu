#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>

typedef unsigned long long uint64;

<<<<<<< HEAD
__device__ uint64 d_gcd(uint64 u, uint64 v) {
=======
uint64 gcd(uint64 u, uint64 v) {
>>>>>>> 4f9415858bf958bb7dc338d7bad05c31cb6ac7b3
  int shift;
  if (u == 0) return v;
  if (v == 0) return u;
  for (shift = 0; ((u | v) & 1) == 0; ++shift) {
    u >>= 1;
    v >>= 1;
  }
    
  while ((u & 1) == 0)
    u >>= 1;
    
  do {
    while ((v & 1) == 0)
      v >>= 1;
    
    if (u > v) {
<<<<<<< HEAD
      int64_t t = v; v = u; u = t;}
=======
      uint64 t = v; v = u; u = t;}
>>>>>>> 4f9415858bf958bb7dc338d7bad05c31cb6ac7b3
    v = v - u; 
  } while (v != 0);
  
  return u << shift;
<<<<<<< HEAD
  /* uint64 t;
  while(v) {
    t = u; u = v; v = t % v;
  }
  return u; */
=======
}

__device__ uint64 d_gcd(uint64 u, uint64 v) {
  int shift;
  if (u == 0) return v;
  if (v == 0) return u;
  for (shift = 0; ((u | v) & 1) == 0; ++shift) {
    u >>= 1;
    v >>= 1;
  }
    
  while ((u & 1) == 0)
    u >>= 1;
    
  do {
    while ((v & 1) == 0)
      v >>= 1;
    
    if (u > v) {
      uint64 t = v; v = u; u = t;}
    v = v - u; 
  } while (v != 0);
  
  return u << shift;
>>>>>>> 4f9415858bf958bb7dc338d7bad05c31cb6ac7b3
}

__global__ void clearPara(uint64 * para, uint64 m) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  para[idx * 4] = 0;
  para[idx * 4 + 1] = 0;
  para[idx * 4 + 2] = para[idx * 4 + 2] % (m - 1) + 1;
  para[idx * 4 + 3] = para[idx * 4 + 3] % (m - 1) + 1;
}

__global__ void pollardKernel(uint64 num, uint64 * resultd, uint64* para) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64 n = num;
  uint64 x, y, a, c;
  x = para[idx * 4];
  y = para[idx * 4 + 1];
  a = para[idx * 4 + 2];
  c = para[idx * 4 + 3];

  x = (a * x * x + c) % n;
  y =  a * y * y + c;
  y = (a * y * y + c) % n;

  uint64 z = x > y ? (x - y) : (y - x);
  uint64 d = d_gcd(z, n);
  
  para[idx * 4] = x;
  para[idx * 4 + 1] = y;

  if (d != 1 && d != n) *resultd = d;
}

uint64 pollard(uint64 num) {
  uint64 upper = sqrt(num), result = 0;

  int nT = 256, nB = 256;

  if (num % 2 == 0) return 2;
  if (num % 3 == 0) return 3;
  if (num % 5 == 0) return 5;
  if (num % 7 == 0) return 7;
  
  if (upper * upper == num) return upper;

  uint64 *resultd = NULL, *para = NULL;
  cudaMalloc((void**)&resultd, sizeof(uint64));
  cudaMemset(resultd, 0, sizeof(uint64));

  cudaMalloc((void**)&para, 4 * nB * nT * sizeof(uint64));
  
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  curandGenerateLongLong(gen, para, nB * nT);
  clearPara<<<nB, nT>>>(para, upper);

  while(result == 0) {
    pollardKernel<<<nB, nT>>>(num, resultd, para);
    cudaMemcpy(&result, resultd, sizeof(uint64), cudaMemcpyDeviceToHost);
  }
  
  cudaFree(para);
  cudaFree(resultd);
  curandDestroyGenerator(gen);
  return result;
}
<<<<<<< HEAD
int main(int argc, char* argv[]) {
  int t = clock();
  
=======

uint64 pollardhost(uint64 num){
  uint64 upper = sqrt(num), result = 0;

  if (num % 2 == 0) return 2;
  if (num % 3 == 0) return 3;
  if (num % 5 == 0) return 5;
  if (num % 7 == 0) return 7;  

  if (upper * upper == num) return upper;


  bool quit = false;


  uint64 x = 0;
  uint64 a = rand() % (upper-1) + 1;
  uint64 c = rand() % (upper-1) + 1;
  uint64 y, d, z;

  y = x;
  d = 1;

  do
    {
    x = (a * x * x + c) % num;
    y =  a * y * y + c;
    y = (a * y * y + c) % num;
    uint64 z = x > y ? (x - y) : (y - x);
    d = gcd(z,num);
    } while (d == 1 && !quit );


    if (d != 1 && d != num )
    {
    quit = true;
    result = d;
    }

    
    return result;
}





int main(int argc, char* argv[]) {
  int t1 = clock();
  srand(time(NULL));
>>>>>>> 4f9415858bf958bb7dc338d7bad05c31cb6ac7b3
  uint64 num = atol(argv[1]);

  uint64 result = pollard(num);

  printf("Result: %lld = %lld * %lld\n", num, result, num / result);
<<<<<<< HEAD
  printf("Time  : %fs\n", 1.0 * (clock() - t) / CLOCKS_PER_SEC);
  return 0;
}

=======
  printf("Time  : %fs\n", 1.0 * (clock() - t1) / CLOCKS_PER_SEC);
  int t2 = clock();
  result = pollardhost(num);
  printf("Result: %lld = %lld * %lld\n", num, result, num / result);
  printf("Time  : %fs\n", 1.0 * (clock() - t1) / CLOCKS_PER_SEC);
  return 0;
}










>>>>>>> 4f9415858bf958bb7dc338d7bad05c31cb6ac7b3
