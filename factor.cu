#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>

#include <sys/time.h>

typedef unsigned long long uint64;

static double t0 = 0;
double getTime() {
  timeval tv;
  gettimeofday(&tv, NULL);
  double t = tv.tv_sec + 1e-6 * tv.tv_usec;
  double s = t - t0;
  t0 = t;
  return s;
}

__host__ __device__ uint64 gcd(uint64 u, uint64 v) {
  uint64 shift;
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
}

__global__ void clearPara(uint64 * da, uint64 * dc, uint64 m) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  da[idx] = da[idx] % (m - 1) + 1;
  dc[idx] = dc[idx] % (m - 1) + 1;
}

__global__ void pollardKernel(uint64 num, uint64 * resultd, uint64 * dx, uint64 * dy, uint64 * da, uint64 * dc) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64 n = num;
  uint64 x, y, a, c;
  x = dx[idx];
  y = dy[idx];
  a = da[idx];
  c = dc[idx];

  x = (a * x * x + c) % n;
  y =  a * y * y + c;
  y = (a * y * y + c) % n;

  uint64 z = x > y ? (x - y) : (y - x);
  uint64 d = gcd(z, n);
  
  dx[idx] = x;
  dy[idx] = y;

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

  uint64 *resultd = NULL, *dx = NULL, *dy = NULL, *da = NULL, *dc = NULL;
  cudaMalloc((void**)&resultd, sizeof(uint64));
  cudaMemset(resultd, 0, sizeof(uint64));

  cudaMalloc((void**)&dx, nB * nT * sizeof(uint64));
  cudaMalloc((void**)&dy, nB * nT * sizeof(uint64));
  cudaMalloc((void**)&da, nB * nT * sizeof(uint64));
  cudaMalloc((void**)&dc, nB * nT * sizeof(uint64));
  
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64);
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  curandGenerateLongLong(gen, da, nB * nT);
  curandGenerateLongLong(gen, dc, nB * nT);
  cudaMemset(dx, 0, nB * nT * sizeof(uint64));
  cudaMemset(dy, 0, nB * nT * sizeof(uint64));
  clearPara<<<nB, nT>>>(da, dc, upper);

  while(result == 0) {
    pollardKernel<<<nB, nT>>>(num, resultd, dx, dy, da, dc);
    cudaMemcpy(&result, resultd, sizeof(uint64), cudaMemcpyDeviceToHost);
  }
  
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(da);
  cudaFree(dc);
  cudaFree(resultd);
  curandDestroyGenerator(gen);
  return result;
}

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
  uint64 y, d;

  y = x;
  d = 1;

  do {
    x = (a * x * x + c) % num;
    y =  a * y * y + c;
    y = (a * y * y + c) % num;
    uint64 z = x > y ? (x - y) : (y - x);
    d = gcd(z, num);
  } while (d == 1 && !quit);


  if (d != 1 && d != num ) {
    quit = true;
    result = d;
  }
    
  return result;
}

int main(int argc, char* argv[]) {
  getTime();
  srand(time(NULL));

  uint64 num = atol(argv[1]);

  uint64 result = pollard(num);

  printf("Result: %lld = %lld * %lld\n", num, result, num / result);

  printf("Time  : %.6fs\n", getTime());

  int t2 = clock();
  result = pollardhost(num);
  printf("Result: %lld = %lld * %lld\n", num, result, num / result);
  printf("Time  : %.6fs\n", getTime());
  return 0;
}
