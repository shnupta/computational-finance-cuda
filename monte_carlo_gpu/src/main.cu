#include "option/european.h"

#include <curand_kernel.h>
#include <helper_cuda.h>

#include <iostream>
#include <stdio.h>

#define THREADBLOCK_SIZE 512
/* #define THREADBLOCK_SIZE 2 */

#define r 0.03

__global__ void InitRandomStates(curandState_t* devStates) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence
            number, no offset */
  curand_init(1234, id, 0, &devStates[id]);
}

__global__ void InitRandomStatesQuasi(curandStateScrambledSobol32_t* devStates,
    curandDirectionVectors32_t* directionVectors, unsigned int* scrambleConstants) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  // Use the same timeSteps dimensions, but offset each thread by ?
  for (int i = 0; i < 300; i++) {
    unsigned int offset = id * ((10000 / blockDim.x) + 1);
    /* printf("blockDim.x = %d    blockIdx.x = %d\n", blockDim.x, blockIdx.x); */
    /* printf("rand init    id = %d    dim = %d    offset = %u    state idx = %d\n", id, i, offset, 300 * id + i); */
    curand_init(directionVectors[i], scrambleConstants[i], offset, &devStates[300 * id + i]);
  }
}

__device__ void GenerateSamplePath(VanillaEuropean option, double* path,
    const int timeSteps, curandState_t* state) {
  double St = option.GetS0();
  double sigma = option.GetSigma();
  double T = option.GetTTM();
  int m = timeSteps;
  for (int k = 0; k < m; k++) {
    const double z = curand_normal_double(state);
    path[k] = St * exp((r - sigma*sigma * 0.5) * (T/m) + sigma * sqrt(T/m) 
        * z);
    St = path[k];
    /* if (threadIdx.x == 0 && blockIdx.x == 0) */
    /*   printf("z = %f  St = %f\n", z, St); */
  }
}

__device__ void GenerateSamplePathQuasi(VanillaEuropean option, double* path,
    const int timeSteps, curandStateScrambledSobol32_t* states) {
  double St = option.GetS0();
  double sigma = option.GetSigma();
  double T = option.GetTTM();
  int m = timeSteps;
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  for (int k = 0; k < m; k++) {
    const double z = curand_normal_double(&states[k]);
    if (&path[k] == 0x0)
      printf("ERROR: Accessing random state index = %d\n&path = %p\n===\n", 300 * id + k, path);
    path[k] = St * exp((r - sigma*sigma * 0.5) * (T/m) + sigma * sqrt(T/m) 
        * z);
    St = path[k];
    /* if (threadIdx.x == 0 && blockIdx.x == 0) */
    /*   printf("generator addr = %p  z = %f  St = %f\n", &states[k], z, St); */
  }
}

__global__ void PriceByMC(VanillaEuropean* options, double* optionValues, 
    double* optionDeltas, const int optionsNum, const long simNum, 
    const int timeSteps, curandState_t* devStates, double* dev_paths) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int id = tid + bid * blockDim.x;

  int simNo = tid;
  int i = 0;

  __shared__ double payoffs[THREADBLOCK_SIZE];
  __shared__ double deltas[THREADBLOCK_SIZE];
  double threadPayoff = 0.0;
  double threadDelta = 0.0;

  curandState_t* state = &devStates[id];

  if (bid >= optionsNum) return;

  VanillaEuropean option = options[bid];
  double* path = &dev_paths[300 * id];

  while (simNo < simNum) {
    // Generate sample path of this option
    GenerateSamplePath(option, path, timeSteps, state);
    threadPayoff = (i * threadPayoff + option.Payoff(path, timeSteps)) 
      / (i + 1.0);
    /* if (threadIdx.x == 96) */
    /*   printf("last path = %f  avg payoff so far = %f\n", path[timeSteps-1], threadPayoff); */
    double dY_dS0 = (path[timeSteps - 1] / option.GetS0())
      * (path[timeSteps - 1] > option.GetStrike() ? 1.0 : 0.0);
    threadDelta = (i * threadDelta + dY_dS0) / (i + 1.0);
    simNo += THREADBLOCK_SIZE;
    i++;
  }
  payoffs[tid] = threadPayoff;
  deltas[tid] = threadDelta;
  __syncthreads();
  /* printf("FINAL THREAD PAYOFF: %f\n", threadPayoff); */

  if (tid == 0) {
    double avg = 0.0;
    double deltaAvg = 0.0; 
    for (int j = 0; j < THREADBLOCK_SIZE; ++j) {
      avg = (j * avg + payoffs[j]) / (j + 1.0);
      deltaAvg = (j * deltaAvg + deltas[j]) / (j + 1.0);
    }
    optionValues[bid] = exp(-r * option.GetTTM()) * avg;
    optionDeltas[bid] = exp(-r * option.GetTTM()) * deltaAvg;
  }
}

__global__ void PriceByMCQuasi(VanillaEuropean* options, double* optionValues, 
    double* optionDeltas, const int optionsNum, const long simNum, 
    const int timeSteps, curandStateScrambledSobol32_t* devStates, double* dev_paths) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int id = tid + bid * blockDim.x;

  int simNo = tid;
  int i = 0;

  __shared__ double payoffs[THREADBLOCK_SIZE];
  __shared__ double deltas[THREADBLOCK_SIZE];
  double threadPayoff = 0.0;
  double threadDelta = 0.0;

  curandStateScrambledSobol32_t* states = &devStates[300 * id];

  if (bid >= optionsNum) return;

  VanillaEuropean option = options[bid];
  double* path = &dev_paths[300 * id];

  while (simNo < simNum) {
    // Generate sample path of this option
    GenerateSamplePathQuasi(option, path, timeSteps, states);
    threadPayoff = (i * threadPayoff + option.Payoff(path, timeSteps)) 
      / (i + 1.0);
    /* if (id < 5) */
    /*   printf("id = %d  last path = %f  avg payoff so far = %f\n", id, path[timeSteps-1], threadPayoff); */
    double dY_dS0 = (path[timeSteps - 1] / option.GetS0())
      * (path[timeSteps - 1] > option.GetStrike() ? 1.0 : 0.0);
    threadDelta = (i * threadDelta + dY_dS0) / (i + 1.0);
    simNo += THREADBLOCK_SIZE;
    i++;
  }
  payoffs[tid] = threadPayoff;
  deltas[tid] = threadDelta;
  __syncthreads();
  /* printf("FINAL THREAD PAYOFF: %f\n", threadPayoff); */

  if (tid == 0) {
    double avg = 0.0;
    double deltaAvg = 0.0; 
    for (int j = 0; j < THREADBLOCK_SIZE; ++j) {
      avg = (j * avg + payoffs[j]) / (j + 1.0);
      deltaAvg = (j * deltaAvg + deltas[j]) / (j + 1.0);
    }
    optionValues[bid] = exp(-r * option.GetTTM()) * avg;
    optionDeltas[bid] = exp(-r * option.GetTTM()) * deltaAvg;
  }
}

int main() {
  const bool isCall = true;
  const double strike = 100.0;
  const double s0 = 100.0;
  const double sigma = 0.2;
  const double ttm = 1.0 / 12.0; // 1 month

  const int optionsNum = 10;
  const long simNum = 30000;
  const int timeSteps = 300;

  VanillaEuropean options[optionsNum];
  for (int i = 0; i < optionsNum; ++i) {
    options[i] = VanillaEuropean(isCall, strike, s0, sigma, ttm);
  }

  VanillaEuropean* dev_options;
  cudaMalloc((void**) &dev_options, sizeof(VanillaEuropean) * optionsNum);

  cudaMemcpy(dev_options, options, optionsNum * sizeof(VanillaEuropean),
      cudaMemcpyHostToDevice);

  double optionValues[optionsNum];
  double optionDeltas[optionsNum];

  double* dev_optionValues;
  double* dev_optionDeltas;
  cudaMalloc((void**) &dev_optionValues, sizeof(double) * optionsNum);
  cudaMalloc((void**) &dev_optionDeltas, sizeof(double) * optionsNum);

  const int totalThreads = THREADBLOCK_SIZE * optionsNum;

  double* dev_paths;
  checkCudaErrors(cudaMalloc((void**) &dev_paths, totalThreads * timeSteps * sizeof(double)));

  curandState_t* devStates;
  cudaMalloc((void**) &devStates, totalThreads * sizeof(curandState_t));

  // Each thread has timeSteps states
  curandStateScrambledSobol32_t* devQuasiStates;
  checkCudaErrors(cudaMalloc((void**) &devQuasiStates,
      totalThreads * timeSteps * sizeof(curandStateScrambledSobol32_t)));

  curandDirectionVectors32_t* directionVectors;
  curandGetDirectionVectors32(&directionVectors,
      CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);

  // We need timeSteps dimensions (dimensions are shared but with each thread having an offset)
  curandDirectionVectors32_t* dev_directionVectors;
  checkCudaErrors(cudaMalloc((void**) &dev_directionVectors,
      timeSteps * sizeof(curandDirectionVectors32_t)));
  /* printf("%d\n", totalThreads * timeSteps); */
  /* printf("%lu\n", totalThreads * timeSteps * sizeof(curandDirectionVectors32_t)); */

  checkCudaErrors(cudaMemcpy(dev_directionVectors, directionVectors,
      timeSteps * sizeof(curandDirectionVectors32_t), cudaMemcpyHostToDevice));

  unsigned int* scrambleConstants;
  curandGetScrambleConstants32(&scrambleConstants);
  unsigned int* dev_scrambleConstants;
  checkCudaErrors(cudaMalloc((void**) &dev_scrambleConstants,
        timeSteps * sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(dev_scrambleConstants, scrambleConstants,
        timeSteps * sizeof(unsigned int), cudaMemcpyHostToDevice));

  InitRandomStates<<<optionsNum, THREADBLOCK_SIZE>>>(devStates);
  InitRandomStatesQuasi<<<optionsNum, THREADBLOCK_SIZE>>>(devQuasiStates,
      dev_directionVectors, dev_scrambleConstants);
  getLastCudaError("Initialisation failed\n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  PriceByMCQuasi<<<optionsNum, THREADBLOCK_SIZE>>>(dev_options, dev_optionValues, 
      dev_optionDeltas, optionsNum,
      simNum, timeSteps, devQuasiStates, dev_paths);
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  getLastCudaError("Quasi kernel failed\n");

  /* cudaEventRecord(start); */
  /* PriceByMC<<<optionsNum, THREADBLOCK_SIZE>>>(dev_options, dev_optionValues, */ 
  /*     dev_optionDeltas, optionsNum, */
  /*     simNum, timeSteps, devStates, dev_paths); */
  /* cudaEventRecord(stop); */
  /* cudaDeviceSynchronize(); */
  /* getLastCudaError("Normal kernel failed\n"); */

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Took: %fms\n", milliseconds);

  cudaMemcpy(optionValues, dev_optionValues, sizeof(double) * optionsNum,
      cudaMemcpyDeviceToHost);
  cudaMemcpy(optionDeltas, dev_optionDeltas, sizeof(double) * optionsNum,
      cudaMemcpyDeviceToHost);

  std::cout << "S(0) = " << options[0].GetS0() << std::endl;
  std::cout << "K = " << options[0].GetStrike() << std::endl;
  std::cout << "TTM = " << options[0].GetTTM() << std::endl;
  std::cout << "sigma = " << options[0].GetSigma() << std::endl;
  std::cout << "isCall? " << options[0].IsCall() << std::endl;
  std::cout << "r = " << r << std::endl;

  std::cout << std::endl << "=== Calculated ===" << std::endl;
  for (int i = 0; i < optionsNum; ++i) {
    std::cout << "Option " << i << "\n---------" << std::endl;
    std::cout << "Option value = " << optionValues[i] << std::endl;
    std::cout << "Delta = " << optionDeltas[i] << std::endl;
    std::cout << "BS Forumla value = " << options[i].PriceByBSFormula(r) 
      << std::endl;
    std::cout << "Absolute error = " << abs(options[i].PriceByBSFormula(r) - optionValues[i]) << std::endl << std::endl;
  }

  cudaFree(dev_options);
  cudaFree(dev_optionValues);
  cudaFree(devStates);
  cudaFree(devQuasiStates);
  cudaFree(dev_directionVectors);
  cudaFree(dev_paths);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
