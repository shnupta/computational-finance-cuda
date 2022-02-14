#include "option/european.h"

#include <curand_kernel.h>

#include <iostream>
#include <stdio.h>

#define THREADBLOCK_SIZE 1024

#define r 0.03

__global__ void InitRandomStates(curandState_t* devStates) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence
            number, no offset */
  curand_init(1234, id, 0, &devStates[id]);
}

__device__ void GenerateSamplePath(VanillaEuropean option, double* path,
    const int timeSteps, curandState_t* state) {
  double St = option.GetS0();
  double sigma = option.GetSigma();
  double T = option.GetTTM();
  int m = timeSteps;
  for (int k = 0; k < m; k++) {
    path[k] = St * exp((r - sigma*sigma * 0.5) * (T/m) + sigma * sqrt(T/m) 
        * curand_normal_double(state));
    St = path[k];
  }
}

__global__ void PriceByMC(VanillaEuropean* options, double* optionValues, 
    double* optionDeltas, const int optionsNum, const long simNum, 
    const int timeSteps, curandState_t* devStates) {
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
  double* path = new double[timeSteps];

  while (simNo < simNum) {
    // Generate sample path of this option
    GenerateSamplePath(option, path, timeSteps, state);
    threadPayoff = (i * threadPayoff + option.Payoff(path, timeSteps)) 
      / (i + 1.0);
    double dY_dS0 = (path[timeSteps - 1] / option.GetS0())
      * (path[timeSteps - 1] > option.GetStrike() ? 1.0 : 0.0);
    threadDelta = (i * threadDelta + dY_dS0) / (i + 1.0);
    simNo += THREADBLOCK_SIZE;
    i++;
  }
  payoffs[tid] = threadPayoff;
  deltas[tid] = threadDelta;
  __syncthreads();

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

  const int optionsNum = 1;
  const long simNum = 100000;
  const int timeSteps = 300;

  VanillaEuropean options[optionsNum];
  options[0] = VanillaEuropean(isCall, strike, s0, sigma, ttm);

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

  const int totalThreads = 1024 * optionsNum;
  curandState_t* devStates;
  cudaMalloc((void**) &devStates, totalThreads * sizeof(curandState_t));

  InitRandomStates<<<optionsNum, 1024>>>(devStates);

  PriceByMC<<<optionsNum, 1024>>>(dev_options, dev_optionValues, 
      dev_optionDeltas, optionsNum,
      simNum, timeSteps, devStates);

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
  std::cout << "Option value = " << optionValues[0] << std::endl;
  std::cout << "By BS Forumla = " << options[0].PriceByBSFormula(r) 
    << std::endl;
  std::cout << "Delta = " << optionDeltas[0] << std::endl;

  cudaFree(dev_options);
  cudaFree(dev_optionValues);
  cudaFree(devStates);
}
