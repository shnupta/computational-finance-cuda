#include "binomial_gpu.h"
#include "common.h"
#include "option.h"

#include <helper_cuda.h>

typedef struct ProcessedOptionData {
  double S;
  double K;
  double vDt;
  double puByDf;
  double pdByDf;
} ProcessedOptionData;

// Device constant memory for all threads to get option data
static __constant__ ProcessedOptionData dev_OptionData[OPTIONS_NUM];
// Memory not stored in constant memory but still resides on device
static __device__ double dev_CallValues[OPTIONS_NUM];


#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (TIME_STEPS/THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif

__device__ inline double ExpiryCallValue(double S, double K, double vDt, int i)
{
  // Calculates a value of the call option at expiry between final stock prices
  // S^(-TIME_STEPS) to S^(TIME_STEPS), depending on i
  double d = S * exp(vDt * (2.0 * i - TIME_STEPS)) - K;
  return (d > 0.0) ? d : 0.0;
}

__global__ void BinomialOptionsKernel() {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  // callExchange[i] will equal the option value at timestep i * ELEMS_PER_THREAD
  __shared__ double callExchange[THREADBLOCK_SIZE + 1];

  const double S = dev_OptionData[bid].S;
  const double K = dev_OptionData[bid].K;
  const double vDt = dev_OptionData[bid].vDt;
  const double puByDf = dev_OptionData[bid].puByDf;
  const double pdByDf = dev_OptionData[bid].pdByDf;

  double call[ELEMS_PER_THREAD + 1];
  // This thread calculates the expiry option value from underlying final prices
  // (i.e.) it works out the expiration value for ELEMS_PER_THREAD different leaf nodes
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    call[i] = ExpiryCallValue(S, K, vDt,  tid * ELEMS_PER_THREAD + i);
  }

  // Fill the top element of callExchange with the expiration value of the call where
  // the stock price has incremented every dt
  // This is used by thread with tid THREADBLOCK_SIZE-1 (the last thread) to work out
  // its top value (P.O.V. as a horizontal tree) of the next timestep
  if (tid == 0) {
    callExchange[THREADBLOCK_SIZE] = ExpiryCallValue(S, K, vDt, TIME_STEPS);
  }

  // This thread is responsible for working out the tree values until this timestep
  // This is because with each timestep the number of values to calculate decreases
  // Imagine the thread responsible for the top ELEMS_PER_THREAD values at the leaf
  // After ELEMS_PER_THREAD timesteps it has reduced its values to a single call value
  // which is then passed on to the thread below (and it no longer needs to calculate anything)
  int finalIt = max(0, tid * ELEMS_PER_THREAD - 1);

#pragma unroll 16
  for (int i = TIME_STEPS; i > 0; i--) {
    callExchange[tid] = call[0]; // Pass down your bottom call value to the thread below
    __syncthreads();
    call[ELEMS_PER_THREAD] = callExchange[tid + 1]; // Grab the bottom value from the thread above you
    __syncthreads();

    // While you still need to calculate new call values
    // Iterate through your call values from the previous timestep and calculate the values
    // for the next timestep
    if (i > finalIt) {
#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; j++) {
        call[j] = puByDf * call[j+1] + pdByDf * call[j];
      }
    }
  }
}

void BinomialPricingGPU(double *callValues, EuropeanOption *options, int optionsNum) {
  ProcessedOptionData host_OptionData[OPTIONS_NUM];

  // Preprocess the option data
  for (int i = 0; i < optionsNum; i++) {
    const double T = options[i].T;
    const double R = options[i].R;
    const double V = options[i].V;

    const double dt = T / static_cast<double>(TIME_STEPS);
    const double vDt = V * sqrt(dt);
    const double rDt = R * dt;

    // Per-step interest and discount factors
    const double If = exp(rDt);
    const double Df = exp(-rDt);

    // Values and psuedoprobabilities of upwards and downwards moves
    const double u = exp(vDt);
    const double d = exp(-vDt);
    const double pu = (If - d) / (u - d); 
    const double pd = 1.0 - pu;
    const double puByDf = pu * Df;
    const double pdByDf = pd * Df;

    host_OptionData[i].S = options[i].S;
    host_OptionData[i].K = options[i].K;
    host_OptionData[i].vDt = vDt;
    host_OptionData[i].puByDf = puByDf;
    host_OptionData[i].pdByDf = pdByDf;
  }
  
  // Copy host option data to device
  // TODO: Add CUDA error checking
  checkCudaErrors(cudaMemcpyToSymbol(dev_OptionData, host_OptionData, optionsNum * sizeof(ProcessedOptionData)));
  BinomialOptionsKernel<<<optionsNum, THREADBLOCK_SIZE>>>();
  getLastCudaError("BinomialOptionsKernel() failed.\n");
  checkCudaErrors(cudaMemcpyFromSymbol(callValues, dev_CallValues, optionsNum * sizeof(double)));
  
}
