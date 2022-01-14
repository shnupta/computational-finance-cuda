#include "binomial_gpu.h"
#include "common.h"
#include "option.h"

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
static __device__ double dev_CallValue[OPTIONS_NUM];

__global__ void BinomialKernel() {

}

void BinomialPricingGPU(double *callValue, EuropeanOption *option, int optionsNum) {

}
