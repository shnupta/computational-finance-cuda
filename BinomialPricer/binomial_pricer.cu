#include "common.h"
#include "option.h"
#include "black_scholes.h"
#include "binomial_gpu.h"
#include <iostream>
#include <cmath>

#include <helper_cuda.h>

// Generate a uniformly distributed random float in the range [low, high]
double UniformRandom(double low, double high) {
  double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
  return (1 - r) * low + r * high;
}

int main() {
  const int numOptions = OPTIONS_NUM;
  const int timeSteps = TIME_STEPS;

  float elapsedTimeGPU;
  cudaEvent_t start, end;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&end));

  EuropeanOption options[numOptions];
  double callValueBS[numOptions];
  double callValueCPU[numOptions];
  double callValueGPU[numOptions];

  std::cout << "Generating input option data...\n";

  srand(1896);

  // Generate input option data
  for (int i = 0; i < numOptions; i++) {
    options[i].S = UniformRandom(5.0f, 30.0f); 
    options[i].K = UniformRandom(1.0f, 100.0f);
    options[i].T = UniformRandom(0.25f, 10.0f);
    options[i].R = 0.06f;
    options[i].V = 0.10f;
    // Calculate the value of this option using Black-Scholes formula for comparison later
    BlackScholesCall(callValueBS[i], options[i]);
  }

  std::cout << "Generated " << numOptions << " options.\n";
  std::cout << "Running over " << timeSteps << " time steps.\n";

  std::cout << "Running GPU kernel...\n";
  checkCudaErrors(cudaEventRecord(start, 0));
  BinomialPricingGPU(callValueGPU, options, numOptions);
  checkCudaErrors(cudaEventRecord(end, 0));
  checkCudaErrors(cudaEventSynchronize(end));

  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeGPU, start, end));
  std::cout << "Time taken: " << elapsedTimeGPU << " ms\n";

  return 0;
}
