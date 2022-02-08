#include "common.h"
#include "option.h"
#include "black_scholes.h"
#include "binomial_gpu.h"
#include "binomial_cpu.h"
#include <iostream>
#include <cmath>
#include <chrono>

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
  cudaEvent_t startGpu, endGpu;
  checkCudaErrors(cudaEventCreate(&startGpu));
  checkCudaErrors(cudaEventCreate(&endGpu));

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
  std::cout << "Running over " << timeSteps << " time steps.\n\n";

  std::cout << "Running GPU kernel...\n";
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(startGpu, 0));
  BinomialPricingGPU(callValueGPU, options, numOptions);
  checkCudaErrors(cudaEventRecord(endGpu, 0));
  checkCudaErrors(cudaEventSynchronize(endGpu));

  checkCudaErrors(cudaEventElapsedTime(&elapsedTimeGPU, startGpu, endGpu));
  std::cout << "Time taken: " << elapsedTimeGPU << " ms\n\n";

  std::cout << "Running CPU version...\n";
  std::chrono::steady_clock::time_point startCpu = std::chrono::steady_clock::now();
  for (int i = 0; i < numOptions; i++) {
    BinomialPricingCPU(callValueCPU[i], options[i]);
  }
  std::chrono::steady_clock::time_point endCpu = std::chrono::steady_clock::now();
  std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(endCpu - startCpu).count() << " ms\n\n";

  std::cout << "Comparing the results...\n";
  double sumDelta = 0;
  double sumRef   = 0;
  std::cout << "GPU binomial vs. Black-Scholes\n";

  for (int i = 0; i < numOptions; i++)
  {
      sumDelta += fabs(callValueBS[i] - callValueGPU[i]);
      sumRef += fabs(callValueBS[i]);
  }

  if (sumRef > 1E-5)
  {
    std::cout<< "L1 norm: " << static_cast<double>(sumDelta / sumRef) << std::endl << std::endl;
  }
  else
  {
    std::cout << "Avg. diff: " << (sumDelta / numOptions) << std::endl << std::endl;
  }

  std::cout << "CPU binomial vs. Black-Scholes\n";
  sumDelta = 0;
  sumRef   = 0;

  for (int i = 0; i < numOptions; i++)
  {
      sumDelta += fabs(callValueBS[i]- callValueCPU[i]);
      sumRef += fabs(callValueBS[i]);
  }

  if (sumRef >1E-5)
  {
    std::cout<< "L1 norm: " << static_cast<double>(sumDelta / sumRef) << std::endl << std::endl;
  }
  else
  {
    std::cout << "Avg. diff: " << (sumDelta / numOptions) << std::endl << std::endl;
  }

  std::cout<< "CPU binomial vs. GPU binomial\n";
  sumDelta = 0;
  sumRef   = 0;

  for (int i = 0; i < numOptions; i++)
  {
      sumDelta += fabs(callValueGPU[i] - callValueCPU[i]);
      sumRef += callValueCPU[i];
  }

  double errorVal;
  if (sumRef > 1E-5)
  {
    errorVal = static_cast<double>(sumDelta / sumRef);
    std::cout<< "L1 norm: " << static_cast<double>(sumDelta / sumRef) << std::endl << std::endl;
  }
  else
  {
    std::cout << "Avg. diff: " << (sumDelta / numOptions) << std::endl << std::endl;
  }

  std::cout << "Shutting down...\n\n";

  checkCudaErrors(cudaEventDestroy(startGpu));
  checkCudaErrors(cudaEventDestroy(endGpu));

  if (errorVal > 5e-4)
  {
    std::cout << "Test failed!\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "Test passed\n";
  exit(EXIT_SUCCESS);
}
