#include "common.h"
#include "option.h"
#include "black_scholes.h"
#include <iostream>
#include <cmath>

// Generate a uniformly distributed random float in the range [low, high]
double UniformRandom(double low, double high) {
  double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
  return (1 - r) * low + r * high;
}

int main() {
  const int numOptions = OPTIONS_NUM;
  const int timeSteps = TIME_STEPS;

  EuropeanCallOption options[numOptions];
  double callValueBS[numOptions];
  double callValueCPU[numOptions];
  double callValueGPU[numOptions];

  std::cout << "Generating input option data...\n";

  srand(1896);

  // Generate input option data
  for (int i = 0; i < numOptions; i++) {
    double S = UniformRandom(5.0f, 30.0f); 
    double K = UniformRandom(1.0f, 100.0f);
    double T = UniformRandom(0.25f, 10.0f);
    double R = 0.06f;
    double V = 0.10f;
    options[i] = EuropeanCallOption(S, K, T, R, V);
    // Calculate the value of this option using Black-Scholes formula for comparison later
    BlackScholesCall(callValueBS[i], options[i]);
  }

  return 0;
}
