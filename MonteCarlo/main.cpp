#include <iostream>
#include "BSModel.h"
#include "PathDepOption.h"

using namespace std;

int main() {
  double S0 = 100.0, r = 0.03, sigma = 0.2;
  BSModel model(S0, r, sigma);

  double T = 1.0/12.0, K = 100.0;
  int m = 30;
  ArthmAsianCall option(T, K, m);

  long N = 30000;
  double epsilon = 0.001;
  cout << "Asian Call Price = " << option.PriceByMC(model, N, epsilon) << endl;
  cout << "Pricing Error = " << option.GetPricingError() << endl;
  cout << "Option Delta = " << option.GetDelta() << endl;

  return 0;
}
