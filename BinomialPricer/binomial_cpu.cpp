#include "binomial_cpu.h"
#include "common.h"

#include <cmath>

double ExpiryCallValue(double S, double K, double vDt, int i) {
  double d = S * exp(vDt * (2 * i - TIME_STEPS)) - K;
  return (d > 0.0) ? d : 0.0;
}

void BinomialPricingCPU(double &callResult, EuropeanOption option) {
  static double call[TIME_STEPS + 1];

  const double S = option.S;
  const double K = option.K;
  const double T = option.T;
  const double R = option.R;
  const double V = option.V;

  const double dt = T / static_cast<double>(TIME_STEPS);
  const double vDt = V * sqrt(dt);
  const double rDt = R * dt;
  //Per-step interest and discount factors
  const double If = exp(rDt);
  const double Df = exp(-rDt);
  //Values and pseudoprobabilities of upward and downward moves
  const double u = exp(vDt);
  const double d = exp(-vDt);
  const double pu = (If - d) / (u - d);
  const double pd = 1.0 - pu;
  const double puByDf = pu * Df;
  const double pdByDf = pd * Df;

  ///////////////////////////////////////////////////////////////////////
  // Compute values at expiration date:
  // call option value at period end is V(T) = S(T) - X
  // if S(T) is greater than X, or zero otherwise.
  // The computation is similar for put options.
  ///////////////////////////////////////////////////////////////////////
  for (int i = 0; i <= TIME_STEPS; i++)
      call[i] = ExpiryCallValue(S, K, vDt, i);

  ////////////////////////////////////////////////////////////////////////
  // Walk backwards up binomial tree
  ////////////////////////////////////////////////////////////////////////
  for (int i = TIME_STEPS; i > 0; i--)
      for (int j = 0; j <= i - 1; j++)
          call[j] = puByDf * call[j + 1] + pdByDf * call[j];

  callResult = static_cast<double>(call[0]);

}
