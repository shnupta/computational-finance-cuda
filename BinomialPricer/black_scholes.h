#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include "option.h"
#include <cmath>

static double CND(double d)
{
  const double       A1 = 0.31938153;
  const double       A2 = -0.356563782;
  const double       A3 = 1.781477937;
  const double       A4 = -1.821255978;
  const double       A5 = 1.330274429;
  const double RSQRT2PI = 0.39894228040143267793994605993438;

  double
  K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double
  cnd = RSQRT2PI * exp(- 0.5 * d * d) *
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0)
      cnd = 1.0 - cnd;

  return cnd;
}

void BlackScholesCall(double &callResult, EuropeanOption option)
{
  double S = option.S;
  double K = option.K;
  double T = option.T;
  double R = option.R;
  double V = option.V;

  double sqrtT = sqrt(T);
  double    d1 = (log(S / K) + (R + (double)0.5 * V * V) * T) / (V * sqrtT);
  double    d2 = d1 - V * sqrtT;
  double CNDD1 = CND(d1);
  double CNDD2 = CND(d2);

  //Calculate Call and Put simultaneously
  double expRT = exp(- R * T);
  callResult   = (double)(S * CNDD1 - K * expRT * CNDD2);
}

#endif
