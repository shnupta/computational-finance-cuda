#pragma once

#include "common.h"

class VanillaEuropean {
  public:
    VanillaEuropean(bool isCall, double strike, double S0, double sigma,
        double ttm);
    VanillaEuropean() = default;

    __host__ double PriceByBSFormula(double r);
    __host__ __device__ double Payoff(double* path, const int size);
    __host__ __device__ double GetS0() { return S0_; }
    __host__ __device__ double GetSigma() { return sigma_; }
    __host__ __device__ double GetTTM() { return ttm_; }
    __host__ __device__ double GetStrike() { return strike_; }
    __host__ __device__ bool IsCall() { return isCall_; }

  private:
    __host__ double d_plus(double r);
    __host__ double d_minus(double r);

    bool isCall_;
    double strike_;
    double S0_;
    double sigma_;
    double ttm_; // Time to maturity
};
