#include "european.h"

VanillaEuropean::VanillaEuropean(bool isCall, double strike, double S0,
    double sigma, double ttm) : isCall_(isCall), strike_(strike), S0_(S0),
  sigma_(sigma), ttm_(ttm) {

}

__host__ __device__ double VanillaEuropean::Payoff(double* path, const int size) {
  double diff = path[size-1] - strike_;
  if (!isCall_) diff = -diff;
  return diff > 0 ? diff : 0;
} 

double N(double x)
{
   double gamma = 0.2316419;     double a1 = 0.319381530;
   double a2    =-0.356563782;   double a3 = 1.781477937;
   double a4    =-1.821255978;   double a5 = 1.330274429;
   double pi    = 4.0*atan(1.0); double k  = 1.0/(1.0+gamma*x);
   if (x>=0.0)
   {
      return 1.0-((((a5*k+a4)*k+a3)*k+a2)*k+a1)
                  *k*exp(-x*x/2.0)/sqrt(2.0*pi);
   }
   else return 1.0-N(-x);
}

double VanillaEuropean::d_plus(double r)
{
   return (log(S0_/strike_)+
      (r+0.5*pow(sigma_,2.0))*ttm_)
      /(sigma_*sqrt(ttm_));
}

double VanillaEuropean::d_minus(double r)
{
   return d_plus(r)-sigma_*sqrt(ttm_);
}

double VanillaEuropean::PriceByBSFormula(double r)
{
   return S0_*N(d_plus(r))
      -strike_*exp(-r*ttm_)*N(d_minus(r));
}
