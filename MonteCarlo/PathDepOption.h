#ifndef PATHDEPOPTION_H
#define PATHDEPOPTION_H

#include "BSModel.h"

class PathDepOption {
  public:
    double PriceByMC(BSModel model, long N, double epsilon);
    virtual double Payoff(SamplePath& S) = 0;

    double GetT() { return T; }
    int GetM() { return m; }
    double GetPrice() { return price; }
    double GetPricingError() { return pricingError; }
    double GetDelta() { return delta; }

  protected:
    double T, price, pricingError, delta;
    int m;
};

class ArthmAsianCall : public PathDepOption {
  public:
    ArthmAsianCall(double T_, double K_, int m_) : K(K_) { T = T_; m = m_; }
    double Payoff(SamplePath& S);

    double GetK() { return K; }

  private:
    double K;
};

#endif
