#ifndef PathDepOption_h
#define PathDepOption_h

#include "BSModel.h"

class PathDepOption {
  public:
    double PriceByMC(BSModel model, long N, double epsilon);
    double PriceByVarRedMC(BSModel model, long N, PathDepOption& cvOption);
    virtual double PriceByBSFormula(BSModel model) { return 0.0; }
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

class DifferenceOfOptions : public PathDepOption {
  public:
    DifferenceOfOptions(double T_, int m_, PathDepOption* ptr1_, PathDepOption* ptr2_)
      : ptr1(ptr1_), ptr2(ptr2_) {
        T = T_;
        m = m_;
      }

    double Payoff(SamplePath& S) {
      return ptr1->Payoff(S) - ptr2->Payoff(S);
    }

  private:
    PathDepOption* ptr1;
    PathDepOption* ptr2;
};

class ArthmAsianCall : public PathDepOption {
  public:
    ArthmAsianCall(double T_, double K_, int m_) : K(K_) { T = T_; m = m_; }
    double Payoff(SamplePath& S);

    double GetK() { return K; }

  private:
    double K;
};

class GmtrAsianCall : public PathDepOption {
  public:
    GmtrAsianCall(double T_, double K_, int m_) : K(K_) { T = T_; m = m_; }
    double Payoff(SamplePath& S);
    double PriceByBSFormula(BSModel model);

  private:
    double K;
};

class PathDepBasketOption {
  public:
    double PriceByMC(BSModelBasket model, long N);
    virtual double Payoff(BasketSamplePath& S) = 0;
  protected:
    double T;
    int m;
};

class ArthmAsianBasketCall : public PathDepBasketOption {
  public:
    ArthmAsianBasketCall(double T_, double K_, int m_) : K(K_) { T = T_; m = m_; }
    double Payoff(BasketSamplePath& S);

  private:
    double K;
};

#endif
