#ifndef EurCall_h
#define EurCall_h

#include "BSModel.h"

class EurCall
{
  public:
		double T, K;
    int m;
		EurCall(double T_, double K_, int m_){T=T_; K=K_; m=m_;}
    EurCall(double T_, double K_) {T=T_; K=K_;}
		double d_plus(double S0, double sigma, double r);
		double d_minus(double S0, double sigma, double r);
    double PriceByMC(BSModel model, long N);
		double PriceByBSFormula(double S0,
			 double sigma, double r);
		double VegaByBSFormula(double S0,
			 double sigma, double r);
};

#endif

