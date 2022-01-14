#ifndef OPTION_H
#define OPTION_H

#include <cmath>

class EuropeanOption {
  protected:
    double S; // Inital price of the underlying
    double K; // Strike
    double T; // Time to expiration date
    double R; // Risk neutral interest rate
    double V; // Option price

  public:
    EuropeanOption(double S, double K, double T, double R, double V) : S(S), K(K), T(T), R(R), V(V) {}
    EuropeanOption() {}
    virtual double Payoff();

    // Getters
    double GetS() { return S; }
    double GetK() { return K; }
    double GetT() { return T; }
    double GetR() { return R; }
    double GetV() { return V; }
};

class EuropeanCallOption : public EuropeanOption {
  public:
    EuropeanCallOption(double S, double K, double T, double R, double V) : EuropeanOption(S, K, T, R, V) {}
    EuropeanCallOption() {}
    double Payoff() override { return fmax(S - K, 0); }
};

class EuropeanPutOption : public EuropeanOption {
  public:
    EuropeanPutOption(double S, double K, double T, double R, double V) : EuropeanOption(S, K, T, R, V) {}
    EuropeanPutOption() {}
    double Payoff() override { return fmax(K - S, 0); }
};

#endif
