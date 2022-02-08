#ifndef BSModel_h
#define BSModel_h

#include "Matrix.h"

#include <vector>
#include <ctime>
#include <random>

using namespace std;

typedef vector<double> SamplePath;
typedef vector<Vector> BasketSamplePath;

class BSModel {
  public:
    BSModel(double S0_, double r_, double sigma_) : S0(S0_), r(r_), sigma(sigma_) {
      srand(time(NULL));
    }

    void GenerateSamplePath(double T, int m, SamplePath& S);
    double GetS0() { return S0; }
    double GetR() { return r; }
    double GetSigma() { return sigma; }

  private:
    double S0, r, sigma;
};

class BSModelBasket {
  public:
    BSModelBasket(Vector S0_, double r_, Matrix C_);
    void GenerateSamplePath(double T, int m, BasketSamplePath& S);
    Vector GetS0() { return S0; }
    double GetR() { return r; }
    Vector GetSigma() { return sigma; }


  private:
    Vector S0, sigma;
    double r;
    Matrix C;
};

#endif
