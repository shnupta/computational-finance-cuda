#ifndef BS_MODEL_H
#define BS_MODEL_H

#include <vector>
#include <ctime>
#include <random>

using namespace std;

typedef vector<double> SamplePath;

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

#endif
