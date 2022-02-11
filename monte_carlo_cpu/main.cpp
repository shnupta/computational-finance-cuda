#include <iostream>
#include "BSModel.h"
#include "PathDepOption.h"
#include "EurCall.h"

using namespace std;

int main() {
  double S0 = 100.0, r = 0.03, sigma = 0.2;
  BSModel model(S0, r, sigma);

  double T = 1.0/12.0, K = 100.0;
  int m = 30;
  ArthmAsianCall option(T, K, m);
  GmtrAsianCall cvOption(T, K, m);
  EurCall eur(T, K, m);

  long N = 30000;
  double epsilon = 0.001;
  option.PriceByVarRedMC(model, N, cvOption);
  cout << "European Call Price = " << eur.PriceByMC(model, N) << endl;
  cout << "By BS Forumla = " << eur.PriceByBSFormula(S0, sigma, r) << endl << endl;
  cout << "Asian Call Price (w/ Variance Reduction) = " << option.GetPrice() << endl;
  cout << "Pricing Error = " << option.GetPricingError() << endl << endl;
  /* cout << "Option Delta = " << option.GetDelta() << endl; */

  option.PriceByMC(model, N, epsilon);
  cout << "Price by direct MC = " << option.GetPrice() << endl;
  cout << "MC Error = " << option.GetPricingError() << endl << endl;

  cout << "Basket Options" << endl;
  int d = 3;
  Vector vS0(d);
  vS0[0] = 40.0;
  vS0[1] = 60.0;
  vS0[2] = 100.0;
  Matrix C(d, Vector(d));
  C[0][0] = 0.1; C[0][1] = -0.1; C[0][2] = 0.0;
  C[1][0] = -0.1; C[1][1] = 0.2; C[1][2] = 0.0;
  C[2][0] = 0.0; C[2][1] = 0.0; C[2][2] = 0.3;
  BSModelBasket basketModel(vS0, r, C);
  double basketK = 200.0;
  ArthmAsianBasketCall basketCall(T, basketK, m);
  cout << "Arithmetic Basket Call Price = " << basketCall.PriceByMC(basketModel, N) << endl;

  return 0;
}
