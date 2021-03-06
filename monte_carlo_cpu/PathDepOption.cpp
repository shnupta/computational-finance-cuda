#include "PathDepOption.h"
#include "EurCall.h"
#include <cmath>

void Rescale(SamplePath& S, double x) {
  int m = S.size();
  for (int j = 0; j < m; j++) S[j] = x * S[j];
}

double PathDepOption::PriceByMC(BSModel model, long N, double epsilon) {
  double H = 0.0, Hsq = 0.0, Heps = 0.0;
  SamplePath S(m);
  for (long i = 0; i < N; i++) {
    model.GenerateSamplePath(T, m, S);
    H = (i * H + Payoff(S)) / (i + 1.0);
    Hsq = (i * Hsq + pow(Payoff(S), 2.0)) / (i + 1.0); 
    Rescale(S, 1.0 + epsilon);
    Heps = (i * Heps + Payoff(S)) / (i + 1.0);
  }

  price = exp(-model.GetR() * T) * H;
  pricingError = exp(-model.GetR() * T) * sqrt(Hsq - H*H) / sqrt(N - 1.0);
  delta = exp(-model.GetR() * T) * (Heps - H) / (model.GetS0() * epsilon);
  return price;
}

double PathDepOption::PriceByVarRedMC(BSModel model, long N, PathDepOption& cvOption) {
  DifferenceOfOptions varRedOpt(T, m, this, &cvOption);

  price = varRedOpt.PriceByMC(model, N, 0.001) + cvOption.PriceByBSFormula(model);

  pricingError = varRedOpt.pricingError;

  return price;
}

double ArthmAsianCall::Payoff(SamplePath& S) {
  double avg = 0.0;
  for (int k = 0; k < m; k++) avg = (k * avg + S[k]) / (k + 1.0);
  if (avg < K) return 0.0;
  return avg - K;
}

double GmtrAsianCall::Payoff(SamplePath& S) {
  double prod = 1.0;
  for (int i = 0; i < m; i++) {
    prod *= S[i];
  }
  if (pow(prod, 1.0/m) < K) return 0.0;
  return pow(prod, 1.0/m) - K;
}

double GmtrAsianCall::PriceByBSFormula(BSModel model) {
  double a = exp(-model.GetR() * T) * model.GetS0() * exp((m + 1.0) * T / (2.0 * m) * (model.GetR()
      + model.GetSigma()*model.GetSigma() * ((2.0 * m + 1.0) / (3.0 * m) - 1.0) / 2.0));
  double b = model.GetSigma() * sqrt((m + 1.0) * (2.0 * m + 1.0) / (6.0 * m*m));

  EurCall G(T, K);
  price = G.PriceByBSFormula(a, b, model.GetR());
  return price;
}

double PathDepBasketOption::PriceByMC(BSModelBasket model, long N) {
  double H = 0.0;
  BasketSamplePath S(m);
  for (long i = 0; i < N; i++) {
    model.GenerateSamplePath(T, m, S);
    H = (i * H + Payoff(S)) / (i + 1.0);
  }
  return exp(-model.GetR() * T) * H;
}

double ArthmAsianBasketCall::Payoff(BasketSamplePath& S) {
  double avg = 0.0;
  int d = S[0].size();
  Vector one(d);
  for (int i = 0; i < d; i++) one[i] = 1.0;
  for (int k = 0; k < m; k++) {
    avg = (k * avg + (one ^ S[k])) / (k + 1.0);
  }
  if (avg < K) return 0.0;
  return avg - K;
}
