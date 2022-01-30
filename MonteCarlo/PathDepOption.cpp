#include "PathDepOption.h"
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

double ArthmAsianCall::Payoff(SamplePath& S) {
  double avg = 0.0;
  for (int k = 0; k < m; k++) avg = (k * avg + S[k]) / (k + 1.0);
  if (avg < K) return 0.0;
  return avg - K;
}
