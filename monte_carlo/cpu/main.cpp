#include "main.h"

int main() {
  const double spot = 100.0;
  const double strike = 100.0;
  const double vol = 0.2;
  const double rate = 0.03;
  const double ttm = 1.0 / 12.0;

  BlackScholes<double> model(spot, vol, false, rate);
  EuropeanCall<double> product(strike, ttm);

  NumericalParam params{false, false, 32000};

  auto results = Value(model, product, params);
  printf("val = %f\n", results.values[0]);

  return 0;
}
