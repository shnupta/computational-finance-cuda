#pragma once

#include "mc_base.h"
#include "mc_product.h"
#include "mc_model_black_scholes.h"
#include "mrg32k3a.h"
#include "sobol.h"
#include "store.h"

#include <numeric>
#include <fstream>

using namespace std;

struct NumericalParam {
  bool parallel;
  bool useSobol;
  int numPath;
  int seed1 = 12345;
  int seed2 = 1234;
};

inline auto Value(const Model<double>& model, const Product<double>& product,
    const NumericalParam& num) {
  // Random number generator
  unique_ptr<RNG> rng;
  if (num.useSobol) rng = make_unique<Sobol>();
  else rng = make_unique<Mrg32k3a>(num.seed1, num.seed2);

  // Simulate
  /* const auto resultMat = num.parallel ? */ 
  const auto resultMat = MonteCarloSimulation(product, model, *rng, num.numPath);

  // Returns 2 vectors: the payoff identifiers and their values
  struct {
    vector<string> identifiers;
    vector<double> values;
  } results;

  // Average over paths
  const size_t nPayoffs = product.PayoffLabels().size();
  results.identifiers = product.PayoffLabels();
  results.values.resize(nPayoffs);
  for (size_t i = 0; i < nPayoffs; ++i) {
    results.values[i] = accumulate(resultMat.begin(), resultMat.end(), 0.0,
        [i](const double acc, const vector<double>& v) {
        return acc + v[i];
        }) / num.numPath;
  }

  return results;
}

// Generic valuation
inline auto Value(const string& modelId, const string& productId,
    const NumericalParam& num) {
  const Model<double>* model = GetModel(modelId);
  const Product<double>* product = GetProduct(productId);

  if (!model || !product) {
    throw runtime_error("Value() : Could not retrieve model and product");
  }

  return Value(*model, *product, num);
}
