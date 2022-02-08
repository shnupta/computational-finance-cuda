#pragma once

#include "mc_base.h"
#include "mc_model_black_scholes.h"
#include "mc_product.h"

#include <unordered_map>
#include <memory>

using namespace std;

#define EPS 1.0e-08

using ModelStore = unordered_map<string, unique_ptr<Model<double>>>;
using ProductStore = unordered_map<string, unique_ptr<Product<double>>>;

ModelStore modelStore;
ProductStore productStore;

void PutBlackScholes(const double spot, const double volatility,
    const bool qSpot, const double rate, const double dividendYield, 
    const string& store) {
  unique_ptr<Model<double>> model = 
    make_unique<BlackScholes<double>>(spot, volatility, qSpot, rate,
        dividendYield);

  modelStore[store] = move(model);
}

// void PutDupire

const Model<double>* GetModel(const string& store) {
  auto it = modelStore.find(store);
  if (it == modelStore.end()) return nullptr;
  else return it->second.get();
}

void PutEuropeanCall(const double strike, const Time exerciseDate,
    const Time settlementDate, const string& store) {
  unique_ptr<Product<double>> product = make_unique<EuropeanCall<double>>(
      strike, exerciseDate, settlementDate
      );

  productStore[store] = move(product);
}

void PutBarrier(const double strike, const double barrier, const Time maturity,
    const double monitorFreq, const double smooth, const string& store) {
  const double smoothFactor = smooth <= 0 ? EPS : smooth;

  unique_ptr<Product<double>> product = make_unique<UpOutCall<double>>(
      strike, barrier, maturity, monitorFreq, smoothFactor);

  productStore[store] = move(product);
}

void PutContingent(const double coupons, const Time maturity,
    const double payFreq, const double smooth, const string& store) {
  const double smoothFactor = smooth <= 0 ? 0.0 : smooth;

  unique_ptr<Product<double>> product = make_unique<ContingentBond<double>>(
      maturity, coupons, payFreq, smoothFactor);

  productStore[store] = move(product);
}

void PutEuropeanCallPortfolio(const vector<double>& maturities,
    const vector<double>& strikes, const string& store) {
  map<Time, vector<double>> options;
  for (size_t i = 0; i< maturities.size(); ++i) {
    options[maturities[i]].push_back(strikes[i]);
  }

  unique_ptr<Product<double>> product = 
    make_unique<EuropeanCallPortfolio<double>>(options);

  productStore[store] = move(product);
}

const Product<double>* GetProduct(const string& store) {
  auto it = productStore.find(store);
  if (it == productStore.end()) return nullptr;
  else return it->second.get();
}
