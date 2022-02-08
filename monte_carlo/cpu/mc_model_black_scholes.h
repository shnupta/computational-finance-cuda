#pragma once

#include "mc_base.h"

#include <cmath>

template<class T>
class BlackScholes : public Model<T> {
  public:
    template <class U>
    BlackScholes(const U spot, const U volatility,
        const bool spotMeasure = false, const U rate = U(0.0),
        const U dividendYield = U(0.0)) :
      spot(spot), volatility(volatility), rate(rate),
      dividendYield(dividendYield), isSpotMeasure(spotMeasure),
      parameters(4), parameterLabels(4) {
        parameterLabels[0] = "spot";
        parameterLabels[1] = "volatility";
        parameterLabels[2] = "rate";
        parameterLabels[3] = "dividend yield";

        SetParamPointers();
      }

    T Spot() const {
      return spot;
    }

    const T Volatility() const {
      return volatility;
    }

    const T Rate() const {
      return rate;
    }

    const T DividendYield() const {
      return dividendYield;
    }

    const vector<T*>& Parameters() override {
      return parameters;
    }

    const vector<string>& ParameterLabels() const override {
      return parameterLabels;
    }

    unique_ptr<Model<T>> clone() const override {
      auto clone = make_unique<BlackScholes<T>>(*this);
      // Must reset pointers to it's own copies of the parameters
      clone->SetParamPointers();
      return clone;
    }

    void Allocate(const vector<Time>& productTimeline,
        const vector<SampleDef>& defline) override {
      // Simulation timeline = today + product timeline
      timeline.clear();
      timeline.push_back(systemTime);
      for (const auto& time: productTimeline) {
        if (time > systemTime) timeline.push_back(time);
      }

      isTodayOnTimeline = (productTimeline[0] == systemTime);
      defline = &defline;

      stds.resize(timeline.size() - 1);
      drifts.resize(timeline.size() - 1);

      const size_t n = productTimeline.size();
      numeraires.resize(n);
      discounts.resize(n);
      for (size_t j = 0; j < n; ++j) {
        discounts[j].resize(defline[j].discountMats.size());
      }
      forwardFactors.resize(n);
      for (size_t j = 0; j < n; ++j ) {
        forwardFactors[j].resize(defline[j].forwardMats.size());
      }
      libors.resize(n);
      for (size_t j = 0; j < n; ++j) {
        libors[j].resize(defline[j].liborDefs.size());
      }
    }

    void Initialise(const vector<Time>& productTimeline,
        const vector<SampleDef>& defline) override {
      // Pre-compute the standard deviations and drifts over simulation timeline
      const T mu = rate - dividendYield;
      const size_t n = timeline.size() - 1;

      for (size_t i = 0; i < n; ++i) {
        const double dt = timeline[i+1] - timeline[i];
        stds[i] = volatility * sqrt(dt);

        if (isSpotMeasure) {
          drifts[i] = (mu + 0.5 * volatility * volatility) * dt;
        } else {
          drifts[i] = (mu - 0.5 * volatility * volatility) * dt;
        }
      }

      // Pre-compute ...
      const size_t m = productTimeline.size();
      for (size_t i = 0; i < m; ++i) {
        if (defline[i].numeraire) {
          if (isSpotMeasure) {
            numeraires[i] = exp(dividendYield * productTimeline[i]) / spot;
          } else {
            numeraires[i] = exp(rate * productTimeline[i]);
          }
        }

        // Forward factors
        const size_t pFF = defline[i].forwardMats.size();
        for (size_t j = 0; j < pFF; ++j) {
          forwardFactors[i][j] =
            exp(mu * (defline[i].forwardMats[j] - productTimeline[i]));
        }

        // Discount factors
        const size_t pDF = defline[i].discountMats.size();
        for (size_t j = 0; j < pDF; ++j) {
          discounts[i][j] =
            exp(-rate * (defline[i].discountMats[j] - productTimeline[i]));
        }

        // Libors
        const size_t pL = defline[i].liborDefs.size();
        for (size_t j = 0; j < pL; ++j) {
          const double dt = 
            defline[i].liborDefs[j].end - defline[i].liborDefs[j].start;
          libors[i][j] = (exp(rate * dt) - 1.0) / dt;
        }
      }
    }

    size_t SimulationDimension() const override {
      return timeline.size() - 1;
    }

    void GeneratePath(const vector<double>& gaussianVec, Scenario<T>& path) 
      const override 
    {
      // Spot price today
      T spot = spot;
      size_t index = 0;
      if (isTodayOnTimeline) {
        FillScen(index, spot, path[index], (*defline)[index]);
        ++index;
      }

      // Iterate through timeline and apply sampling scheme
      const size_t n = timeline.size() - 1;
      for (size_t i = 0; i < n; ++i) {
        spot = spot * exp(drifts[i] + stds[i] * gaussianVec[i]);

        // Store in the sample
        FillScen(index, spot, path[index], (*defline)[index]);
        ++index;
      }
    }

  private:
    void SetParamPointers() {
      parameters[0] = &spot;
      parameters[1] = &volatility;
      parameters[2] = &rate;
      parameters[3] = &dividendYield;
    }

    // Mapping functino, fills a Sample given the spot
    inline void FillScen(const size_t index, const T& spot, Sample<T>& scen,
        const SampleDef& def) const {
      if (def.numeraire) {
        scen.numeraire = numeraires[index];
        if (isSpotMeasure) scen.numeraire *= spot;
      }

      // Fill forwards
      transform(forwardFactors[index].begin(), forwardFactors[index].end(),
          scen.forwards.begin(),
          [&spot](const T& ff) {
            return spot * ff;
          });

      // Fill discounts and libors
      copy(discounts[index].begin(), discounts[index].end(),
          scen.discounts.begin());
      copy(libors[index].begin(), libors[index].end(),
          scen.libors.begin());
    }

    // These would be picked on the linear market in a production system
    T spot; // Today's spot
    T rate; // Constant rate
    T dividendYield;
    T volatility;

    // false = risk neutral measure
    // true = spot measure
    const bool isSpotMeasure;

    // SImulation timeline = today + product timeline
    // Black and Scholes implement big steps because its transition
    // probabilities are known and cheap and mapping is time-wise
    vector<Time> timeline;
    bool isTodayOnTimeline;

    // Reference to product's defline
    const vector<SampleDef>* defline;

    // Pre-calculated on initialisation
    // For the Gaussian traditional distributions

    vector<T> stds;
    vector<T> drifts;

    // For mapping spot to sample
    // forward factors exp((r - d) * (T - t))
    vector<vector<T>> forwardFactors;

    // Pre-calculated ...
    vector<T> numeraires;
    vector<vector<T>> discounts;
    vector<vector<T>> libors;

    // Exported parameters
    vector<T*> parameters;
    vector<string> parameterLabels;
};
