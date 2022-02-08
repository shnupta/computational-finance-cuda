#pragma once

#include "mc_base.h"

#include <sstream>
#include <map>

#define ONE_HOUR 0.000114469
#define ONE_DAY 0.003773585

template <class T>
class EuropeanCall : public Product<T> {
  public:
    // Constructor
    // Store data, build timeline, defline and labels
    EuropeanCall(const double strike, const Time exerciseDate,
        const Time settlementDate)
      : strike(strike), 
        exerciseDate(exerciseDate),
        settlementDate(settlementDate) 
    {
      // Timeline = { exercise date }
      timeline.push_back(exerciseDate);

      // Defline: one sample on the exercise date
      defline.resize(1);
      // Payment of a cash flow ==> numeraire required
      defline[0].numeraire = true;
      // One forward to settlement date
      defline[0].forwardMats.push_back(settlementDate);
      // One discount to settlement date
      defline[0].discountMats.push_back(settlementDate);

      // Identify the product
      // One payoff = "call strike K exercise date settlement date"
      ostringstream ss;
      ss.precision(2);
      ss << fixed;
      if (settlementDate == exerciseDate) {
        ss << "call " << strike << " " << exerciseDate; 
      } else {
        ss << "call " << strike << " " << exerciseDate << " " 
          << settlementDate;
      }
      labels[0] = ss.str();
    }

    // Another constructor implicitly settlement date = exercise date
    EuropeanCall(const double strike, const Time exerciseDate) 
      : EuropeanCall(strike, exerciseDate, exerciseDate) {} 

    // Virtual copy constructor
    unique_ptr<Product<T>> clone() const override {
      return make_unique<EuropeanCall<T>>(*this);
    }

    // Accessors
    
    const vector<Time>& Timeline() const override {
      return timeline;
    }

    const vector<SampleDef>& Defline() const override {
      return defline;
    }

    const vector<string>& PayoffLabels() const override {
      return labels;
    }

    // Computation of the payoff on a path
    void Payoffs(const Scenario<T>& path, vector<T>& payoffs) const override {
      // path[0] = exercise date as specified in timeline
      // forwards[0] = forward (exercise, settlement)
      // discounts[0] = discount (exercise, settlement)
      // numeraire at exercise as specified in definition
      payoffs[0] = max(path[0].forwards[0] - strike, 0.0)
        * path[0].discounts[0] / path[0].numeraire;
    }

  private:
    // Internal data
    double strike;
    Time exerciseDate;
    Time settlementDate;

    // Timeline, defline and payoff labels
    vector<Time> timeline;
    vector<SampleDef> defline;
    vector<string> labels;
};

// Up and out call barrier option
template <class T>
class UpOutCall : public Product<T> {
  public:
    UpOutCall(const double strike, const double barrier, const Time maturity,
        const Time monitorFreq, const double smoothingFactor) : strike(strike),
          barrier(barrier), maturity(maturity),
          smoothingFactor(smoothingFactor), labels(2) {
      // Timeline = system date to maturity with an event date every
      // monitoring period
      timeline.push_back(systemTime);
      Time t = systemTime + monitorFreq;

      // Barrier monitoring
      while (maturity - t > ONE_HOUR) {
        timeline.push_back(t);
        t += monitorFreq;
      }

      // Maturity
      timeline.push_back(maturity);

      // Defiline
      const size_t n = timeline.size();
      defline.resize(n);
      for (size_t i = 0; i < n; ++i) {
        // Only last event date is a payment date
        // so numeraire only required on last event date
        defline[i].numeraire = false;
        // spot(t) = forward (t, t) required on every event date for barrier
        // monitoring and cash-flow determination
        defline[i].forwardMats.push_back(timeline[i]);
      }
      defline.back().numeraire = false;

      ostringstream ss;
      ss.precision(2);
      ss << fixed;

      // Second payoff = European
      ss << "call " << maturity << " " << strike;
      labels[1] = ss.str();

      // First payoff = barrier
      ss << "up and out " << barrier << " monitoring freq " << monitorFreq
        << " smooth " << smoothingFactor;
      labels[0] = ss.str();
    }

    unique_ptr<Product<T>> clone() const override {
      return make_unique<UpOutCall<T>>(*this);
    }

    // Accessors

    const vector<Time>& Timeline() const override {
      return timeline;
    }

    const vector<SampleDef>& Defline() const override {
      return defline;
    }    

    const vector<string>& PayoffLabels() const override {
      return labels;
    }

    void Payoffs(const Scenario<T>& path, vector<T>& payoffs) const override {
      // Application of the smooth barrier technique to stabilise risks
      // Smoothing factor = x% of the spot both ways (untemplated)
      const double smooth = double(path[0].forwards[0] * smoothingFactor);
      const double twoSmooth = 2 * smooth;
      const double barSmooth = barrier + smooth;

      // Start alive
      T alive(1.0);

      // Iterate trhough path, update alive status
      for (const auto& sample : path) {
        // Breached
        if (sample.forwards[0] > barSmooth) {
          alive = T(0.0);
          break;
        }

        // Semi-breached: apply smoothing
        if (sample.forwards[0] > barrier - smooth) {
          alive *= (barSmooth - sample.forwards[0]) / twoSmooth;
        }
      }

      // Payoffs
      // European
      payoffs[1] = max(path.back().forwards[0] - strike, 0.0)
        / path.back().numeraire;
      // Barrier
      payoffs[0] = alive * payoffs[1];
    }

  private:
    double strike;
    double barrier;
    Time maturity;
    double smoothingFactor;

    vector<Time> timeline;
    vector<SampleDef> defline;
    vector<string> labels;
};

template <class T>
class EuropeanCallPortfolio : public Product<T> {
  public:
    EuropeanCallPortfolio(const map<Time, vector<double>>& options) {
      const size_t n = options.size(); // num of maturities

      // Timeline = each maturity is an event date
      for (const pair<Time, vector<double>>& p : options) {
        timeline.push_back(p.first);
        strikes.push_back(p.second);
      }

      // Defline = numeraire and spot(t) = forward(t,t) on every maturity
      defline.resize(n);
      for (size_t i = 0; i < n; ++i) {
        defline[i].numeraire = true;
        defline[i].forwardMats.push_back(timeline[i]);
      }

      // Identify all the payoffs by maturity and strike
      for (const auto& option: options) {
        for (const auto& strike : option.second) {
          ostringstream ss;
          ss.precision(2);
          ss << fixed;
          ss << "call " << option.first << " " << strike;
          labels.push_back(ss.str());
        }
      }
    }

    unique_ptr<Product<T>> clone() const override {
      return make_unique<EuropeanCallPortfolio<T>>(*this);
    }

    // Accessors

    const vector<Time>& Timeline() const override { 
      return timeline;
    }

    const vector<vector<double>>& Strikes() const {
      return strikes;
    }

    const vector<SampleDef>& Defline() const override {
      return defline;
    }

    const vector<string>& PayoffLabels() const override {
      return labels;
    }

    void Payoffs(const Scenario<T>& path, vector<T>& payoffs) const override {
      const size_t numT = timeline.size();
      auto payoffIt = payoffs.begin();
      for (size_t i = 0; i < numT; ++i) {
        transform(strikes[i].begin(), strikes[i].end(), payoffIt,
            [spot = path[i].forwards[0], num = path[i].numeraire]
            (const double& k)
            {
              return max(spot - k, 0.0) / num;
            });
        payoffIt += strikes[i].size();
      }
    }

  private:
    vector<Time> timeline;
    vector<vector<double>> strikes;
    vector<SampleDef> defline;
    vector<string> labels;
};

// Payoff = sum { (liber(Ti, Ti+1) + cpn) * coverage(Ti, Ti+1) if Si+1 >= Si }
template <class T>
class ContingentBond : public Product<T> {
  public:
    ContingentBond(const Time maturity, const double coupons,
        const Time payFreq, const double smoothingFactor) : maturity(maturity),
    coupons(coupons), smoothingFactor(smoothingFactor), labels(1) {
      timeline.push_back(systemTime);
      Time t = systemTime + payFreq;

      // Payment schedule
      while (maturity - t > ONE_DAY) {
        coverages.push_back(t - timeline.back());
        timeline.push_back(t);
        t += payFreq;
      }

      // Maturity
      coverages.push_back(maturity - timeline.back());
      timeline.push_back(maturity);

      const size_t n = timeline.size();
      defline.resize(n);
      for (size_t i = 0; i < n; ++i) {
        // spot(Ti) = forward(Ti, Ti) on every step
        defline[i].forwardMats.push_back(timeline[i]);

        // libor(Ti, Ti+1) and discount(Ti, Ti+1) on every but last step
        if (i < n - 1) {
          defline[i].liborDefs.push_back(
              SampleDef::RateDef(timeline[i], timeline[i+1], "libor"));
        }

        // Numeraire on every step but first as not payment date
        defline[i].numeraire = i > 0;
      }

      ostringstream ss;
      ss.precision(2);
      ss << fixed;
      ss << "contingent bond " << maturity << " " << coupons;
      labels[0] = ss.str();
    }

    unique_ptr<Product<T>> clone() const override {
      return make_unique<ContingentBond>(*this);
    }

    // Accessors

    const vector<Time>& Timeline() const override {
      return timeline;
    }

    const vector<SampleDef>& Defline() const override {
      return defline;
    }

    const vector<string>& PayoffLabels() const override {
      return labels;
    }

    void Payoffs(const Scenario<T>& path, vector<T>& payoffs) const override {
      const double smooth = double(path[0].forwards[0] * smoothingFactor);
      const double twoSmooth = 2 * smooth;

      // Period by period
      const size_t n = path.size() - 1;
      payoffs[0] = 0;
      for (size_t i = 0; i < n; ++i) {
        const auto& start = path[i];
        const auto& end = path[i+1];

        const T s0 = start.forwards[0];
        const T s1 = end.forwards[0];

        T digital;

        // In
        if (s1 - s0 > smooth) {
          digital = T(1.0);
        } else if (s1 - s0 < -smooth) { // Out
          digital = T(0.0);
        } else { // "Fuzzy" region = interpolate
          digital = (s1 - s0 + smooth) / twoSmooth;
        }

        // ~smoothing
        payoffs[0] += digital * (start.libors[0] + coupons) * coverages[i]
          / end.numeraire;
        payoffs[0] += 1.0 / path.back.numeraire;
      }
    }

  private:
    Time maturity;
    double coupons;
    double smoothingFactor;

    vector<Time> timeline;
    vector<SampleDef> defline;
    vector<string> labels;

    // Pre-computed coverages
    vector<double> coverages;
};
