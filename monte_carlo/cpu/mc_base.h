#pragma once

#include <vector>
#include <string>
#include <memory>

using namespace std;

using Time = double;
extern Time systemTime;

// Definition of what data must be simulates
struct SampleDef {
  // Do we need numeraire for this sample?
  // true for payment dates
  // false otherwise
  bool numeraire = true;

  // Maturities of the forwards on this event date
  // size = number of forwards
  vector<Time> forwardMats;
  // Maturities of the discounts on this event date
  vector<Time> discountMats;

  // Specification of a Libor rate: start date, end date, curve index
  struct RateDef {
    Time start;
    Time end;
    string curve;

    RateDef(const Time start, const Time end, const string& curve)
      : start(start), end(end), curve(curve) {}
  };

  // Specification of the Libors on this event date
  vector<RateDef> liborDefs;
};

// A sample is the collection of market observations on an event date for the 
// evaluation of the payoff
template <class T>
struct Sample {
  T numeraire;
  vector<T> forwards;
  vector<T> discounts;
  vector<T> libors;

  // Allocate given corresponding SampleDef
  void Allocate(const SampleDef& def) {
    forwards.resize(def.forwardMats.size());
    discounts.resize(def.discountMats.size());
    libors.resize(def.liborDefs.size());
  }

  // Initialse defaults
  void Initialise() {
    numeraire = T(1.0);
    fill(forwards.begin(), forwards.end(), T(100.0));
    fill(discounts.begin(), discounts.end(), T(1.0));
    fill(libors.begin(), libors.end(), T(0.0));
  }
};


// Scenarios are collections of samples
// They are the objects that models and products communicate to one another
template <class T>
using Scenario = vector<Sample<T>>;

// Batch allocate a collection of samples
template <class T>
inline void AllocatePath(const vector<SampleDef>& defline, Scenario<T>& path) {
  path.resize(defline.size());
  for (size_t i = 0; i < defline.size(); ++i) {
    path[i].Allocate(defline[i]);
  }
}

// Batch initialise a collection of samples
template <class T>
inline void InitialisePath(Scenario<T>& path) {
  for (auto& scenario : path) scenario.Initialise();
}

template <class T>
class Product {
  public:
    // Access to the product timeline (event dates) along with the sample
    // definitions (defline)
    virtual const vector<Time>& Timeline() const = 0;
    virtual const vector<SampleDef>& Defline() const = 0;

    // Labels of all the payoffs in the product
    // size = number of payoffs in the product
    virtual const vector<string>& PayoffLabels() const = 0;

    // Compute payoffs given a path (on the product timeline)
    // path: one entry per event date
    // payoffs: pre-allocated space for the resulting payoffs
    virtual void Payoffs(const Scenario<T>& path, vector<T>& payoffs)
      const = 0;

    // Virtual copy constructor
    virtual unique_ptr<Product<T>> clone() const = 0;
    
    // Virtual destructor
    virtual ~Product() {}
};

template <class T>
class Model {
  public:
    // Separate allocation and initialisation with product timeline and defline
    virtual void Allocate(const vector<Time>& prdTimeline,
        const vector<SampleDef>& prdDefline) = 0;
    virtual void Initialise(const vector<Time>& prdTimeline,
        const vector<SampleDef>& prdDefline) = 0;

    // Access to the Monte Carlo dimension, after initialisation
    virtual size_t SimulationDimension() const = 0;

    // Simulate a path consuming a vector[SimulationDimension()] 
    // of independent Gaussians
    virtual void GeneratePath(const vector<double>& gaussVec,
        Scenario<T>& path) const = 0;

    // Virtual copy constructor
    virtual unique_ptr<Model<T>> clone() const = 0;

    // Virtual destructor
    virtual ~Model() {}

    // Access to all the model parameters and what they mean
    // size = number of parameters
    virtual const vector<T*>& Parameters() = 0;
    virtual const vector<string>& ParameterLabels() const = 0;

    // Number of parameters
    size_t NumParameters() const {
      // Parameters is not const, we must cast
      // Casting here is okay as we know we don't modify anything
      return const_cast<Model*>(this)->Parameters().size();
    }
};

class RNG {
  public:
    // Initialise with dimension simDim
    virtual void Initialise(const size_t simDim) = 0;

    // Compute the next vector[simDim] of independent Uniforms or Gaussians
    // The vector is filled by the function and must be pre-allocated
    // Non-const as modifies the internal state
    virtual void NextUniform(vector<double>& uniformVec) = 0;
    virtual void NextGaussian(vector<double>& gaussianVec) = 0;

    // Virtual copy constructor
    virtual unique_ptr<RNG> clone() const = 0;

    // Virtual destructor
    virtual ~RNG() {}

    // Access dimension
    virtual size_t SimulationDimension() const = 0;
};

// Monte Carlo simulator: free function that conducts simulations
// and returns a matrix of payoffs
inline vector<vector<double>> MonteCarloSimulation(
    const Product<double>& product,
    const Model<double>& model,
    const RNG& rng,
    const size_t numPaths
    ) {
  // Work with copies of the model and RNG which are modified on set up
  auto modelCopy = model.clone();
  auto rngCopy = rng.clone();

  // Allocate results
  const size_t numPayoffs = product.PayoffLabels().size();
  vector<vector<double>> results(numPaths, vector<double>(numPayoffs));
  // Initialise the simulation timeline
  modelCopy->Allocate(product.Timeline(), product.Defline());
  modelCopy->Initialise(product.Timeline(), product.Defline());
  // Initialise the RNG
  rngCopy->Initialise(modelCopy->SimulationDimension());
  // Allocate the Gaussian vector
  vector<double> gaussianVec(modelCopy->SimulationDimension());
  // Allocate and initialse the path
  Scenario<double> path;
  AllocatePath(product.Defline(), path);
  InitialisePath(path);

  // Iterate over the paths
  for (size_t i = 0; i < numPaths; ++i) {
    // Next Gaussian vector with dimension D
    rngCopy->NextGaussian(gaussianVec);
    // Generate path and consume the Gaussian vector
    modelCopy->GeneratePath(gaussianVec, path);
    // Compute the payoffs
    product.Payoffs(path, results[i]);
  }

  return results;
}
