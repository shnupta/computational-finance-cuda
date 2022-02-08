#pragma once

#include "mc_base.h"
#include "gaussians.h"

#include <cstring>

#define ONEOVER2POW32 2.3283064365387E-10

const unsigned* const* getjkDir();

class Sobol : public RNG {
  public:
    // Virtual copy constructor
    unique_ptr<RNG> clone() const override {
      return make_unique<Sobol>(*this);
    }

    // Initialiser
    void Initialise(const size_t simDim) override {
      // Set pointer on direction numbers
      jkDir = getjkDir();

      // Dimension
      dim = simDim;
      state.resize(dim);

      Reset();
    }

    void Reset() {
      // Set state to 0
      memset(state.data(), 0, dim * sizeof(unsigned));
      index = 0;
    }

    void Next() {
      // Gray code, find position j of rightmost zero bit of current index n
      unsigned n = index, j = 0;
      while (n & 1) {
        n >>= 1;
        ++j;
      }

      // Direction numbers
      const unsigned* dirNums = jkDir[j];

      // XOR the appropriate direction number into each component of the 
      // integer sequence
      for (size_t i = 0; i < dim; ++i) {
        state[i] ^= dirNums[i];
      }

      // Update count
      ++index;
    }

    void NextUniform(vector<double>& uniformVec) override {
      Next();
      transform(state.begin(), state.end(), uniformVec.begin(),
          [](const unsigned long i) { return ONEOVER2POW32 * i; });
    }

    void NextGaussian(vector<double>& gaussianVec) override {
      Next();
      transform(state.begin(), state.end(), gaussianVec.begin(),
          [](const unsigned long i) { 
            return invNormalCdf(ONEOVER2POW32 * i); 
          });
    }

  private:
    // Dimension
    size_t dim;
    // State Y
    vector<unsigned> state;
    // Current index in the sequence
    unsigned index;
    // The direction numbers listen in sobol.cpp
    // jkDir[i][dim] gives the ith (0 to 31) direction number of dimension dim
    const unsigned* const* jkDir;
};
