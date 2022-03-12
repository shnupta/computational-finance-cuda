#pragma once

#include "mc_base.h"
#include "gaussians.h"

class Mrg32k3a : public RNG {
	public:
		// Constructor with seed
		Mrg32k3a(const unsigned a = 12345, const unsigned b = 12346) : a(a), b(b) {
			Reset();
		}

		// Reset state to 0 (seed)
		void Reset() {
			xn = xn1 = xn2 = a;
			yn = yn1 = yn2 = b;

			// Anti = false: generate next
			anti = false;
		}

		// Virtual copy constructor
		unique_ptr<RNG> clone() const override {
			return make_unique<Mrg32k3a>(*this);
		}

		// Initialiser
		void Initialise(const size_t simDim) override {
			dim = simDim;
			cachedUniforms.resize(dim);
			cachedGaussians.resize(dim);
		}

		void NextUniform(vector<double>& uniformVec) override {
			if (anti) {
				// Don't generate, negate cached
				transform(cachedUniforms.begin(), cachedUniforms.end(),
						uniformVec.begin(), [](const double d) { return 1.0 - d; });
				// Generate next
				anti = true;
			} else {
				generate(cachedUniforms.begin(), cachedUniforms.end(),
						[this]() { return NextNumber(); });
				copy(cachedUniforms.begin(), cachedUniforms.end(), uniformVec.begin());
				// Don't generate next
				anti = false;
			}
		}

		void NextGaussian(vector<double>& gaussianVec) override {
			if (anti) {
				// Don't generate, negate cached
				transform(cachedGaussians.begin(), cachedGaussians.end(),
						gaussianVec.begin(), [](const double n) { return -n; });
				// Generate next
				anti = false;
			} else {
				generate(cachedGaussians.begin(), cachedGaussians.end(),
						[this]() { return invNormalCdf(NextNumber()); });
				copy(cachedGaussians.begin(), cachedGaussians.end(),
						gaussianVec.begin());
				// Don't generate next
				anti = true;
			}
		}

		// Access dimension
		size_t SimulationDimension() const override {
			return dim;
		}

	private:
		// Seed
		const double a, b;
		// Dimension
		size_t dim;
		// State
		double xn, xn1, xn2, yn, yn1, yn2;
		// Antithetic
		// false: generate new
		// true: negate cached
		bool anti;
		vector<double> cachedUniforms;
		vector<double> cachedGaussians;
		// Constants
		static constexpr double m1 = 4294967087;
		static constexpr double m2 = 4294944443;
		static constexpr double a12 = 1403580;
		static constexpr double a13 = 810728;
		static constexpr double a21 = 527612;
		static constexpr double a23 = 1370589;
		// Divide the final uniform by m1 + 1 so we never hit 1
		static constexpr double m1p1 = 4294967088;

		// Produce next number and update state
		double NextNumber() {
			// Update X (recursion)
			double x = a12 * xn1 - a13 * xn2;
			// Modulus
			x -= long(x / m1) * m1;
			if (x < 0) x += m1;
			// Update
			xn2 = xn1;
			xn1 = xn;
			xn = x;

			// Same for y
			double y = a21 * yn - a23 * yn2;
			y -= long(y / m2) * m2;
      if (y < 0) y += m2;
			yn2 = yn1;
			yn1 = yn;
			yn = y;

			// Uniform
			const double u = x > y ? (x - y) / m1p1 : (x - y + m1) / m1p1;

			return u;
		}
};
