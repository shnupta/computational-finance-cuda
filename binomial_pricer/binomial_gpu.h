#ifndef BINOMIAL_GPU_H
#define BINOMIAL_GPU_H

#include "option.h"

void BinomialPricingGPU(double *callValue, EuropeanOption *options, int optionsNum);

#endif
