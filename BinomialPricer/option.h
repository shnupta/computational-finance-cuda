#ifndef OPTION_H
#define OPTION_H

typedef struct EuropeanOption {
    double S; // Inital price of the underlying
    double K; // Strike
    double T; // Time to expiration date
    double R; // Risk neutral interest rate
    double V; // Option price
} EuropeanOption;

#endif
