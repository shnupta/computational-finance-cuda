cmake_minimum_required(VERSION 3.16)
project(BinomialPricer LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

include_directories(/homes/cjw19/dev/year3/individual/temp/samples/common/inc)

add_executable(BinomialPricer binomial_pricer.cu binomial_gpu.h binomial_gpu.cu black_scholes.h common.h option.h)
