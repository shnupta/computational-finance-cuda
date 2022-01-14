project(BinomialPricer LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

add_executable(BinomialPricer binomial_pricer.cu)
