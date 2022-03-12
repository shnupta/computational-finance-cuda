cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(MonteCarloCPU LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

# include_directories(/homes/cjw19/dev/year3/individual/temp/samples/common/inc)

add_executable(monte_carlo_cpu 
	main.cpp
	main.h
	mc_base.h mc_base.cpp
	mc_product.h
	mc_model_black_scholes.h	
	mrg32k3a.h
	sobol.h sobol.cpp
	store.h
	)
