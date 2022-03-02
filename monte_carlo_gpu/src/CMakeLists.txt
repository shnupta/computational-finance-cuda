# set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -G -g") # Debug

add_library(MonteCarloLib option/european.h option/european.cu option/common.h)
set_target_properties(MonteCarloLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(MonteCarloGPU main.cu)
set_target_properties(MonteCarloGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(MonteCarloGPU MonteCarloLib -lcublas -lcurand)
