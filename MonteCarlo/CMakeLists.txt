cmake_minimum_required(VERSION 3.16)
project(MonteCarlo LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

include_directories(/homes/cjw19/dev/year3/individual/temp/samples/common/inc)

add_executable(MonteCarlo main.cpp BSModel.h BSModel.cpp PathDepOption.h PathDepOption.cpp)
