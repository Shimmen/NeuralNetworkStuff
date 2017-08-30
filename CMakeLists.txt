cmake_minimum_required(VERSION 3.5)
project(ConvergingPatterns)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp matrix.cpp matrix.h)
add_executable(ConvergingPatterns ${SOURCE_FILES})