cmake_minimum_required(VERSION 3.5)
project(NeuralNetworkStuff)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# ANN stuff
add_subdirectory(converging_pattern)
add_subdirectory(deterministic_hopfield_model)

# Etc. stuff
add_subdirectory(plot_test)
