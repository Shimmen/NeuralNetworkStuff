cmake_minimum_required(VERSION 3.5)
project(NeuralNetworkStuff)

# ANN stuff
add_subdirectory(converging_pattern)
add_subdirectory(deterministic_hopfield_model)
add_subdirectory(stochastic_hopfield_model)
add_subdirectory(supervised_learning)
add_subdirectory(supervised_learning_hidden_layer)

# Etc. stuff
add_subdirectory(plot_test)
