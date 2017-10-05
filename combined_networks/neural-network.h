//
// Created by Simon Moos on 2017-09-18.
//

#ifndef NN_NEURAL_NETWORK_H
#define NN_NEURAL_NETWORK_H

#include <vector>
#include <random>

#include <matrix.h>

/**
 * Simple perceptron neural network with no hidden layers.
 */
class NeuralNetwork
{
public:

    typedef double (*ActivationFunction)(double);

    NeuralNetwork();
    NeuralNetwork(size_t num_inputs, size_t num_outputs, ActivationFunction activation_function, ActivationFunction activation_function_derivative);
    ~NeuralNetwork() = default;

    const Matrix<double>& get_output_weights() const;
    const std::vector<double>& get_output_thresholds() const;

    void reset_weights(std::default_random_engine rng, double min, double max);
    void reset_thresholds(std::default_random_engine rng, double min, double max);

    void train(const std::vector<double>& training_input, const std::vector<double>& expected_output, double learning_rate);
    const std::vector<double>& run(std::vector<double> input_data) const;

    const size_t num_inputs;
    const size_t num_outputs;

private:

    double calculate_b_i(size_t i, const std::vector<double> &input_data) const;

    // Used to temporarily store outputs for training and returning results, therefore mutable!
    mutable std::vector<double> output_data;

    //std::vector<double> weights;
    Matrix<double> output_weights;
    std::vector<double> output_thresholds;

    ActivationFunction activation_function;
    ActivationFunction activation_function_derivative;

};

#endif // NN_NEURAL_NETWORK_H
