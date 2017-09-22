//
// Created by Simon Moos on 2017-09-18.
//

#ifndef NN_NEURAL_NETWORK_H
#define NN_NEURAL_NETWORK_H

#include <vector>

#include <matrix.h>
#include <random.h>

/**
 * Simple perceptron neural network with one hidden layer.
 */
class NeuralNetwork
{
public:

    typedef double (*ActivationFunction)(double);

    NeuralNetwork(size_t num_inputs, size_t num_hidden, size_t num_outputs, ActivationFunction activation_function, ActivationFunction activation_function_derivative);
    ~NeuralNetwork() {}

    const Matrix<double>& get_output_weights() const;
    const std::vector<double>& get_output_thresholds() const;

    const Matrix<double>& get_hidden_weights() const;
    const std::vector<double>& get_hidden_thresholds() const;

    void reset_weights(double min, double max);
    void reset_thresholds(double min, double max);

    void train(const std::initializer_list<double>& training_input, const std::initializer_list<double>& expected_output, double learning_rate);
    const std::vector<double>& run(std::vector<double> input_data) const;

    const size_t num_inputs;
    const size_t num_hidden;
    const size_t num_outputs;

private:

    double calculate_b_i(size_t i) const;
    double calculate_b_j(size_t j, const std::vector<double> &input_data) const;

    // Used to temporarily store for training and returning results, therefore mutable!
    mutable std::vector<double> hidden_data;
    mutable std::vector<double> output_data;

    Matrix<double> hidden_weights;
    std::vector<double> hidden_thresholds;
    Matrix<double> output_weights;
    std::vector<double> output_thresholds;

    ActivationFunction activation_function;
    ActivationFunction activation_function_derivative;

};

#endif // NN_NEURAL_NETWORK_H
