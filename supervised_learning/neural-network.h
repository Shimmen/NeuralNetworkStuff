//
// Created by Simon Moos on 2017-09-18.
//

#ifndef NN_NEURAL_NETWORK_H
#define NN_NEURAL_NETWORK_H

#include <vector>

#include <matrix.h>
#include <random.h>

/**
 * Simple perceptron neural network with no hidden layers.
 */
class NeuralNetwork
{
public:

    typedef double (*ActivationFunction)(double);

    NeuralNetwork(size_t num_inputs, size_t num_outputs, ActivationFunction activation_function, ActivationFunction activation_function_derivative);
    ~NeuralNetwork() {}

    const Matrix<double>& get_weights() const;
    const std::vector<double>& get_thresholds() const;

    void reset_weights(double min, double max);
    void reset_thresholds(double min, double max);

    void train(const std::initializer_list<double>& training_input, const std::initializer_list<double>& expected_output, double learning_rate);
    const std::vector<double>& run(std::vector<double> input_data) const;

    const size_t num_inputs;
    const size_t num_outputs;

private:

    double calculate_local_field(size_t i, const std::vector<double>& input_data) const;

    // Used to temporarily store outputs for training and returning results, therefore mutable!
    mutable std::vector<double> outputs;

    //std::vector<double> weights;
    Matrix<double> weights;
    std::vector<double> thresholds;

    ActivationFunction activation_function;
    ActivationFunction activation_function_derivative;

};

#endif // NN_NEURAL_NETWORK_H
