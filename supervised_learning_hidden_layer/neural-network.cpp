#include "neural-network.h"

NeuralNetwork::NeuralNetwork(size_t num_inputs, size_t num_hidden, size_t num_outputs, ActivationFunction activation_function, ActivationFunction activation_function_derivative)
    : num_inputs(num_inputs)
    , num_hidden(num_hidden)
    , num_outputs(num_outputs)
    , activation_function(activation_function)
    , activation_function_derivative(activation_function_derivative)
    , hidden_weights(num_inputs, num_hidden)
    , output_weights(num_hidden, num_outputs)
{
    hidden_thresholds.resize(num_hidden);
    output_thresholds.resize(num_outputs);
    hidden_data.resize(num_hidden);
    output_data.resize(num_outputs);
}

const Matrix<double>&
NeuralNetwork::get_output_weights() const
{
    return output_weights;
}

const std::vector<double>&
NeuralNetwork::get_output_thresholds() const
{
    return output_thresholds;
}

const Matrix<double>&
NeuralNetwork::get_hidden_weights() const
{
    return hidden_weights;
}

const std::vector<double>&
NeuralNetwork::get_hidden_thresholds() const
{
    return hidden_thresholds;
}

void
NeuralNetwork::reset_weights(double min, double max)
{
    for (size_t i = 0; i < hidden_weights.num_elements; ++i) {
        hidden_weights.set_linear(i, random_double_in_range(min, max));
    }
    for (size_t i = 0; i < output_weights.num_elements; ++i) {
        output_weights.set_linear(i, random_double_in_range(min, max));
    }
}

void
NeuralNetwork::reset_thresholds(double min, double max)
{
    for (size_t i = 0; i < hidden_thresholds.size(); ++i) {
        hidden_thresholds[i] = random_double_in_range(min, max);
    }
    for (size_t i = 0; i < output_thresholds.size(); ++i) {
        output_thresholds[i] = random_double_in_range(min, max);
    }
}

double
NeuralNetwork::calculate_b_i(size_t i) const
{
    double b_i = 0.0;

    // Apply weighted sum
    for (size_t j = 0; j < num_hidden; ++j) {
        b_i += output_weights.get(i, j) * hidden_data[j];
    }

    // Apply threshold
    b_i -= output_thresholds[i];

    return b_i;
}

double
NeuralNetwork::calculate_b_j(size_t j, const std::vector<double> &input_data) const
{
    assert(input_data.size() == num_inputs);

    double b_j = 0.0;

    // Apply weighted sum
    for (size_t k = 0; k < num_inputs; ++k) {
        b_j += hidden_weights.get(j, k) * input_data[k];
    }

    // Apply threshold
    b_j -= hidden_thresholds[j];

    return b_j;
}

void
NeuralNetwork::train(const std::initializer_list<double>& training_input, const std::initializer_list<double>& expected_output, double learning_rate)
{
    assert(training_input.size() == num_inputs);

    // Converting an initializer list (which has no subscript operator) to a c-array
    const auto& expected_out = expected_output.begin();
    const auto& training_in = training_input.begin();

    //
    // Perform backpropagation using gradient descent
    //

    // First run the network to get actual output (stored in output)
    run(training_input);

    // Memoize d_i for all i:s
    static std::vector<double> error_output(num_outputs);

    // Output layer propagation
    static Matrix<double> delta_output_weights(num_hidden, num_outputs);
    static std::vector<double> delta_output_thresholds(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {

        double b_i = calculate_b_i(i);
        double error_i = (expected_out[i] - output_data[i]) * activation_function_derivative(b_i);
        error_output[i] = error_i;

        // Calculate delta output weights
        for (size_t j = 0; j < num_hidden; ++j) {
            double delta_weight = learning_rate * error_i * hidden_data[j];
            delta_output_weights.set(i, j, delta_weight);
        }

        // Calculate delta output thresholds
        double delta_threshold = -learning_rate * error_i;
        delta_output_thresholds[i] = delta_threshold;
    }

    // Hidden layer propagation
    static Matrix<double> delta_hidden_weights(num_inputs, num_hidden);
    static std::vector<double> delta_hidden_thresholds(num_hidden);
    for (size_t j = 0; j < num_hidden; ++j) {

        double b_j = calculate_b_j(j, training_input);

        // Calculate error
        double error_j = 0.0;
        for (size_t i = 0; i < num_outputs; ++i) {
            double error_term = error_output[i] * output_weights.get(i, j) * activation_function_derivative(b_j);
            error_j += error_term;
        }

        // Calculate delta hidden weights
        for (size_t k = 0; k < num_inputs; ++k) {
            double delta_weight = learning_rate * error_j * training_in[k];
            delta_hidden_weights.set(j, k, delta_weight);
        }

        // Calculate delta hidden thresholds
        double delta_threshold = 0.0;
        for (size_t i = 0; i < num_outputs; ++i) {
            delta_threshold += error_output[i] * output_weights.get(i, j) * -activation_function_derivative(b_j);
        }
        delta_threshold = learning_rate * delta_threshold;
        delta_hidden_thresholds[j] = delta_threshold;
    }

    // Apply deltas
    for (size_t i = 0; i < num_outputs; ++i) {
        for (size_t j = 0; j < num_hidden; ++j) {
            double new_w_ij = output_weights.get(i, j) + delta_output_weights.get(i, j);
            output_weights.set(i, j, new_w_ij);
        }
        output_thresholds[i] += delta_output_thresholds[i];
    }
    for (size_t j = 0; j < num_hidden; ++j) {
        for (size_t k = 0; k < num_inputs; ++k) {
            double new_w_jk = hidden_weights.get(j, k) + delta_hidden_weights.get(j, k);
            hidden_weights.set(j, k, new_w_jk);
        }
        hidden_thresholds[j] += delta_hidden_thresholds[j];
    }
}

const std::vector<double>&
NeuralNetwork::run(std::vector<double> input_data) const
{
    assert(input_data.size() == num_inputs);

    for (size_t j = 0; j < num_hidden; ++j) {
        double b_j = calculate_b_j(j, input_data);
        hidden_data[j] = activation_function(b_j);
    }

    for (size_t i = 0; i < num_outputs; ++i) {
        double b_i = calculate_b_i(i);
        output_data[i] = activation_function(b_i);
    }

    return output_data;
}
