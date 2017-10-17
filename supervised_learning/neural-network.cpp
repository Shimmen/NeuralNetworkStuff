#include "neural-network.h"

NeuralNetwork::NeuralNetwork()
    : num_inputs(0)
    , num_outputs(0)
    , output_weights(0, 0)
{
    // Empty default constructor
}

NeuralNetwork::NeuralNetwork(size_t num_inputs, size_t num_outputs, ActivationFunction activation_function, ActivationFunction activation_function_derivative)
    : num_inputs(num_inputs)
    , num_outputs(num_outputs)
    , activation_function(activation_function)
    , activation_function_derivative(activation_function_derivative)
    , output_weights(num_inputs, num_outputs)
{
    output_thresholds.resize(num_outputs);
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

void
NeuralNetwork::reset_weights(double min, double max)
{
    for (size_t i = 0; i < output_weights.num_elements; ++i) {
        output_weights.set_linear(i, random_double_in_range(min, max));
    }
}

void
NeuralNetwork::reset_thresholds(double min, double max)
{
    for (size_t i = 0; i < output_thresholds.size(); ++i) {
        output_thresholds[i] = random_double_in_range(min, max);
    }
}

double
NeuralNetwork::calculate_b_i(size_t i, const std::vector<double> &input_data) const
{
    double local_field = 0.0;

    // Apply weighted sum
    for (size_t j = 0; j < num_inputs; ++j) {
        local_field += output_weights.get(i, j) * input_data[j];
    }

    // Apply threshold
    local_field -= output_thresholds[i];

    return local_field;
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

    static Matrix<double> delta_weights(num_inputs, num_outputs);
    static std::vector<double> delta_thresholds(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {

        double local_field = calculate_b_i(i, training_input);
        double error = (expected_out[i] - output_data[i]) * activation_function_derivative(local_field);

        // Calculate delta weights
        for (size_t j = 0; j < num_inputs; ++j) {
            double delta_weight = learning_rate * error * training_in[j];
            delta_weights.set(i, j, delta_weight);
        }

        // Calculate delta thresholds
        double delta_threshold = -learning_rate * error;
        delta_thresholds[i] = delta_threshold;
    }

    // Apply deltas
    for (size_t i = 0; i < num_outputs; ++i) {
        for (size_t j = 0; j < num_inputs; ++j) {
            double new_wij = output_weights.get(i, j) + delta_weights.get(i, j);
            output_weights.set(i, j, new_wij);
        }
        output_thresholds[i] += delta_thresholds[i];
    }
}

const std::vector<double>&
NeuralNetwork::run(std::vector<double> input_data) const
{
    assert(input_data.size() == num_inputs);
    for (size_t i = 0; i < num_outputs; ++i) {

        double local_field = calculate_b_i(i, input_data);
        output_data[i] = activation_function(local_field);
    }

    return output_data;
}
