#include "kohonen-network.h"

#include "random.h"

KohonenNetwork::KohonenNetwork(size_t num_inputs, size_t num_outputs, NeighbouringFunction neighbouring_function)
    : num_inputs(num_inputs)
    , num_outputs(num_outputs)
    , weights(num_inputs, num_outputs)
    , neighbouring_function(neighbouring_function)
{
}

const
Matrix<double>& KohonenNetwork::get_weights() const
{
    return weights;
};

void
KohonenNetwork::reset_weights(double min, double max)
{
    for (size_t i = 0; i < weights.num_elements; ++i) {
        weights.set_linear(i, random_double_in_range(min, max));
    }
}

void
KohonenNetwork::train(const std::initializer_list<double> &training_input, double learning_rate, double gaussian_width)
{
    // Training data must match network
    assert(training_input.size() == weights.width);

    // Converting an initializer list (which has no subscript operator) to a c-array
    const auto& input = training_input.begin();

    Matrix<double> delta_weights(num_inputs, num_outputs);
    size_t winning_neuron_index = calculate_winning_neuron(training_input);

    // Calculate deltas
    for (size_t i = 0; i < weights.height; ++i) {

        double neighbour_factor = neighbouring_function(i, winning_neuron_index, gaussian_width);

        for (size_t j = 0; j < weights.width; ++j) {
            double delta = learning_rate * neighbour_factor * (input[j] - weights.get(i, j));
            delta_weights.set(i, j, delta);
        }
    }

    // Apply deltas
    for (size_t i = 0; i < weights.num_elements; ++i) {
        double new_weight = weights.get_linear(i) + delta_weights.get_linear(i);
        weights.set_linear(i, new_weight);
    }

}

size_t
KohonenNetwork::calculate_winning_neuron(const std::initializer_list<double> &training_input) const
{
    // Converting an initializer list (which has no subscript operator) to a c-array
    const auto& input = training_input.begin();

    double closest_distance = std::numeric_limits<double>::infinity();
    size_t index_of_closest = 0;

    for (size_t i = 0; i < weights.height; ++i) {

        double relative_distance = 0.0;
        for (size_t j = 0; j < weights.width; ++j) {
            double dist = input[j] - weights.get(i, j);
            relative_distance += dist * dist;
        }

        if (relative_distance < closest_distance) {
            closest_distance = relative_distance;
            index_of_closest = i;
        }

    }

    return index_of_closest;
}
