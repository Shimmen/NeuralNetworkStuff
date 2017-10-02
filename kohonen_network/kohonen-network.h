//
// Created by Simon Moos on 2017-10-02.
//

#ifndef NN_KOHONEN_NETWORK_H
#define NN_KOHONEN_NETWORK_H

#include <vector>

#include <matrix.h>

/**
 * Simple Kohonen neural network
 */
class KohonenNetwork
{
public:

    typedef double (*NeighbouringFunction)(size_t i, size_t i0, double gaussian_width);

    KohonenNetwork(size_t num_inputs, size_t num_outputs, NeighbouringFunction neighbouring_function);
    ~KohonenNetwork() = default;

    const Matrix<double>& get_weights() const;
    void reset_weights(double min, double max);

    void train(const std::initializer_list<double>& training_input, double learning_rate, double gaussian_width);

    const size_t num_inputs;
    const size_t num_outputs;

private:

    size_t calculate_winning_neuron(const std::initializer_list<double> &training_input) const;

    Matrix<double> weights;
    NeighbouringFunction neighbouring_function;

};

#endif // NN_KOHONEN_NETWORK_H
