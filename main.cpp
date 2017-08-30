#include <iostream>

#include "pattern.h"

float calculate_weight(const Pattern &pattern, size_t i, size_t j)
{
    if (i == j) return 0.0f;
    return (1.0f / pattern.num_neurons) * pattern.get_linear(i) * pattern.get_linear(j);
}

void create_weight_matrix(std::vector<float> &weights, const Pattern &pattern)
{
    size_t weight_matrix_side = pattern.num_neurons;
    size_t weight_matrix_size = weight_matrix_side * weight_matrix_side;

    // Resize to fit a NxN weight matrix
    weights.resize(weight_matrix_size);

    for (size_t i = 0; i < weight_matrix_side; ++i)  {
        for (size_t j = 0; j < weight_matrix_side; ++j)  {
            weights[j + i * weight_matrix_side] = calculate_weight(pattern, i, j);
        }
    }
}

int sgn(float x) {
    return (0 < x) - (x < 0);
}

void update_neuron(Pattern& pattern, const std::vector<float>& weights, size_t i)
{
    float h = 0.0f;
    for (size_t j = 0; j < pattern.num_neurons; ++j) {
        float Wij = weights[j + i * pattern.num_neurons]; // TODO: make matrix class or similar for weights to keep track of size etc.
        h += Wij * pattern.get_linear(j);
    }
    int new_value = sgn(h);

    pattern.set_linear(i, new_value);
}

int main()
{
    std::srand(static_cast<uint>(std::time(0)));

    std::vector<float> weights;

    Pattern pattern = Pattern(6, 9, {
            -1, -1, -1, -1, -1, -1,
            -1, +1, +1, +1, +1, -1,
            -1, +1, -1, -1, -1, -1,
            -1, +1, -1, -1, -1, -1,
            -1, +1, +1, +1, -1, -1,
            -1, +1, -1, -1, -1, -1,
            -1, +1, -1, -1, -1, -1,
            -1, +1, +1, +1, +1, -1,
            -1, -1, -1, -1, -1, -1
    });
    Pattern current_state = Pattern(6, 9, {
            +1, -1, +1, -1, -1, -1,
            -1, +1, -1, +1, +1, -1,
            -1, +1, -1, -1, +1, -1,
            +1, +1, -1, -1, -1, -1,
            +1, -1, +1, +1, -1, -1,
            -1, +1, -1, -1, -1, +1,
            -1, +1, -1, -1, +1, +1,
            -1, +1, +1, +1, -1, -1,
            -1, -1, -1, -1, -1, -1
    });

    // Assign weights according to the ... formula
    create_weight_matrix(weights, pattern);

    std::cout << "Goal pattern:" << std::endl;
    pattern.debug_print();

    std::cout << "Initial state:" << std::endl;
    current_state.debug_print();

    if (pattern == current_state) {
        std::cout << "The initial state is already equal to the pattern." << std::endl;
        return 0;
    }

    constexpr int MAX_NUM_ITERATIONS = 10000;
    for (int iteration = 1; iteration <= MAX_NUM_ITERATIONS; iteration++) {

        // Pick random neuron to update
        size_t i = std::rand() % pattern.num_neurons;
        update_neuron(current_state, weights, i);

        // Display pattern
        std::cout << "Iteration " << iteration << ":" << std::endl;
        current_state.debug_print();

        if (pattern == current_state) {
            std::cout << "Converged to pattern after " << iteration << " iterations." << std::endl;
            break;
        }
    }

    return 0;
}

