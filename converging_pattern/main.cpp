#include <iostream>

#include "matrix.h"

float
calculate_weight(const Pattern& pattern, size_t i, size_t j)
{
    if (i == j) return 0.0f;
    return (1.0f / pattern.num_elements) * pattern.get_linear(i) * pattern.get_linear(j);
}

WeightMatrix
create_weight_matrix(const Pattern& pattern)
{
    size_t weight_matrix_side = pattern.num_elements;
    WeightMatrix weights(weight_matrix_side, weight_matrix_side, {});

    for (size_t i = 0; i < weight_matrix_side; ++i)  {
        for (size_t j = 0; j < weight_matrix_side; ++j)  {
            weights.set(i, j, calculate_weight(pattern, i, j));
        }
    }

    return weights;
}

int
sgn(float x) {
    return (0 < x) - (x < 0);
}

void
update_neuron(Pattern& pattern, const WeightMatrix& weights, size_t i)
{
    float h = 0.0f;
    for (size_t j = 0; j < pattern.num_elements; ++j) {
        h += weights.get(i, j) * pattern.get_linear(j);
    }
    int new_value = sgn(h);

    pattern.set_linear(i, new_value);
}

int
main()
{
    std::srand(static_cast<uint>(std::time(0)));

    Pattern pattern(6, 9, {
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
    Pattern current_state(6, 9, {
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
    WeightMatrix weights = create_weight_matrix(pattern);

    std::cout << "Goal pattern:" << std::endl;
    pattern.debug_print();

    std::cout << "Initial state:" << std::endl;
    current_state.debug_print();

/*
    // NOTE: CLion *incorrectly* states that the == is always true... Not sure how to fix that.
    if (pattern == current_state) {
        std::cout << "The initial state is already equal to the pattern." << std::endl;
        return 0;
    }
*/

    constexpr int MAX_NUM_ITERATIONS = 10000;
    for (int iteration = 1; iteration <= MAX_NUM_ITERATIONS; iteration++) {

        // Pick random neuron to update
        size_t i = std::rand() % pattern.num_elements;
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

