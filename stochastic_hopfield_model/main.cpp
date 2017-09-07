#include <iostream>
#include <cmath>

#include <matrix.h>

#define WITHOUT_NUMPY
#include <matplotlib-cpp/matplotlibcpp.h>
namespace plt = matplotlibcpp;

///////////////////////////////////////////////
// Pattern initialization

int
random_plus_minus_one()
{
    return (std::rand() % 2) * 2 - 1;
}

void
fill_with_random_noise(Pattern& pattern)
{
    for (size_t i = 0; i < pattern.num_elements; ++i) {
        pattern.set_linear(i, random_plus_minus_one());
    }
}

///////////////////////////////////////////////
// Hebb's rule related

float
calculate_weight_at_ij(const std::vector<Pattern>& patterns, size_t N, size_t i, size_t j)
{
    if (i == j) return 0.0f;

    float weight = 0.0f;
    for (size_t idx = 0; idx < patterns.size(); ++idx) {
        auto& pattern = patterns.at(idx);
        weight += pattern.get_linear(i) * pattern.get_linear(j);
    }
    return weight / static_cast<float>(N);
}

WeightMatrix
create_weight_matrix(const std::vector<Pattern>& patterns, size_t num_bits)
{
    size_t N = num_bits;
    WeightMatrix weights(N, N);

    for (size_t i = 0; i < N; ++i)  {
        for (size_t j = 0; j < N; ++j)  {
            float w = calculate_weight_at_ij(patterns, N, i, j);
            weights.set(i, j, w);
        }
    }

    return weights;
}

///////////////////////////////////////////////
// Stochastic update rule

float
local_field(Pattern& pattern, size_t i, const WeightMatrix& weights)
{
    float b = 0.0f;
    for (size_t j = 0; j < pattern.num_elements; ++j) {
        b += weights.get(i, j) * pattern.get_linear(j);
    }
    return b;
}

float random_float()
{
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

#define g_func(b, beta) stochastic_signum(b, beta)

float
stochastic_signum(float b, float beta)
{
    return 1.0f / (1.0f + expf(-2.0f * b * beta));
}

void
stochastic_update_neuron(Pattern& pattern, size_t i, const WeightMatrix& weights, float beta)
{
    float b = local_field(pattern, i, weights);
    int new_value = random_float() <= g_func(b, beta) ? +1 : -1;
    pattern.set_linear(i, new_value);
}

///////////////////////////////////////////////
// Order parameter calculation

float
order_parameter_for_iteration(const Pattern &stored_pattern, const Pattern &test_pattern)
{
    assert(stored_pattern.num_elements == test_pattern.num_elements);
    size_t N = stored_pattern.num_elements;

    float m = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        m += test_pattern.get_linear(i) * stored_pattern.get_linear(i);
    }
    m /= static_cast<float>(N);

    return m;
}

///////////////////////////////////////////////
// Test procedure

int
main()
{
    std::srand(static_cast<uint>(std::time(0)));

    const size_t NUM_TESTS_TO_PERFORM = 20;
    const size_t NUM_ASYNC_STEPS_TO_PERFORM_PER_TEST = 10000;

    const size_t N = 200;
    const size_t p = 5;
    const float beta = 2;

    std::cout << "Testing for p = " << p << ", N = " << N << ", and beta = " << beta << ":" << std::endl;

    // Create p random patterns to store
    std::vector<Pattern> stored_patterns;
    stored_patterns.reserve(p);
    for (size_t i = 0; i < p; ++i) {
        stored_patterns.emplace_back(N);
        fill_with_random_noise(stored_patterns.at(i));
    }

    // Store patterns in the weight matrix according to Hebb's rule
    const WeightMatrix& weights = create_weight_matrix(stored_patterns, N);

    // Feed the first stored pattern to the network
    Pattern test_pattern = stored_patterns.at(0);

    // For storing data for plotting
    std::vector<float> iteration_step_vector;
    std::vector<float> order_parameter_for_iteration_vector;
    iteration_step_vector.resize(NUM_ASYNC_STEPS_TO_PERFORM_PER_TEST);
    order_parameter_for_iteration_vector.resize(NUM_ASYNC_STEPS_TO_PERFORM_PER_TEST);

    plt::figure();
    plt::title("Phase diagram for p = 5");

    for (size_t current_test = 0; current_test < NUM_TESTS_TO_PERFORM; ++current_test) {

        std::cout << "Performing test " << (current_test + 1) << " out of " << NUM_TESTS_TO_PERFORM << std::endl;

        for (size_t current_step = 0; current_step < NUM_ASYNC_STEPS_TO_PERFORM_PER_TEST; ++current_step) {

            // Pick random neuron to update
            size_t i = std::rand() % N;

            stochastic_update_neuron(test_pattern, i, weights, beta);

            float m = order_parameter_for_iteration(stored_patterns.at(0), test_pattern);

            // Store data for plotting
            iteration_step_vector.at(current_step) = current_step;
            order_parameter_for_iteration_vector.at(current_step) = m;
        }

        plt::plot(iteration_step_vector, order_parameter_for_iteration_vector, "r-");
    }

    std::cout << "Tests done, plotting" << std::endl;

    // Draw and show graph results
    plt::xlabel("t");
    plt::ylabel("m(t)");
    plt::ylim(0.0, 1.1);
    plt::grid(true);
    plt::show();

    return 0;
}

