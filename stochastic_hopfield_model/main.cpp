#include <iostream>
#include <sstream>
#include <cmath>

#include <matrix.h>

#define WITHOUT_NUMPY
#include <matplotlib-cpp/matplotlibcpp.h>
namespace plt = matplotlibcpp;

typedef Matrix<int> Pattern;
typedef Matrix<double> WeightMatrix;

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

double
calculate_weight_at_ij(const std::vector<Pattern>& patterns, size_t N, size_t i, size_t j)
{
    if (i == j) return 0.0f;

    double weight = 0.0f;
    for (size_t idx = 0; idx < patterns.size(); ++idx) {
        auto& pattern = patterns.at(idx);
        weight += pattern.get_linear(i) * pattern.get_linear(j);
    }
    return weight / static_cast<double>(N);
}

WeightMatrix
create_weight_matrix(const std::vector<Pattern>& patterns, size_t num_bits)
{
    size_t N = num_bits;
    WeightMatrix weights(N, N);

    for (size_t i = 0; i < N; ++i)  {
        for (size_t j = 0; j < N; ++j)  {
            double w = calculate_weight_at_ij(patterns, N, i, j);
            weights.set(i, j, w);
        }
    }

    return weights;
}

///////////////////////////////////////////////
// Stochastic update rule

double
local_field(Pattern& pattern, size_t i, const WeightMatrix& weights)
{
    double b = 0.0f;
    for (size_t j = 0; j < pattern.num_elements; ++j) {
        b += weights.get(i, j) * pattern.get_linear(j);
    }
    return b;
}

double random_double()
{
    return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}

double
activation_function(double b, double beta)
{
    return 1.0f / (1.0f + exp(-2.0f * b * beta));
}

void
stochastic_update_neuron(Pattern& pattern, size_t i, const WeightMatrix& weights, double beta)
{
    double b = local_field(pattern, i, weights);
    int new_value = random_double() <= activation_function(b, beta) ? +1 : -1;
    pattern.set_linear(i, new_value);
}

///////////////////////////////////////////////
// Order parameter calculation

double
order_parameter_for_iteration(const Pattern &stored_pattern, const Pattern &test_pattern)
{
    assert(stored_pattern.num_elements == test_pattern.num_elements);
    size_t N = stored_pattern.num_elements;

    double m = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        m += test_pattern.get_linear(i) * stored_pattern.get_linear(i);
    }
    m /= static_cast<double>(N);

    return m;
}

///////////////////////////////////////////////
// Test procedure

int
main()
{
    std::srand(static_cast<uint>(std::time(0)));

    const size_t NUM_TESTS_TO_PERFORM = 20;
    const size_t NUM_ASYNC_STEPS_TO_PERFORM_PER_TEST = 25000;
    const int SMOOTHING_KERNEL_SIZE = 1000;

    const size_t N = 200;
    const size_t p_list[] = { 5, 40 };
    const double beta = 2;

    for (int current_p_index = 0; current_p_index < 2; ++current_p_index) {

        size_t p = p_list[current_p_index];

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

        // For storing data for plotting
        std::vector<double> iteration_step_vector;
        std::vector<double> order_parameter_for_iteration_vector;
        iteration_step_vector.resize(NUM_ASYNC_STEPS_TO_PERFORM_PER_TEST);
        order_parameter_for_iteration_vector.resize(NUM_ASYNC_STEPS_TO_PERFORM_PER_TEST);

        for (size_t current_test = 0; current_test < NUM_TESTS_TO_PERFORM; ++current_test) {

            std::cout << " test " << (current_test + 1) << "/" << NUM_TESTS_TO_PERFORM << std::endl;

            // Feed the first stored pattern to the network
            Pattern test_pattern = stored_patterns.at(0);

            for (size_t current_step = 0; current_step < NUM_ASYNC_STEPS_TO_PERFORM_PER_TEST; ++current_step) {

                // Pick random neuron to update
                size_t i = std::rand() % N;

                stochastic_update_neuron(test_pattern, i, weights, beta);

                double m = order_parameter_for_iteration(stored_patterns.at(0), test_pattern);

                // Store data for plotting
                iteration_step_vector.at(current_step) = current_step;
                order_parameter_for_iteration_vector.at(current_step) = m;
            }

            // Apply moving average filter
            size_t num_elements = order_parameter_for_iteration_vector.size();
            std::vector<double> smooth_order_parameter_for_iteration_vector(num_elements);
            for (int i = 0; i < num_elements; ++i) {

                double sum = 0.0;
                int count = 0;

                for (int j = -SMOOTHING_KERNEL_SIZE / 2; j < SMOOTHING_KERNEL_SIZE / 2; ++j) {
                    int idx = i + j;
                    if (idx < 0) idx = 0;
                    if (idx >= num_elements) idx = (int) num_elements - 1;
                    size_t index = static_cast<size_t>(idx);
                    sum += order_parameter_for_iteration_vector.at(index);
                    count += 1;
                }

                double average = sum / count;
                smooth_order_parameter_for_iteration_vector.at((size_t)i) = average;
            }

            // Plot results
            {
                int row = 2 * current_p_index;
                std::stringstream title;
                title << "Phase diagram for p = " << p;

                plt::subplot(2, 2, row + 1);
                plt::title(title.str());
                plt::plot(iteration_step_vector, order_parameter_for_iteration_vector, "r-");
                plt::xlabel("t");
                plt::ylabel("m(t)");
                plt::ylim(0.0, 1.1);
                plt::grid(true);

                plt::subplot(2, 2, row + 2);
                title << " (smoothed)";
                plt::title(title.str());
                plt::plot(iteration_step_vector, smooth_order_parameter_for_iteration_vector, "b-");
                plt::xlabel("t");
                plt::ylabel("m(t)");
                plt::ylim(0.0, 1.1);
                plt::grid(true);
            }
        }
    }

    std::cout << "Tests done, plotting" << std::endl;
    plt::show();

    return 0;
}

