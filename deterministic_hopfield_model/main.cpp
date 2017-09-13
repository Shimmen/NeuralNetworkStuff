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

double
calculate_weight_at_ij(const std::vector<Pattern>& patterns, size_t N, size_t i, size_t j)
{
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
// McCulloch-Pitts dynamic / update rule

#define sgn(x) math_signum(x)

int
math_signum(double x) {
    // TODO: Why do I get zero all the time now?!
    //assert(x != 0); // (very unlikely)
    int val = (x >= 0) - (x < 0);
    return val;
}

int mcculloch_pitts_step(const Pattern& pattern, const WeightMatrix& weights, size_t i)
{
    // Sum
    double h = 0.0f;
    for (size_t j = 0; j < pattern.num_elements; ++j) {
        h += weights.get(i, j) * pattern.get_linear(j);
    }

    // Limit
    int new_value = sgn(h);

    return new_value;
}

bool
would_update_neuron(const Pattern& pattern, const WeightMatrix& weights, size_t i)
{
    // Return true if the pattern would be updated
    int new_value = mcculloch_pitts_step(pattern, weights, i);
    int old_value = pattern.get_linear(i);
    return (old_value != new_value);
}

void
update_neuron(Pattern& pattern, const WeightMatrix& weights, size_t i)
{
    int new_value = mcculloch_pitts_step(pattern, weights, i);
    pattern.set_linear(i, new_value);
}

///////////////////////////////////////////////
// Theoretical one-step error probability

double
cumulative_erf_from_x_to_inf(double x, double mean, double variance)
{
    static constexpr double ONE_HALF = 1.0 / 2.0;

    return ONE_HALF * (x - std::erf(
            (x - mean) / (sqrt(variance * 2))
    ));
}

std::vector<double>
theoretical_p_error(std::vector<double> p_div_n_vector, double N)
{
    std::vector<double> result;
    result.reserve(p_div_n_vector.size());
    for (double& p_div_n : p_div_n_vector) {

        // (A slightly hacky way to get the current p without changing too much code around)
        double p = p_div_n * N;
        double mean_for_p = - (p - 1) / N;
        double expected_error = cumulative_erf_from_x_to_inf(1.0, mean_for_p, p_div_n);
        result.push_back(expected_error);

    }
    return result;
}

///////////////////////////////////////////////
// Test procedure

struct TestSet
{
    WeightMatrix weight_matrix;
    std::vector<Pattern> stored_patterns;
    TestSet() : weight_matrix(0), stored_patterns() {}
};

int
main()
{
    std::srand(static_cast<uint>(std::time(0)));

    const size_t N = 200;
    const size_t BITS_TO_TEST_PER_P = 100000;
    const std::vector<size_t> p_vector = {1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400};

    // To store test results in
    std::vector<double> p_div_n_vector;
    std::vector<double> p_error_vector;
    p_div_n_vector.reserve(p_vector.size());
    p_error_vector.reserve(p_vector.size());

    std::cout << "N = " << N << std::endl;

    for (auto p : p_vector)  {

        std::cout << "Testing for p = " << p << ":" << std::endl;

        // Create all test sets for this p
        size_t num_tests_per_set = p * N;
        size_t num_test_sets = static_cast<size_t>(ceil(BITS_TO_TEST_PER_P / static_cast<double>(num_tests_per_set)));
        std::vector<TestSet> test_sets;
        test_sets.resize(num_test_sets);
        for (size_t i = 0; i < num_test_sets; ++i) {

            TestSet *current = &test_sets.at(i);

            current->stored_patterns.reserve(p);
            for (size_t j = 0; j < p; ++j) {
                current->stored_patterns.emplace_back(N);
                fill_with_random_noise(current->stored_patterns.at(j));
            }

            current->weight_matrix = create_weight_matrix(current->stored_patterns, N);

        }

        int num_incorrectly_flipped_bits = 0;
        for (size_t current_test = 0; current_test < BITS_TO_TEST_PER_P; ++current_test) {

            // Pick neuron to "update". Since the neurons aren't actually updated this is effectively
            // implementing a synchronous update
            size_t i = current_test % N;

            // Pick pattern to use as current state
            size_t current_p = (current_test / N) % p;

            // Pick test set to use
            size_t current_test_set_index = current_test / (int)num_tests_per_set;
            const TestSet& current_test_set = test_sets.at(current_test_set_index);

            const Pattern& current_state = current_test_set.stored_patterns.at(current_p);

            // Neuron should not be updated since we start with a stored pattern
            if (would_update_neuron(current_state, current_test_set.weight_matrix, i)) {
                num_incorrectly_flipped_bits += 1;
            }
        }

        double p_div_n = static_cast<double>(p) / static_cast<double>(N);
        double p_error = static_cast<double>(num_incorrectly_flipped_bits) / static_cast<double>(BITS_TO_TEST_PER_P);
        std::cout << "  p/N = " << p_div_n << " -> P Error = " << p_error << std::endl;

        p_div_n_vector.push_back(p_div_n);
        p_error_vector.push_back(p_error);
    }

    std::cout << "Done, plotting..." << std::endl;

    // Draw and show graph results
    plt::figure();

    const auto& theoretical_data = theoretical_p_error(p_div_n_vector, N);

    plt::subplot(1, 2, 1);
    {
        plt::title("P Error as a function of p/N");
        plt::named_plot("Theoretical estimation", p_div_n_vector, theoretical_data, "g-");
        plt::named_plot("Empirical results", p_div_n_vector, p_error_vector, "ro");
        plt::xlabel("p/N");
        plt::ylabel("P Error");
        plt::ylim(0.0, 1.0);//plt::ylim(-0.01, 1.0);
        plt::xlim(0.0, 2.0);//plt::xlim(-0.02, 2.02);
        plt::grid(true);
        plt::legend();
    }

    plt::subplot(1, 2, 2);
    {
        plt::title("P Error as a function of p/N (zoomed in)");
        plt::named_plot("Theoretical estimation", p_div_n_vector, theoretical_data, "g-");
        plt::named_plot("Empirical results", p_div_n_vector, p_error_vector, "ro");
        plt::xlabel("p/N");
        plt::ylabel("P Error");
        plt::ylim(0.0, 0.045);//plt::ylim(-0.001, 0.045);
        plt::xlim(0.0, 2.0);//plt::xlim(-0.02, 2.02);
        plt::grid(true);
        plt::legend();
    }

    plt::show();

    return 0;
}

