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
// McCulloch-Pitts dynamic / update rule

#define sgn(x) math_signum(x)

int
math_signum(float x) {
    return (0 < x) - (x < 0);
}

int mcculloch_pitts_step(Pattern& pattern, const WeightMatrix& weights, size_t i)
{
    // Sum
    float h = 0.0f;
    for (size_t j = 0; j < pattern.num_elements; ++j) {
        h += weights.get(i, j) * pattern.get_linear(j);
    }

    // Limit
    int new_value = sgn(h);

    return new_value;
}

bool
would_update_neuron(Pattern& pattern, const WeightMatrix& weights, size_t i)
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
theoretical_p_error(double p_div_n)
{
    double n_div_p = 1.0 / p_div_n;
    return (1.0 / 2.0) * (1.0 - std::erf(sqrt(1.0 / 2.0 * n_div_p)));
}

std::vector<double>
theoretical_p_error(std::vector<double> p_div_n_vector)
{
    std::vector<double> result;
    result.reserve(p_div_n_vector.size());
    for (double& p_div_n : p_div_n_vector) {
        result.push_back(theoretical_p_error(p_div_n));
    }
    return result;
}

///////////////////////////////////////////////
// Test procedure

int
main()
{
    std::srand(static_cast<uint>(std::time(0)));

    const size_t N = 200;
    const int BITS_TO_TEST_PER_P = 10000;
    const std::vector<size_t> p_vector = {1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400};

    // To store test results in
    std::vector<double> p_div_n_vector;
    std::vector<double> p_error_vector;

    std::cout << "N = " << N << std::endl;

    for (auto p : p_vector)  {

        std::cout << "Testing for p = " << p << ":" << std::endl;

        // Create p random patterns to store
        std::vector<Pattern> stored_patterns;
        stored_patterns.reserve(p);
        for (size_t i = 0; i < p; ++i) {
            stored_patterns.emplace_back(20, 10); // since 20 * 10 = 200 = N (could just as well be 1D)
            fill_with_random_noise(stored_patterns.at(i));
        }

        // Store patterns in the weight matrix according to Hebb's rule
        const WeightMatrix &weights = create_weight_matrix(stored_patterns, N);

        // Choose one of the patterns to test on. Since they are all random we can just as well choose the first.
        Pattern test_pattern = stored_patterns.at(0);

        int num_incorrectly_flipped_bits = 0;
        for (size_t current_test = 0; current_test < BITS_TO_TEST_PER_P; ++current_test) {

            // Pick random neuron to "update"
            size_t i = std::rand() % N;

            // Neuron should not be updated since we start with a stored pattern
            if (would_update_neuron(test_pattern, weights, i)) {
                num_incorrectly_flipped_bits += 1;
            }
        }

        double p_div_n = static_cast<float>(p) / static_cast<float>(N);
        double p_error = static_cast<double>(num_incorrectly_flipped_bits) / static_cast<double>(BITS_TO_TEST_PER_P);
        std::cout << "  p/N = " << p_div_n << " -> P[Error] = " << p_error << std::endl;

        p_div_n_vector.push_back(p_div_n);
        p_error_vector.push_back(p_error);
    }

    // Draw and show graph results
    plt::figure();
    plt::title("P[Error] in terms of p/N");

    plt::named_plot("Empirical results", p_div_n_vector, p_error_vector, "r-");
    plt::named_plot("Analytical estimation", p_div_n_vector, theoretical_p_error(p_div_n_vector), "g-");

    plt::xlabel("p/N");
    plt::ylabel("P[Error]");
    plt::ylim(0.0, 1.0);

    plt::grid(true);
    plt::legend();
    plt::show();

    return 0;
}

