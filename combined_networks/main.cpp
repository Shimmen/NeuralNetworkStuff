#include <iostream>
#include <fstream>
#include <random>

#include <matrix.h>

#include "neural-network.h"

#define WITHOUT_NUMPY
#include <matplotlib-cpp/matplotlibcpp.h>

namespace plt = matplotlibcpp;

///////////////////////////////////////////////
// Input data

struct Pattern
{
    double x;
    double y;
    double out;
    Pattern(double x, double y, double out) : x(x), y(y), out(out) {}
};

std::vector<Pattern>
read_and_parse_data(const std::string& file_name)
{
    std::vector<Pattern> patterns;

    std::ifstream ifs(file_name);

    double x1;
    double x2;
    double out;

    while (ifs >> out >> x1 >> x2) {
        patterns.emplace_back(x1, x2, out);
    }

    return patterns;
}

///////////////////////////////////////////////
// Unsupervised network stuff

void
reset_weights(Matrix<double>& weights, std::default_random_engine rng)
{
    std::uniform_real_distribution<double> random_double(-1.0, 1.0);

    for (size_t i = 0; i < weights.num_elements; ++i) {
        weights.set_linear(i, random_double(rng));
    }
}

double
distance_squared(double x1, double y1, double x2, double y2)
{
    double dx = x2 - x1;
    double dy = y2 - y1;
    return (dx * dx) + dy * dy;
}

std::vector<double>
get_activations(const Matrix<double>& weights, const Pattern& pattern)
{
    double denominator = 0.0;
    for (size_t i = 0; i < weights.height; ++i) {
        denominator += std::exp(-distance_squared(
                pattern.x, pattern.y,
                weights.get(i, 0), weights.get(i, 1)
        ) / 2.0);
    }

    std::vector<double> activations(weights.height);

    for (size_t j = 0; j < weights.height; ++j) {

        double numerator = std::exp(-distance_squared(
                pattern.x, pattern.y,
                weights.get(j, 0), weights.get(j, 1)
        ) / 2.0);

        activations[j] = numerator / denominator;
    }

    return activations;
}

size_t
get_winning_neuron_index(const Matrix<double>& weights, const Pattern& pattern)
{
    const auto& activations = get_activations(weights, pattern);

    size_t winning_neuron_index = 0;
    double maximum_activation = 0.0;

    for (size_t j = 0; j < activations.size(); ++j) {
        if (activations[j] > maximum_activation) {
            maximum_activation = activations[j];
            winning_neuron_index = j;
        }
    }

    return winning_neuron_index;
}

void
train_unsupervised(Matrix<double>& weights, const Pattern& pattern, double learning_rate)
{
    size_t winning_neuron_index = get_winning_neuron_index(weights, pattern);

    // Update (only) winning neuron
    double delta_weight_x = learning_rate * (pattern.x - weights.get(winning_neuron_index, 0));
    double delta_weight_y = learning_rate * (pattern.y - weights.get(winning_neuron_index, 1));

    weights.set(winning_neuron_index, 0, weights.get(winning_neuron_index, 0) + delta_weight_x);
    weights.set(winning_neuron_index, 1, weights.get(winning_neuron_index, 1) + delta_weight_y);
}

///////////////////////////////////////////////
// Supervised network stuff

const double SUPERVISED_ACTIVATION_BETA = 0.5;

double tanh_activation_function(double b)
{
    return tanh(SUPERVISED_ACTIVATION_BETA * b);
}

double tanh_activation_function_derivative(double b)
{
    return SUPERVISED_ACTIVATION_BETA  * (1.0 - pow(tanh_activation_function(b), 2));
}

///////////////////////////////////////////////
// Full network definition etc.

struct FullNetwork
{
    Matrix<double> unsupervised_weights;
    NeuralNetwork supervised_network;
    explicit FullNetwork(size_t K)
        : unsupervised_weights(2, K)
        , supervised_network(K, 1, tanh_activation_function, tanh_activation_function_derivative) {}
};

double
run_network(const FullNetwork& network, const Pattern& pattern)
{
    const auto& activations = get_activations(network.unsupervised_weights, pattern);
    const auto& output = network.supervised_network.run(activations);
    assert(output.size() == 1);
    return output[0];
}

void
plot_network_and_decision_boundary(const FullNetwork& network, const std::vector<Pattern>& training_data)
{
    plt::plot();
    plt::title("Network weights and decision boundary, and input data");

    // Plot training data
    {
        std::vector<double> p_x, p_y, n_x, n_y;
        for (const auto &data: training_data) {
            if (data.out > 0) {
                p_x.push_back(data.x);
                p_y.push_back(data.y);
            } else {
                n_x.push_back(data.x);
                n_y.push_back(data.y);
            }
        }
        plt::plot(p_x, p_y, "ro");
        plt::plot(n_x, n_y, "go");
    }

    // Run network on patterns and draw dots for results
    {
        const size_t NUM_POINTS_X1 = 280;
        const size_t NUM_POINTS_X2 = 280;

        const double MIN_X1 = -15.0;
        const double MIN_X2 = -10.0;
        const double MAX_X1 = +25.0;
        const double MAX_X2 = +15.0;

        const double SCALE_X1 = MAX_X1 - MIN_X1;
        const double SCALE_X2 = MAX_X2 - MIN_X2;

        std::vector<double> p_x, p_y, n_x, n_y;

        for (int x2_step = 0; x2_step < NUM_POINTS_X2; ++x2_step) {
            for (int x1_step = 0; x1_step < NUM_POINTS_X1; ++x1_step) {

                double x1 = static_cast<double>(x1_step) / static_cast<double>(NUM_POINTS_X1) * SCALE_X1 + MIN_X1;
                double x2 = static_cast<double>(x2_step) / static_cast<double>(NUM_POINTS_X2) * SCALE_X2 + MIN_X2;

                Pattern pattern(x1, x2, 0 /* not used! */);
                double result = run_network(network, pattern);

                if (result > 0) {
                    p_x.push_back(x1);
                    p_y.push_back(x2);
                } else {
                    n_x.push_back(x1);
                    n_y.push_back(x2);
                }
            }
        }

        plt::plot(p_x, p_y, "rx");
        plt::plot(n_x, n_y, "gx");
    }

    // Plot/illustrate weights of gaussian neurons
    {
        assert(network.unsupervised_weights.width == 2);
        for (size_t i = 0; i < network.unsupervised_weights.height; ++i) {
            double wx1 = network.unsupervised_weights.get(i, 0);
            double wx2 = network.unsupervised_weights.get(i, 1);
            plt::plot({wx1},{ wx2 }, "bo");
        }
    }

    plt::grid(true);
    plt::xlabel("x1");
    plt::ylabel("x2");
    plt::show();
}

double
fsgn(double x)
{
    return (x > 0) - (x < 0);
}

double
classification_error_for_data_set(const FullNetwork& network, const std::vector<Pattern>& data_set)
{
    double error = 0.0;
    for (const auto &pattern: data_set) {

        double actual_output = run_network(network, pattern);
        double expected_output = pattern.out;

        error += std::fabs(expected_output - fsgn(actual_output));
    }

    error = 1.0 / (2.0 * data_set.size()) * error;
    return error;
}

///////////////////////////////////////////////
// Test procedure

double
average_value(const std::vector<double>& list)
{
    double average = 0.0;
    for (const auto& e: list) {
        average += e;
    }
    return average / list.size();
}

void
train_network(FullNetwork& network, const std::vector<Pattern>& training_data, std::default_random_engine& rng, size_t K)
{
    const size_t UNSUPERVISED_TRAINING_STEPS = 100000;
    const double UNSUPERVISED_LEARNING_RATE = 0.02;

    const size_t SUPERVISED_TRAINING_STEPS = 3000;
    const double SUPERVISED_LEARNING_RATE = 0.1;

    std::uniform_int_distribution<size_t> random_training_data_sample_index(0, training_data.size() - 1);

    //FullNetwork network(K);

    //
    // Phase 1: unsupervised learning
    //

    //Matrix<double> unsupervised_weights(2, K);
    reset_weights(network.unsupervised_weights, rng);

    for (size_t step = 0; step < UNSUPERVISED_TRAINING_STEPS; ++step) {

        size_t i = random_training_data_sample_index(rng);
        auto &pattern = training_data[i];

        train_unsupervised(network.unsupervised_weights, pattern, UNSUPERVISED_LEARNING_RATE);
    }

    //
    // Phase 1.5: transform inputs using the trained unsupervised learning network
    //

    Matrix<double> supervised_input(K, training_data.size()); // (every row is a K-long input vector)

    for (size_t row = 0; row < training_data.size(); ++row) {
        const auto& pattern = training_data[row];

        const auto& activations_for_pattern = get_activations(network.unsupervised_weights, pattern);
        for (size_t k = 0; k < K; ++k) {
            supervised_input.set(row, k, activations_for_pattern[k]);
        }
    }

    //
    // Phase 2: supervised learning
    //

    //NeuralNetwork supervised_network(K, 1, tanh_activation_function, tanh_activation_function_derivative);
    network.supervised_network.reset_weights(rng, -1.0, 1.0);
    network.supervised_network.reset_thresholds(rng, -1.0, 1.0);

    for (size_t step = 0; step < SUPERVISED_TRAINING_STEPS; ++step) {

        size_t i = random_training_data_sample_index(rng);

        std::vector<double> pattern(K);
        for (size_t k = 0; k < K; ++k) {
            pattern[k] = supervised_input.get(i, k);
        }

        network.supervised_network.train(pattern, { training_data[i].out }, SUPERVISED_LEARNING_RATE);
    }
}

int
main()
{
    clock_t start_time = std::clock();

    std::default_random_engine rng;
    rng.seed(static_cast<unsigned int>(start_time));

    const auto& training_data = read_and_parse_data("combined_networks/data.txt");

    //
    // Perform test 1a/1b
    //
    {
        std::cout << "Running 3a/3b" << std::endl;
        std::cout << "  Training ..." << std::endl;

        const size_t NUM_TEST_RUNS = 20;
        const size_t K = 4;// K = 10 for 3b!

        std::vector<double> class_errors(NUM_TEST_RUNS);
        double lowest_class_error = std::numeric_limits<double>::infinity();
        std::shared_ptr<FullNetwork> best_network;

        for (size_t run = 0; run < NUM_TEST_RUNS; ++run) {

            auto network = std::make_shared<FullNetwork>(K);
            train_network(*network, training_data, rng, K);

            double error = classification_error_for_data_set(*network, training_data);
            class_errors[run] = error;

            if (error < lowest_class_error) {
                lowest_class_error = error;
                best_network = network;
            }

        }

        std::cout << "  Average classification error: " << average_value(class_errors) << std::endl;

        std::cout << "  Plotting ..." << std::endl;
        plot_network_and_decision_boundary(*best_network, training_data);
    }

    //
    // Perform test 1c
    //
    {
        std::cout << "Running 3c" << std::endl;

        const size_t NUM_TEST_RUNS_PER_K = 20;

        const size_t MIN_K = 1;
        const size_t MAX_K = 10;
        const size_t NUM_KS = MAX_K - MIN_K + 1;

        std::vector<double> class_errors_for_k_key(NUM_KS);
        std::vector<double> class_errors_for_k_val(NUM_KS);

        for (size_t K = MIN_K; K <= MAX_K; ++K) {

            std::cout << "  Training for K=" << K << " ..." << std::endl;

            std::vector<double> class_errors(NUM_TEST_RUNS_PER_K);
            for (size_t run = 0; run < NUM_TEST_RUNS_PER_K; ++run) {

                auto network = std::make_shared<FullNetwork>(K);
                train_network(*network, training_data, rng, K);

                double error = classification_error_for_data_set(*network, training_data);
                class_errors[run] = error;

            }

            double average_class_error = average_value(class_errors);
            class_errors_for_k_key[K - MIN_K] = K;
            class_errors_for_k_val[K - MIN_K] = average_class_error;

            std::cout << "  Average classification error for K=" << K << ": " << average_class_error << std::endl;
        }

        std::cout << "  Plotting ..." << std::endl;

        plt::plot();
        plt::title("Average classification error as a function of K");
        plt::named_plot("Classification error", class_errors_for_k_key, class_errors_for_k_val, "bo-");
        plt::grid(true);
        plt::legend();
        plt::xlabel("K");
        plt::ylabel("Average classification error");
        plt::ylim(0.0, 1.0);
        plt::show();
    }

    clock_t end_time = std::clock();
    double time_elapsed_s = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << time_elapsed_s << " s" << std::endl;

    return 0;
}
