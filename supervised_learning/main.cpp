#include <iostream>
#include <fstream>
#include <cmath>

#include <matrix.h>
#include <random.h>

#include "neural-network.h"

#define WITHOUT_NUMPY
#include <matplotlib-cpp/matplotlibcpp.h>

namespace plt = matplotlibcpp;

///////////////////////////////////////////////
// Data reading, parsing, preprocessing

struct Pattern
{
    double in_x;
    double in_y;
    double out;
    Pattern(double in_x, double in_y, double out)
            : in_x(in_x), in_y(in_y), out(out) {}
};

std::vector<Pattern>
read_and_parse_data(const std::string& file_name)
{
    std::vector<Pattern> patterns;

    std::ifstream ifs(file_name);

    double in_x;
    double in_y;
    double out;

    while (ifs >> in_x >> in_y >> out) {
        patterns.emplace_back(in_x, in_y, out);
    }

    return patterns;
}

void
normalize_input_data(std::vector<Pattern>& data)
{
    double length = data.size();

    // Calculate mean
    double x_sum = 0.0;
    double y_sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        x_sum += data[i].in_x;
        y_sum += data[i].in_y;
    }
    double x_mean = x_sum / length;
    double y_mean = y_sum / length;

    // Calculate standard deviation
    double nom_sum_x = 0.0;
    double nom_sum_y = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        nom_sum_x += pow(data[i].in_x - x_mean, 2);
        nom_sum_y += pow(data[i].in_y - y_mean, 2);
    }
    double x_standard_dev = sqrt(nom_sum_x / (length - 1));
    double y_standard_dev = sqrt(nom_sum_y / (length - 1));

    // Standardize data (https://en.wikipedia.org/wiki/Feature_scaling#Standardization)
    for (size_t i = 0; i < data.size(); ++i) {
        data[i].in_x = (data[i].in_x - x_mean) / x_standard_dev;
        data[i].in_y = (data[i].in_y - y_mean) / y_standard_dev;
    }
}

///////////////////////////////////////////////
// Utility

double
average_value(const std::vector<double>& list)
{
    double average = 0.0;
    for (const auto& e: list) {
        average += e;
    }
    return average / list.size();
}

double
variance_of_data(const std::vector<double>& data, double mean)
{
    double nom_sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        nom_sum += pow(data[i] - mean, 2);
    }
    return nom_sum / (data.size() - 1);
}

///////////////////////////////////////////////
// Measuring etc.

double
fsgn(double x)
{
    // TODO: Count x == 0
    return (x > 0) - (x < 0);
}

double
classification_error_for_data_set(const NeuralNetwork& network, const std::vector<Pattern>& data_set)
{
    double error = 0.0;
    for (size_t mu = 0; mu < data_set.size(); ++mu) {

        const Pattern& pattern = data_set[mu];

        const std::vector<double>& actual_output_vec = network.run({pattern.in_x, pattern.in_y});
        assert(actual_output_vec.size() == 1);
        double actual_output = actual_output_vec[0];

        double expected_output = pattern.out;
        error += fabs(expected_output - fsgn(actual_output));
    }

    error = 1.0 / (2.0 * data_set.size()) * error;
    return error;
}

double
energy_for_data_set(const NeuralNetwork& network, const std::vector<Pattern>& data_set)
{
    double sum = 0.0;
    for (size_t mu = 0; mu < data_set.size(); ++mu) {
        const Pattern& pattern = data_set[mu];

        const std::vector<double>& actual_output_vec = network.run({pattern.in_x, pattern.in_y});
        assert(actual_output_vec.size() == 1);
        double actual_output = actual_output_vec[0];

        sum += pow(pattern.out - actual_output, 2);
    }

    return -1.0 / 2.0 * sum;
}

///////////////////////////////////////////////
// Test procedure

void
plot_input_data(const std::vector<Pattern>& training_data, const std::vector<Pattern>& validation_data)
{
    std::vector<double> neg_x;
    std::vector<double> neg_y;

    std::vector<double> pos_x;
    std::vector<double> pos_y;

    for (size_t i = 0; i < training_data.size(); ++i) {
        const Pattern& pattern = training_data[i];
        if (pattern.out > 0) {
            pos_x.push_back(pattern.in_x);
            pos_y.push_back(pattern.in_y);
        } else {
            neg_x.push_back(pattern.in_x);
            neg_y.push_back(pattern.in_y);
        }
    }

    for (size_t i = 0; i < validation_data.size(); ++i) {
        const Pattern& pattern = validation_data[i];
        if (pattern.out > 0) {
            pos_x.push_back(pattern.in_x);
            pos_y.push_back(pattern.in_y);
        } else {
            neg_x.push_back(pattern.in_x);
            neg_y.push_back(pattern.in_y);
        }
    }

    plt::named_plot("output = +1", pos_x, pos_y, "go");
    plt::named_plot("output = -1", neg_x, neg_y, "rx");
}

void
plot_network_separation_line(const NeuralNetwork& network)
{
    // Assert the data is all right for plotting using two dimensions
    assert(network.num_outputs == 1);
    assert(network.num_inputs == 2);

    double threshold = network.get_thresholds()[0];

    const auto& weights = network.get_weights();
    double weight_normal_x = weights.get(0, 0);
    double weight_normal_y = weights.get(0, 1);
    double weight_line_x = weight_normal_y;
    double weight_line_y = -weight_normal_x;

    double length = sqrt((weight_line_x * weight_line_x) + (weight_line_y * weight_line_y));
    weight_line_x *= (1.0 / length) * 2.0;
    weight_line_y *= (1.0 / length) * 2.0;

    auto line_x = {-weight_line_x, weight_line_x};
    auto line_y = {-weight_line_y + threshold, weight_line_y + threshold};
    plt::plot(line_x, line_y, "b--");
}

static constexpr double activation_beta = 1.0 / 2.0;

double tanh_activation_function(double b)
{
    return tanh(activation_beta * b);
}

double tanh_activation_function_derivative(double b)
{
    return activation_beta * (1.0 - pow(tanh_activation_function(b), 2));
}

int
main()
{
    std::srand(static_cast<uint>(std::time(0)));

    // Test parameters
    const size_t NUM_TESTS = 10;
    const size_t NUM_TRAINING_ITERATIONS = 1000000;
    const double LEARNING_RATE = 0.02;

    // Load and process data
    std::vector<Pattern> training_data = read_and_parse_data("supervised_learning/train_data.txt");
    std::vector<Pattern> validation_data = read_and_parse_data("supervised_learning/valid_data.txt");
    normalize_input_data(training_data);
    normalize_input_data(validation_data);

    // Test result data (we only need to store results for the last test for this)
    std::vector<double> test_iter_vec;               test_iter_vec.resize(NUM_TRAINING_ITERATIONS);
    std::vector<double> test_training_energy_vec;    test_training_energy_vec.resize(NUM_TRAINING_ITERATIONS);
    std::vector<double> test_validation_energy_vec;  test_validation_energy_vec.resize(NUM_TRAINING_ITERATIONS);

    std::vector<double> classification_errors_training;    classification_errors_training.resize(NUM_TESTS);
    std::vector<double> classification_errors_validation;  classification_errors_validation.resize(NUM_TESTS);

    std::shared_ptr<NeuralNetwork> network;
    for (size_t test = 0; test < NUM_TESTS; ++test) {

        std::cout << "Performing test " << (test + 1) << "/" << NUM_TESTS << std::endl;

        // Set up neural network
        network = std::make_shared<NeuralNetwork>(2, 1, tanh_activation_function, tanh_activation_function_derivative);
        network->reset_weights(-0.2, +0.2);
        network->reset_thresholds(-1.0, +1.0);

        // Debug plotting (to see if there has been any change)
        //plot_input_data(training_data, validation_data);
        //plot_network_separation_line(network);

        for (size_t training_iteration = 0; training_iteration < NUM_TRAINING_ITERATIONS; ++training_iteration) {

            if (training_iteration % (NUM_TRAINING_ITERATIONS / 100) == 0) {
                std::cout << "\r  training... " << (100.0 * training_iteration / NUM_TRAINING_ITERATIONS) << " %" << std::flush;
            }

            // Train using a random pattern from the training data set
            size_t i = std::rand() % training_data.size();
            const Pattern& pattern = training_data[i];
            network->train(
                    {pattern.in_x, pattern.in_y},
                    {pattern.out},
                    LEARNING_RATE
            );

            double training_energy = energy_for_data_set(*network, training_data);
            double validation_energy = energy_for_data_set(*network, validation_data);

            // Store results for plotting
            test_iter_vec[training_iteration] = training_iteration;
            test_training_energy_vec[training_iteration] = training_energy;
            test_validation_energy_vec[training_iteration] = validation_energy;
        }
        std::cout << "\r  training... 100 %" << std::endl;

        // Calculate classification errors
        classification_errors_training[test] = classification_error_for_data_set(*network, training_data);
        classification_errors_validation[test] = classification_error_for_data_set(*network, validation_data);
    }

    // Report classification error data
    {
        double min_class_error_t = *std::min_element(classification_errors_training.begin(), classification_errors_training.end());
        std::cout << "Minimum classification error (training set):   " << min_class_error_t << std::endl;
        double min_class_error_v = *std::min_element(classification_errors_validation.begin(), classification_errors_validation.end());
        std::cout << "Minimum classification error (validation set): " << min_class_error_v << std::endl;

        double avg_class_error_t = average_value(classification_errors_training);
        std::cout << "Average classification error (training set):   " << avg_class_error_t << std::endl;
        double avg_class_error_v = average_value(classification_errors_validation);
        std::cout << "Average classification error (validation set): " << avg_class_error_v << std::endl;

        double var_class_error_t = variance_of_data(classification_errors_training, avg_class_error_t);
        std::cout << "Variance of classification error (training set):   " << var_class_error_t << std::endl;
        double var_class_error_v = variance_of_data(classification_errors_validation, avg_class_error_v);
        std::cout << "Variance of classification error (validation set): " << var_class_error_v << std::endl;
    }

    // Plot results (only the results from the *last* test!)
    std::cout << "Plotting" << std::endl;
    plt::figure();
    plt::subplot(1, 2, 1);
    {
        plt::title("Energy as the network is trained");
        plt::xlabel("t");
        plt::ylabel("Energy");
        plt::named_plot("Training set energy", test_iter_vec, test_training_energy_vec, "r-");
        plt::named_plot("Validation set energy", test_iter_vec, test_validation_energy_vec, "b-");
        plt::ylim(-165.0, -75.0);
        plt::legend();
        plt::grid(true);
    }
    plt::subplot(1, 2, 2);
    {
        plt::title("Input data points (training and validation) and network-defined line classifying the data");
        plot_input_data(training_data, validation_data);
        plot_network_separation_line(*network);
        plt::legend();
        plt::grid(true);
    }
    plt::show();


    return 0;
}
