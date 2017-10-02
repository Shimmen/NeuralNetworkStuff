#include <iostream>
#include <fstream>
#include <random>

#define WITHOUT_NUMPY
#include <matplotlib-cpp/matplotlibcpp.h>

namespace plt = matplotlibcpp;

///////////////////////////////////////////////
// Data reading, parsing, preprocessing

struct Pattern
{
    double x;
    double y;
    Pattern(double x, double y) : x(x), y(y) {}
};

std::vector<Pattern>
read_and_parse_data(const std::string& file_name)
{
    std::vector<Pattern> patterns;

    std::ifstream ifs(file_name);

    double x;
    double y;

    while (ifs >> x >> y) {
        patterns.emplace_back(x, y);
    }

    return patterns;
}

void
normalize_training_data_mean(std::vector<Pattern>& training_data)
{
    double length = training_data.size();

    double x_sum = 0.0;
    double y_sum = 0.0;
    for (auto &pattern : training_data) {
        x_sum += pattern.x;
        y_sum += pattern.y;
    }
    double x_mean = x_sum / length;
    double y_mean = y_sum / length;

    for (size_t i = 0; i < training_data.size(); ++i) {
        training_data[i].x -= x_mean;
        training_data[i].y -= y_mean;
    }
}


///////////////////////////////////////////////
// Plotting

void
plot_weight_and_training_data(const double weights[2], const std::vector<Pattern>& training_data)
{
    // Plot input data
    size_t num_inputs = training_data.size();
    std::vector<double> input_x(num_inputs);
    std::vector<double> input_y(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto& point = training_data[i];
        input_x[i] = point.x;
        input_y[i] = point.y;
    }
    plt::named_plot("Input patterns", input_x, input_y, "rx");

    // Plot network weight (hopefully the principle direction)
    plt::named_plot("Weights", { 0.0, weights[0] }, { 0.0, weights[1] }, "b-");
    plt::plot({ weights[0] }, { weights[1] }, "bo");

    plt::grid(true);
    plt::legend();
    plt::xlabel("x1");
    plt::ylabel("x2");
}

///////////////////////////////////////////////
// Test procedure

void train_network(double weights[2], const Pattern& pattern, const double learning_rate)
{
    // Oja's rule
    double target = pattern.x * weights[0] + pattern.y * weights[1];
    double delta_x = learning_rate * target * (pattern.x - target * weights[0]);
    double delta_y = learning_rate * target * (pattern.y - target * weights[1]);

    weights[0] += delta_x;
    weights[1] += delta_y;
}

int
main()
{
    clock_t start_time = std::clock();

    std::default_random_engine rng;
    rng.seed(static_cast<unsigned int>(start_time));

    const double LEARNING_RATE = 0.001;
    const size_t NUM_UPDATE_STEPS = 20000;

    // Load training data
    std::vector<Pattern> training_data = read_and_parse_data("oja_unsupervised/data.txt");

    std::uniform_int_distribution<size_t> random_training_data_sample_index(0, training_data.size() - 1);
    std::uniform_real_distribution<double> random_double(-1.0, 1.0);

    // Set up for plotting
    std::vector<double> weight_magnitude_key(NUM_UPDATE_STEPS);
    std::vector<double> weight_magnitude_val(NUM_UPDATE_STEPS);
    plt::plot();

    //
    // Train with un-normalized data
    //
    {
        // Set up network
        double weights[2] = { random_double(rng), random_double(rng) };

        for (size_t step = 0; step < NUM_UPDATE_STEPS; ++step) {

            // Perform one training step
            size_t i = random_training_data_sample_index(rng);
            train_network(weights, training_data[i], LEARNING_RATE);

            // Record weight magnitude
            weight_magnitude_key[step] = step;
            weight_magnitude_val[step] = std::sqrt(weights[0] * weights[0] + weights[1] * weights[1]);
        }

        // Plot weight magnitude/modulus in panel 1
        plt::subplot(2, 2, 1);
        plt::title("Weight magnitude over time");
        plt::named_plot("Weight magnitude", weight_magnitude_key, weight_magnitude_val, "r-");
        plt::ylim(0.0, 1.25);
        plt::grid(true);
        plt::legend();

        // Plot input data and weights
        plt::subplot(2, 2, 3);
        plt::title("Input patterns & network weight");
        plot_weight_and_training_data(weights, training_data);
        plt::xlim(-2.0, 13.0);
        plt::ylim(-1.0, 4.0);
        plt::grid(true);
        plt::legend();
    }

    //
    // Train with normalized data
    //
    {
        // Set up network
        double weights[2] = { random_double(rng), random_double(rng) };

        // This time, normalize the training data!
        normalize_training_data_mean(training_data);

        for (size_t step = 0; step < NUM_UPDATE_STEPS; ++step) {

            // Perform one training step
            size_t i = random_training_data_sample_index(rng);
            train_network(weights, training_data[i], LEARNING_RATE);

            // Record weight magnitude
            weight_magnitude_key[step] = step;
            weight_magnitude_val[step] = std::sqrt(weights[0] * weights[0] + weights[1] * weights[1]);
        }

        // Plot weight magnitude/modulus in panel 2
        plt::subplot(2, 2, 2);
        plt::title("Weight magnitude over time (data with zero mean)");
        plt::named_plot("Weight magnitude", weight_magnitude_key, weight_magnitude_val, "r-");
        plt::ylim(0.0, 1.25);
        plt::grid(true);
        plt::legend();

        // Plot input data and weights
        plt::subplot(2, 2, 4);
        plt::title("Input patterns & network weight (data with zero mean)");
        plot_weight_and_training_data(weights, training_data);
        plt::grid(true);
        plt::legend();
    }

    clock_t end_time = std::clock();
    double time_elapsed_s = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << time_elapsed_s << " s" << std::endl;

    // Blocks, so perform timing before this call
    plt::show();

    return 0;
}
