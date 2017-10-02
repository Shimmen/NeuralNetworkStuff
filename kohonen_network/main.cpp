#include <iostream>
#include <fstream>
#include <random>

#include <matrix.h>

#include "kohonen-network.h"

#define WITHOUT_NUMPY
#include <matplotlib-cpp/matplotlibcpp.h>

namespace plt = matplotlibcpp;

///////////////////////////////////////////////
// Input data

struct InputPoint
{
    double x;
    double y;
    InputPoint(double x, double y) : x(x), y(y) {}
};

std::vector<InputPoint>
generate_input_data(const size_t count, std::default_random_engine& rng)
{
    std::vector<InputPoint> points;
    points.reserve(count);

    std::uniform_real_distribution<double> random_double_zero_to_one(0.0, 1.0);

    while (points.size() < count) {
        double x1 = random_double_zero_to_one(rng);
        double x2 = random_double_zero_to_one(rng);
        if (x1 <= 0.5 || x2 >= 0.5) {
            points.emplace_back(x1, x2);
        }
    }

    return points;
}

///////////////////////////////////////////////
// Plotting

void
plot_network_and_training_data(const KohonenNetwork& network, const std::vector<InputPoint>& training_data)
{
    const Matrix<double>& weights = network.get_weights();
    assert(weights.width == 2); // must be able to be plotted in 2D!

    // Plot network
    std::vector<double> weight_x(weights.height);
    std::vector<double> weight_y(weights.height);
    for (size_t i = 0; i < weights.height; ++i) {
        weight_x[i] = weights.get(i, 0);
        weight_y[i] = weights.get(i, 1);
    }
    plt::named_plot("Weights", weight_x, weight_y, "bo-");

    // Plot input data
    size_t num_inputs = training_data.size();
    std::vector<double> input_x(num_inputs);
    std::vector<double> input_y(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto& point = training_data[i];
        input_x[i] = point.x;
        input_y[i] = point.y;
    }
    plt::named_plot("Input data", input_x, input_y, "rx");

    plt::grid(true);
    plt::legend();
    plt::xlabel("x1");
    plt::ylabel("x2");
}

///////////////////////////////////////////////
// Test procedure

double
one_dimensional_neighbouring_function(size_t i, size_t i0, double gaussian_width)
{
    // Since we have a 1D Kohonen network the neuron position we use is just its index
    double dist_squared = (i - i0) * (i - i0);
    return std::exp(-dist_squared / (2.0 * gaussian_width * gaussian_width));
}

int
main()
{
    clock_t start_time = std::clock();

    std::default_random_engine rng;

    const size_t TRAINING_DATA_SIZE = 1000;
    const size_t ORDERING_PHASE_STEPS = 1000;
    const size_t CONVERGENCE_PHASE_STEPS = 20000;

    const auto& training_data = generate_input_data(TRAINING_DATA_SIZE, rng);
    std::uniform_int_distribution<size_t> random_training_data_sample_index(0, TRAINING_DATA_SIZE - 1);

    KohonenNetwork network(2, 100, one_dimensional_neighbouring_function);
    network.reset_weights(-1.0, 1.0);

    int next_plot_index = 1;
    int num_plots = 2;
    plt::plot();

    // Debug plot of initial state
    /*
    {
        num_plots = 3;
        plt::subplot(1, num_plots, next_plot_index++);
        plot_network_and_training_data(network, training_data);
        plt::xlim(-1.1, 1.1);
        plt::ylim(-1.1, 1.5);
    }

     */

    //
    // Ordering phase
    //

    const double LEARNING_RATE_ZERO = 0.1;
    const double GAUSSIAN_WIDTH_ZERO = 100.0;
    const double TAU_SIGMA = 300.0;

    for (size_t step = 0; step < ORDERING_PHASE_STEPS; ++step) {

        // Calculate time-dependent parameters
        auto t = static_cast<double>(step);
        double learning_rate = LEARNING_RATE_ZERO * std::exp(-t / TAU_SIGMA);
        double gaussian_width = GAUSSIAN_WIDTH_ZERO * std::exp(-t / TAU_SIGMA);

        // Pick random sample to train on
        size_t i = random_training_data_sample_index(rng);
        auto& point = training_data[i];

        network.train({point.x, point.y}, learning_rate, gaussian_width);
    }

    plt::subplot(1, num_plots, next_plot_index++);
    plt::title("Network state after ordering phase");
    plot_network_and_training_data(network, training_data);
    plt::xlim(-0.1, 1.1);
    plt::ylim(-0.1, 1.25);

    //
    // Convergence phase
    //

    const double LEARNING_RATE_CONVERGENCE = 0.01;
    const double GAUSSIAN_WIDTH_CONVERGENCE = 0.9;

    for (size_t step = 0; step < CONVERGENCE_PHASE_STEPS; ++step) {

        // Pick random sample to train on
        size_t i = random_training_data_sample_index(rng);
        auto& point = training_data[i];

        network.train({point.x, point.y}, LEARNING_RATE_CONVERGENCE, GAUSSIAN_WIDTH_CONVERGENCE);
    }

    plt::subplot(1, num_plots, next_plot_index++);
    plt::title("Network state after convergence phase");
    plot_network_and_training_data(network, training_data);
    plt::xlim(-0.1, 1.1);
    plt::ylim(-0.1, 1.25);

    clock_t end_time = std::clock();
    double time_elaped_s = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << time_elaped_s << " s" << std::endl;

    // Blocks, so perform timing before this call
    plt::show();

    return 0;
}
