//
// Created by Simon Moos on 2017-09-20.
//

#ifndef NN_RANDOM_H
#define NN_RANDOM_H

#include <random>
#include <cassert>

inline
double
random_double()
{
    return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}

inline
double
random_double_in_range(double min, double max)
{
    assert(max > min);
    return min + (max - min) * random_double();
}

#endif // NN_RANDOM_H
