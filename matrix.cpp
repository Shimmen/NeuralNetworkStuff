//
// Created by Simon Moos on 2017-08-30.
//

#include "matrix.h"

#include <cassert>
#include <iostream>

Matrix::Matrix(const size_t width, const size_t height, const std::initializer_list<int>& pattern)
    : width(width)
    , height(height)
    , num_neurons(width * height)
{
    unsigned long size = static_cast<unsigned long>(width * height);
    if (pattern.size()) {
        state.reserve(size);
        state.assign(pattern.begin(), pattern.end());
    } else {
        state.resize(size);
    }
}

void
Matrix::debug_print(bool transform_values) const
{
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            int val = get(i, j);
            val = (transform_values) ? (val + 1) / 2 : val;
            std::cout << " " << val;
        }
        std::cout << std::endl;
    }
}

int
Matrix::get_linear(size_t i) const
{
    assert(i >= 0 && i < num_neurons);
    return state[i];
}

void
Matrix::set_linear(size_t i, int value)
{
    assert(i >= 0 && i < num_neurons);
    state[i] = value;
}

int
Matrix::get(size_t i, size_t j) const
{
    assert(i >= 0 && i < height);
    assert(j >= 0 && j < width);
    return state[j + i * width];
}

void
Matrix::set(size_t i, size_t j, int value)
{
    assert(i >= 0 && i < height);
    assert(j >= 0 && j < width);
    state[j + i * width] = value;
}

bool
Matrix::operator==(const Matrix& other) const
{
    if (this->num_neurons != other.num_neurons) return false;
    if (this->width  != other.width) return false;
    if (this->height != other.height) return false;

    for (size_t i = 0; i < this->num_neurons; ++i) {
        if (this->get_linear(i) != other.get_linear(i)) {
            return false;
        }
    }

    return true;
}


