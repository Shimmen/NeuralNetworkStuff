//
// Created by Simon Moos on 2017-08-30.
//

#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <vector>
#include <cassert>
#include <initializer_list>

template<typename T>
class Matrix
{
public:

    Matrix(const size_t width, const size_t height = 1, const std::initializer_list<T>& pattern = {});
    ~Matrix() {}

    size_t width;
    size_t height;
    size_t num_elements;

    T get_linear(size_t i) const;
    void set_linear(size_t i, T value);

    T get(size_t i, size_t j) const;
    void set(size_t i, size_t j, T value);

    void debug_print(bool transform_values = true) const;

private:

    std::vector<T> state;

};

////////////////////////////

typedef Matrix<int> Pattern;
typedef Matrix<double> WeightMatrix;

////////////////////////////
////// Implementation //////
////////////////////////////

template<typename T>
Matrix<T>::Matrix(const size_t width, const size_t height, const std::initializer_list<T>& pattern)
        : width(width)
        , height(height)
        , num_elements(width * height)
{
    unsigned long size = static_cast<unsigned long>(width * height);

    if (pattern.size()) {
        assert(pattern.size() == size);
        state.reserve(size);
        state.assign(pattern.begin(), pattern.end());
    } else {
        state.resize(size);
    }
}

template<typename T>
void
Matrix<T>::debug_print(bool transform_values) const
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

template<typename T>
T
Matrix<T>::get_linear(size_t i) const
{
    assert(i >= 0 && i < num_elements);
    return state[i];
}

template<typename T>
void
Matrix<T>::set_linear(size_t i, T value)
{
    assert(i >= 0 && i < num_elements);
    state[i] = value;
}

template<typename T>
T
Matrix<T>::get(size_t i, size_t j) const
{
    assert(i >= 0 && i < height);
    assert(j >= 0 && j < width);
    return state[j + i * width];
}

template<typename T>
void
Matrix<T>::set(size_t i, size_t j, T value)
{
    assert(i >= 0 && i < height);
    assert(j >= 0 && j < width);
    state[j + i * width] = value;
}

////////////////////////////

template<typename T>
bool
operator==(const Matrix<T>& first, const Matrix<T>& second)
{
    if (first.num_elements != second.num_elements) return false;
    if (first.width  != second.width) return false;
    if (first.height != second.height) return false;

    for (size_t i = 0; i < first.num_elements; ++i) {
        if (first.get_linear(i) != second.get_linear(i)) {
            return false;
        }
    }

    return true;
}

#endif // NN_MATRIX_H
