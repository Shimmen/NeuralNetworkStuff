//
// Created by Simon Moos on 2017-08-30.
//

#ifndef CONVERGINGPATTERNS_PATTERN_H
#define CONVERGINGPATTERNS_PATTERN_H

#include <vector>
#include <initializer_list>

class Matrix
{
public:

    Matrix(const size_t width, const size_t height, const std::initializer_list<int>& pattern);
    ~Matrix() {}

    const size_t width;
    const size_t height;
    const size_t num_neurons;

    int get_linear(size_t i) const;
    void set_linear(size_t i, int value);

    int get(size_t i, size_t j) const;
    void set(size_t i, size_t j, int value);

    void debug_print(bool transform_values = true) const;

    bool operator==(const Matrix& other) const;

private:

    std::vector<int> state;

};


#endif //CONVERGINGPATTERNS_PATTERN_H
