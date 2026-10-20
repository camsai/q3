#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

namespace q3 {

/**
 * A simple, contiguous 2D array class to replace std::vector<std::vector<T>>.
 * This prevents memory fragmentation and improves cache locality in HPC contexts.
 */
template <typename T> class [[nodiscard]] Matrix {
  private:
    std::vector<T> data_;
    size_t rows_;
    size_t cols_;

  public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(size_t rows, size_t cols, const T& initial_value = T())
        : data_(rows * cols, initial_value), rows_(rows), cols_(cols) {}

    /// Resizes the matrix, invalidating previous content structure if dimensions change
    void resize(size_t rows, size_t cols) {
        data_.resize(rows * cols);
        rows_ = rows;
        cols_ = cols;
    }

    void resize(size_t rows, size_t cols, const T& initial_value) {
        data_.assign(rows * cols, initial_value);
        rows_ = rows;
        cols_ = cols;
    }

    void assign(size_t rows, size_t cols, const T& initial_value) {
        data_.assign(rows * cols, initial_value);
        rows_ = rows;
        cols_ = cols;
    }

    /// Access row `row_index`. Allows matrix[row_index][column_index] syntax to remain valid.
    T* operator[](size_t row_index) {
        return data_.data() + row_index * cols_;
    }

    [[nodiscard]] const T* operator[](size_t row_index) const {
        return data_.data() + row_index * cols_;
    }

    /// 2D indexing operator
    T& operator()(size_t row_index, size_t column_index) {
        return data_[row_index * cols_ + column_index];
    }

    [[nodiscard]] const T& operator()(size_t row_index, size_t column_index) const {
        return data_[row_index * cols_ + column_index];
    }

    [[nodiscard]] size_t rows() const {
        return rows_;
    }
    [[nodiscard]] size_t cols() const {
        return cols_;
    }

    /// Returns the number of rows (to mimic outer std::vector::size())
    [[nodiscard]] size_t size() const {
        return rows_;
    }

    /// Returns the underlying flat data pointer
    T* data() {
        return data_.data();
    }
    [[nodiscard]] const T* data() const {
        return data_.data();
    }

    /// Returns raw vector (useful for MPI calls that need contiguous block)
    std::vector<T>& raw() {
        return data_;
    }
    [[nodiscard]] const std::vector<T>& raw() const {
        return data_;
    }

    [[nodiscard]] bool empty() const {
        return data_.empty();
    }
};

} // namespace q3
