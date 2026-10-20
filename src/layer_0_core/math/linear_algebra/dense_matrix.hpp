#pragma once

/**
 * DenseMatrix — a small dense matrix stored column-major, the layout every
 * BLAS and LAPACK routine reads and every checkpoint block is written in.
 *
 * Matrix<T> (types/matrix.hpp) is row-major and is the shape of a plane-wave
 * coefficient array: a row per band, indexed [band][plane_wave]. A matrix that
 * is multiplied, adjointed or diagonalised is a different object -- it goes to
 * zgemm and zheev as a pointer and a leading dimension, and those read
 * columns -- so it has a type of its own rather than a transposed reading of
 * the other. Element (row, column) lives at data[column * rows + row].
 *
 * The products, the eigensolver and the singular value decomposition live in
 * their own files beside this one; this is the storage and the element-wise
 * operations only.
 */

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layer_0_core/types/type_aliases.hpp"

namespace q3 {

namespace internal::dense_matrix {

[[nodiscard]] inline Real conjugate(Real value) {
    return value;
}

[[nodiscard]] inline Complex conjugate(const Complex& value) {
    return std::conj(value);
}

} // namespace internal::dense_matrix

template <typename T> class [[nodiscard]] DenseMatrix {
  public:
    DenseMatrix() = default;

    DenseMatrix(int rows, int columns, const T& initial_value = T{})
        : data_(static_cast<std::size_t>(require_dimension(rows, "rows")) *
                    static_cast<std::size_t>(require_dimension(columns, "columns")),
                initial_value)
        , rows_(rows)
        , columns_(columns) {}

    /// Adopts a column-major buffer; refuses one whose size is not rows * columns.
    DenseMatrix(int rows, int columns, std::vector<T> column_major_data)
        : data_(std::move(column_major_data))
        , rows_(require_dimension(rows, "rows"))
        , columns_(require_dimension(columns, "columns")) {
        if (data_.size() != number_of_elements()) {
            throw std::invalid_argument(
                "DenseMatrix: a " + std::to_string(rows) + " x " + std::to_string(columns) +
                " matrix holds " + std::to_string(number_of_elements()) + " elements, not the " +
                std::to_string(data_.size()) + " given");
        }
    }

    [[nodiscard]] static DenseMatrix identity(int dimension) {
        DenseMatrix matrix(dimension, dimension, T{});
        for (int index = 0; index < dimension; index++) {
            matrix(index, index) = T{1};
        }
        return matrix;
    }

    [[nodiscard]] int rows() const {
        return rows_;
    }
    [[nodiscard]] int columns() const {
        return columns_;
    }
    [[nodiscard]] std::size_t number_of_elements() const {
        return static_cast<std::size_t>(rows_) * static_cast<std::size_t>(columns_);
    }
    [[nodiscard]] bool empty() const {
        return data_.empty();
    }

    T& operator()(int row, int column) {
        return data_[index_of(row, column)];
    }
    [[nodiscard]] const T& operator()(int row, int column) const {
        return data_[index_of(row, column)];
    }

    /// The column-major buffer: with rows() it is the (pointer, leading dimension) BLAS takes.
    T* data() {
        return data_.data();
    }
    [[nodiscard]] const T* data() const {
        return data_.data();
    }
    /// Read-only: a caller that could resize the buffer could break rows() x columns().
    [[nodiscard]] const std::vector<T>& raw() const {
        return data_;
    }

    /// The conjugate transpose, columns x rows.
    [[nodiscard]] DenseMatrix adjoint() const {
        DenseMatrix result(columns_, rows_, T{});
        for (int column = 0; column < columns_; column++) {
            for (int row = 0; row < rows_; row++) {
                result(column, row) = internal::dense_matrix::conjugate((*this)(row, column));
            }
        }
        return result;
    }

    /// The sum of every element's squared modulus.
    [[nodiscard]] Real frobenius_norm_squared() const {
        Real total = 0.0;
        for (const T& value : data_) {
            total += std::norm(value);
        }
        return total;
    }

    /// factor * this, element by element.
    template <typename Factor> [[nodiscard]] DenseMatrix scaled(const Factor& factor) const {
        DenseMatrix result = *this;
        for (T& value : result.data_) {
            value *= factor;
        }
        return result;
    }

    /// this += factor * other, element by element; the two shapes must agree.
    template <typename Factor>
    DenseMatrix& add_scaled(const Factor& factor, const DenseMatrix& other) {
        require_same_shape(other, "add_scaled");
        for (std::size_t index = 0; index < data_.size(); index++) {
            data_[index] += factor * other.data_[index];
        }
        return *this;
    }

    /**
     * Re tr(this^dagger other), the sum of Re(conj(a) b) over matching
     * elements: the inner product under which the matrices of one shape are
     * a real vector space, which is what a gradient and a search direction
     * are compared in.
     */
    [[nodiscard]] Real real_inner_product(const DenseMatrix& other) const {
        require_same_shape(other, "real_inner_product");
        Real total = 0.0;
        for (std::size_t index = 0; index < data_.size(); index++) {
            total +=
                std::real(internal::dense_matrix::conjugate(data_[index]) * other.data_[index]);
        }
        return total;
    }

    /// The rows and columns named, in the order named; an index outside the matrix is refused.
    [[nodiscard]] DenseMatrix submatrix(const std::vector<int>& row_indices,
                                        const std::vector<int>& column_indices) const {
        DenseMatrix result(
            static_cast<int>(row_indices.size()), static_cast<int>(column_indices.size()), T{});
        for (std::size_t column = 0; column < column_indices.size(); column++) {
            for (std::size_t row = 0; row < row_indices.size(); row++) {
                result(static_cast<int>(row), static_cast<int>(column)) =
                    (*this)(row_indices[row], column_indices[column]);
            }
        }
        return result;
    }

    /// diag(factors) * this: row i scaled by factors[i], which needs one factor per row.
    template <typename Factor>
    [[nodiscard]] DenseMatrix scaled_rows(const std::vector<Factor>& factors) const {
        require_count(factors.size(), rows_, "row");
        DenseMatrix result = *this;
        for (int column = 0; column < columns_; column++) {
            for (int row = 0; row < rows_; row++) {
                result(row, column) *= factors[static_cast<std::size_t>(row)];
            }
        }
        return result;
    }

    /// this * diag(factors): column j scaled by factors[j], which needs one factor per column.
    template <typename Factor>
    [[nodiscard]] DenseMatrix scaled_columns(const std::vector<Factor>& factors) const {
        require_count(factors.size(), columns_, "column");
        DenseMatrix result = *this;
        for (int column = 0; column < columns_; column++) {
            for (int row = 0; row < rows_; row++) {
                result(row, column) *= factors[static_cast<std::size_t>(column)];
            }
        }
        return result;
    }

    [[nodiscard]] bool operator==(const DenseMatrix& other) const {
        return rows_ == other.rows_ && columns_ == other.columns_ && data_ == other.data_;
    }
    [[nodiscard]] bool operator!=(const DenseMatrix& other) const {
        return !(*this == other);
    }

  private:
    [[nodiscard]] std::size_t index_of(int row, int column) const {
        if (row < 0 || row >= rows_ || column < 0 || column >= columns_) {
            throw std::out_of_range("DenseMatrix: element (" + std::to_string(row) + ", " +
                                    std::to_string(column) + ") is outside a " +
                                    std::to_string(rows_) + " x " + std::to_string(columns_) +
                                    " matrix");
        }
        return static_cast<std::size_t>(column) * static_cast<std::size_t>(rows_) +
               static_cast<std::size_t>(row);
    }

    [[nodiscard]] static int require_dimension(int value, const char* name) {
        if (value < 0) {
            throw std::invalid_argument("DenseMatrix: the number of " + std::string(name) +
                                        " cannot be negative, got " + std::to_string(value));
        }
        return value;
    }

    void require_same_shape(const DenseMatrix& other, const char* operation) const {
        if (rows_ != other.rows_ || columns_ != other.columns_) {
            throw std::invalid_argument(
                "DenseMatrix::" + std::string(operation) + ": a " + std::to_string(rows_) + " x " +
                std::to_string(columns_) + " matrix and a " + std::to_string(other.rows_) + " x " +
                std::to_string(other.columns_) + " one have different shapes");
        }
    }

    static void require_count(std::size_t given, int wanted, const char* axis) {
        if (given != static_cast<std::size_t>(wanted)) {
            throw std::invalid_argument("DenseMatrix: one factor per " + std::string(axis) +
                                        " is required, " + std::to_string(wanted) +
                                        " of them, not " + std::to_string(given));
        }
    }

    std::vector<T> data_;
    int rows_ = 0;
    int columns_ = 0;
};

} // namespace q3
