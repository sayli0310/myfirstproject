#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>

// ============================================================
// Matrix: the single most-used class in the whole engine.
// Every projection (Q,K,V, MLP, output head) is a matmul call.
// Stored row-major: element (r,c) lives at data[r*cols + c]
// ============================================================
class Matrix {
public:
    int rows;
    int cols;
    std::vector<float> data;

    // ---- Constructors ----
    Matrix() : rows(0), cols(0) {}

    Matrix(int r, int c) : rows(r), cols(c) {
        if (r <= 0 || c <= 0) {
            throw std::invalid_argument(
                "Matrix: rows and cols must be positive, got (" +
                std::to_string(r) + ", " + std::to_string(c) + ")");
        }
        data.assign(static_cast<size_t>(r) * c, 0.0f);
    }

    // ---- Bounds-checked element access ----
    // at() throws instead of silently reading garbage memory.
    // Use this everywhere except in the hottest inner loops
    // (where you've already proven indices are safe).
    float& at(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw std::out_of_range(
                "Matrix::at(" + std::to_string(r) + "," + std::to_string(c) +
                ") out of bounds for matrix of shape (" +
                std::to_string(rows) + "," + std::to_string(cols) + ")");
        }
        return data[static_cast<size_t>(r) * cols + c];
    }

    const float& at(int r, int c) const {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw std::out_of_range(
                "Matrix::at(" + std::to_string(r) + "," + std::to_string(c) +
                ") out of bounds for matrix of shape (" +
                std::to_string(rows) + "," + std::to_string(cols) + ")");
        }
        return data[static_cast<size_t>(r) * cols + c];
    }

    // Raw pointer to start of a row — used in hot loops where
    // bounds-checked at() would add overhead per element.
    float* row_ptr(int r) {
        if (r < 0 || r >= rows) {
            throw std::out_of_range("Matrix::row_ptr(" + std::to_string(r) +
                                     ") out of bounds, rows=" + std::to_string(rows));
        }
        return &data[static_cast<size_t>(r) * cols];
    }

    const float* row_ptr(int r) const {
        if (r < 0 || r >= rows) {
            throw std::out_of_range("Matrix::row_ptr(" + std::to_string(r) +
                                     ") out of bounds, rows=" + std::to_string(rows));
        }
        return &data[static_cast<size_t>(r) * cols];
    }

    // ---- Matrix multiplication: (rows x cols) . (other.rows x other.cols) ----
    // Shape rule: this.cols MUST equal other.rows
    Matrix matmul(const Matrix& other) const {
        if (this->cols != other.rows) {
            std::ostringstream oss;
            oss << "Matrix::matmul shape mismatch: ("
                << this->rows << "x" << this->cols << ") . ("
                << other.rows << "x" << other.cols
                << ") -- inner dimensions " << this->cols
                << " and " << other.rows << " must match";
            throw std::invalid_argument(oss.str());
        }

        Matrix result(this->rows, other.cols);

        // Loop order i-k-j is deliberately chosen for cache locality:
        // for fixed i,k we stream through a full row of `other`
        // (contiguous in memory) and accumulate into a full row of result.
        for (int i = 0; i < this->rows; i++) {
            const float* a_row = this->row_ptr(i);
            float* out_row = result.row_ptr(i);
            for (int k = 0; k < this->cols; k++) {
                float a_val = a_row[k];
                if (a_val == 0.0f) continue; // skip multiply-by-zero (cheap win)
                const float* b_row = other.row_ptr(k);
                for (int j = 0; j < other.cols; j++) {
                    out_row[j] += a_val * b_row[j];
                }
            }
        }
        return result;
    }

    // ---- Element-wise add (used for residual connections) ----
    Matrix add(const Matrix& other) const {
        if (this->rows != other.rows || this->cols != other.cols) {
            std::ostringstream oss;
            oss << "Matrix::add shape mismatch: ("
                << this->rows << "x" << this->cols << ") vs ("
                << other.rows << "x" << other.cols << ")";
            throw std::invalid_argument(oss.str());
        }
        Matrix result(this->rows, this->cols);
        for (size_t i = 0; i < data.size(); i++) {
            result.data[i] = this->data[i] + other.data[i];
        }
        return result;
    }

    // ---- Transpose ----
    Matrix transpose() const {
        Matrix result(this->cols, this->rows);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result.at(c, r) = this->at(r, c);
        return result;
    }

    // ---- Scale every element by a scalar (used for 1/sqrt(d_k)) ----
    Matrix scale(float factor) const {
        Matrix result(*this);
        for (auto& v : result.data) v *= factor;
        return result;
    }

    void print(const std::string& label = "", int max_rows = 5, int max_cols = 6) const {
        if (!label.empty()) std::cout << label << " ";
        std::cout << "[" << rows << "x" << cols << "]\n";
        for (int r = 0; r < std::min(rows, max_rows); r++) {
            std::cout << "  ";
            for (int c = 0; c < std::min(cols, max_cols); c++) {
                std::cout << at(r, c) << " ";
            }
            if (cols > max_cols) std::cout << "...";
            std::cout << "\n";
        }
        if (rows > max_rows) std::cout << "  ...\n";
    }
};

// ============================================================
// Free function: dot product of two equal-length float vectors.
// Used inside attention score computation.
// ============================================================
inline float dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

inline float dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "dot(): vector size mismatch (" + std::to_string(a.size()) +
            " vs " + std::to_string(b.size()) + ")");
    }
    return dot(a.data(), b.data(), static_cast<int>(a.size()));
}
