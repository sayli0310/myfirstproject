#pragma once
#include "matrix.hpp"
#include <cmath>
#include <stdexcept>

// ============================================================
// LayerNorm: normalizes each token's vector independently across
// the d_model dimension, then applies learned scale (gamma) and
// shift (beta).
// ============================================================
class LayerNorm {
public:
    int d_model;
    float epsilon;
    std::vector<float> gamma; // [d_model], learned, init to 1.0
    std::vector<float> beta;  // [d_model], learned, init to 0.0

    LayerNorm(int d_model_, float epsilon_ = 1e-5f)
        : d_model(d_model_), epsilon(epsilon_),
          gamma(d_model_, 1.0f), beta(d_model_, 0.0f) {
        if (d_model_ <= 0) {
            throw std::invalid_argument("LayerNorm: d_model must be positive");
        }
    }

    // x: [seq_len x d_model]
    Matrix forward(const Matrix& x) const {
        if (x.cols != d_model) {
            throw std::invalid_argument(
                "LayerNorm::forward: d_model mismatch (" +
                std::to_string(x.cols) + " vs " + std::to_string(d_model) + ")");
        }
        Matrix result(x.rows, x.cols);
        for (int i = 0; i < x.rows; i++) {
            const float* row = x.row_ptr(i);

            float mean = 0.0f;
            for (int j = 0; j < d_model; j++) mean += row[j];
            mean /= d_model;

            float variance = 0.0f;
            for (int j = 0; j < d_model; j++) {
                float diff = row[j] - mean;
                variance += diff * diff;
            }
            variance /= d_model;

            float inv_std = 1.0f / std::sqrt(variance + epsilon);

            float* out_row = result.row_ptr(i);
            for (int j = 0; j < d_model; j++) {
                out_row[j] = gamma[j] * (row[j] - mean) * inv_std + beta[j];
            }
        }
        return result;
    }
};

// ============================================================
// GELU activation (the one GPT-2 uses in its MLP block).
// Using the tanh-based approximation, same as the original
// GPT-2 paper/codebase, for numerical match with real weights.
// ============================================================
inline float gelu(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}

// ============================================================
// FeedForward: per-token MLP. Expands d_model -> hidden (usually
// 4x) with GELU, then projects back down to d_model.
//   FFN(x) = W2 . gelu(W1 . x + b1) + b2
// ============================================================
class FeedForward {
public:
    int d_model;
    int d_hidden;
    Matrix W1, W2;            // [d_model x d_hidden], [d_hidden x d_model]
    std::vector<float> b1, b2; // biases

    FeedForward(int d_model_, int d_hidden_)
        : d_model(d_model_), d_hidden(d_hidden_),
          W1(d_model_, d_hidden_), W2(d_hidden_, d_model_),
          b1(d_hidden_, 0.0f), b2(d_model_, 0.0f) {
        if (d_hidden_ <= 0) {
            throw std::invalid_argument("FeedForward: d_hidden must be positive");
        }
    }

    Matrix forward(const Matrix& x) const {
        if (x.cols != d_model) {
            throw std::invalid_argument(
                "FeedForward::forward: d_model mismatch (" +
                std::to_string(x.cols) + " vs " + std::to_string(d_model) + ")");
        }
        Matrix hidden = x.matmul(W1); // [seq x d_hidden]
        for (int i = 0; i < hidden.rows; i++) {
            float* row = hidden.row_ptr(i);
            for (int j = 0; j < d_hidden; j++) {
                row[j] = gelu(row[j] + b1[j]);
            }
        }
        Matrix out = hidden.matmul(W2); // [seq x d_model]
        for (int i = 0; i < out.rows; i++) {
            float* row = out.row_ptr(i);
            for (int j = 0; j < d_model; j++) {
                row[j] += b2[j];
            }
        }
        return out;
    }
};
