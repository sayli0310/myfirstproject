#pragma once
#include "matrix.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

// ============================================================
// softmax: converts a row of raw scores into a probability
// distribution that sums to 1. Operates IN PLACE on a single
// row of floats. Uses the subtract-max trick to avoid expf()
// overflow on large inputs.
// ============================================================
inline void softmax_row(float* row, int n) {
    if (n <= 0) {
        throw std::invalid_argument("softmax_row: length must be positive");
    }
    float max_val = row[0];
    for (int i = 1; i < n; i++) {
        if (row[i] > max_val) max_val = row[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        row[i] = std::exp(row[i] - max_val); // never overflows: exponent <= 0
        sum += row[i];
    }

    if (sum == 0.0f || !std::isfinite(sum)) {
        throw std::runtime_error("softmax_row: degenerate sum (all -inf row? check masking)");
    }

    for (int i = 0; i < n; i++) row[i] /= sum;
}

// Applies softmax independently to every row of a matrix (each
// row = one query token's distribution over all key positions).
inline void softmax_matrix_rows(Matrix& m) {
    for (int r = 0; r < m.rows; r++) {
        softmax_row(m.row_ptr(r), m.cols);
    }
}

// ============================================================
// apply_causal_mask: sets scores[i][j] = -infinity for all j > i,
// so token i can never attend to a future token j. Must be called
// BEFORE softmax (softmax turns -inf into exactly 0 probability).
// ============================================================
inline void apply_causal_mask(Matrix& scores) {
    if (scores.rows != scores.cols) {
        throw std::invalid_argument(
            "apply_causal_mask: expected a square [seq x seq] score matrix, got (" +
            std::to_string(scores.rows) + "x" + std::to_string(scores.cols) + ")");
    }
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < scores.rows; i++) {
        for (int j = i + 1; j < scores.cols; j++) {
            scores.at(i, j) = neg_inf;
        }
    }
}

// ============================================================
// AttentionHead: single-head scaled dot-product attention.
// Owns its own Wq, Wk, Wv projection weights.
//
//   Q = X . Wq     [seq x d_k]
//   K = X . Wk     [seq x d_k]
//   V = X . Wv     [seq x d_k]
//   scores = Q . K^T / sqrt(d_k)   [seq x seq]
//   scores = causal_mask(scores)
//   weights = softmax(scores)      (row-wise)
//   output = weights . V           [seq x d_k]
// ============================================================
class AttentionHead {
public:
    int d_model;
    int d_k;
    Matrix Wq, Wk, Wv; // each [d_model x d_k]

    AttentionHead(int d_model_, int d_k_)
        : d_model(d_model_), d_k(d_k_),
          Wq(d_model_, d_k_), Wk(d_model_, d_k_), Wv(d_model_, d_k_) {
        if (d_k_ <= 0) {
            throw std::invalid_argument("AttentionHead: d_k must be positive");
        }
    }

    // x: [seq_len x d_model] input embeddings (post layer-norm in real usage)
    // causal: whether to apply the future-token mask (true for GPT-style generation)
    Matrix forward(const Matrix& x, bool causal = true) const {
        if (x.cols != d_model) {
            throw std::invalid_argument(
                "AttentionHead::forward: input d_model mismatch (" +
                std::to_string(x.cols) + " vs " + std::to_string(d_model) + ")");
        }

        Matrix Q = x.matmul(Wq);              // [seq x d_k]
        Matrix K = x.matmul(Wk);              // [seq x d_k]
        Matrix V = x.matmul(Wv);              // [seq x d_k]

        Matrix Kt = K.transpose();            // [d_k x seq]
        Matrix scores = Q.matmul(Kt);         // [seq x seq]

        float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
        scores = scores.scale(scale_factor);

        if (causal) {
            apply_causal_mask(scores);
        }

        softmax_matrix_rows(scores);          // now "scores" holds attention weights

        Matrix output = scores.matmul(V);     // [seq x d_k]
        return output;
    }
};
