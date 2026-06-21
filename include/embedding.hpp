#pragma once
#include "matrix.hpp"
#include <stdexcept>

// ============================================================
// Embedding: lookup table mapping token id -> dense vector.
// This is NOT computed -- it's a direct memory read of one row
// out of a [vocab_size x d_model] matrix.
// ============================================================
class Embedding {
public:
    int vocab_size;
    int d_model;
    Matrix table; // [vocab_size x d_model]

    Embedding(int vocab_size_, int d_model_)
        : vocab_size(vocab_size_), d_model(d_model_), table(vocab_size_, d_model_) {}

    // Look up a single token's embedding vector.
    std::vector<float> lookup(int token_id) const {
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::out_of_range(
                "Embedding::lookup: token_id " + std::to_string(token_id) +
                " out of range [0, " + std::to_string(vocab_size) + ")");
        }
        std::vector<float> vec(d_model);
        const float* row = table.row_ptr(token_id);
        std::copy(row, row + d_model, vec.begin());
        return vec;
    }

    // Look up a whole sequence of token ids -> [seq_len x d_model] matrix.
    Matrix lookup_sequence(const std::vector<int>& token_ids) const {
        if (token_ids.empty()) {
            throw std::invalid_argument("Embedding::lookup_sequence: token_ids is empty");
        }
        Matrix result(static_cast<int>(token_ids.size()), d_model);
        for (size_t i = 0; i < token_ids.size(); i++) {
            auto vec = lookup(token_ids[i]); // bounds-checked above
            for (int j = 0; j < d_model; j++) {
                result.at(static_cast<int>(i), j) = vec[j];
            }
        }
        return result;
    }
};

// ============================================================
// PositionalEncoding: a second lookup table, [max_seq_len x d_model],
// added element-wise to token embeddings so the model can tell
// token order apart. GPT-2 uses LEARNED positions (loaded from
// weights), not the sinusoidal formula from the original "Attention
// Is All You Need" paper -- so this class is just another Embedding,
// reused for clarity of intent in calling code.
// ============================================================
class PositionalEncoding {
public:
    int max_seq_len;
    int d_model;
    Matrix table; // [max_seq_len x d_model]

    PositionalEncoding(int max_seq_len_, int d_model_)
        : max_seq_len(max_seq_len_), d_model(d_model_), table(max_seq_len_, d_model_) {}

    // Add positional vectors [0..seq_len) to an existing embedding matrix.
    Matrix add_to(const Matrix& token_embeddings) const {
        int seq_len = token_embeddings.rows;
        if (seq_len > max_seq_len) {
            throw std::invalid_argument(
                "PositionalEncoding::add_to: sequence length " + std::to_string(seq_len) +
                " exceeds max_seq_len " + std::to_string(max_seq_len) +
                " -- this model was not trained on sequences this long");
        }
        if (token_embeddings.cols != d_model) {
            throw std::invalid_argument(
                "PositionalEncoding::add_to: d_model mismatch (" +
                std::to_string(token_embeddings.cols) + " vs " + std::to_string(d_model) + ")");
        }
        Matrix result(seq_len, d_model);
        for (int pos = 0; pos < seq_len; pos++) {
            for (int j = 0; j < d_model; j++) {
                result.at(pos, j) = token_embeddings.at(pos, j) + table.at(pos, j);
            }
        }
        return result;
    }
};
