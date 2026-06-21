#pragma once
#include "matrix.hpp"
#include "embedding.hpp"
#include "transformer_block.hpp"
#include "weight_loader.hpp"
#include "tokenizer.hpp"
#include <vector>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <algorithm>

// ============================================================
// Model: wires together embeddings + N transformer blocks +
// a final projection to vocabulary logits, and exposes
// generate() for greedy autoregressive decoding.
// ============================================================
class Model {
public:
    int vocab_size;
    int d_model;
    int d_k;
    int d_hidden;
    int max_seq_len;
    int num_layers;

    Embedding token_emb;
    PositionalEncoding pos_enc;
    std::vector<TransformerBlock> blocks;
    LayerNorm final_ln;
    Matrix lm_head; // [d_model x vocab_size] -- projects to next-token logits

    Model(int vocab_size_, int d_model_, int d_k_, int d_hidden_,
          int max_seq_len_, int num_layers_)
        : vocab_size(vocab_size_), d_model(d_model_), d_k(d_k_),
          d_hidden(d_hidden_), max_seq_len(max_seq_len_), num_layers(num_layers_),
          token_emb(vocab_size_, d_model_),
          pos_enc(max_seq_len_, d_model_),
          final_ln(d_model_),
          lm_head(d_model_, vocab_size_) {
        if (d_k_ != d_model_) {
            // See TransformerBlock note: this teaching version uses a single
            // head whose output width must equal d_model for the residual
            // add to be shape-valid. (Real multi-head models concatenate
            // h heads of width d_model/h back to d_model -- noted as a
            // stretch goal in the README.)
            throw std::invalid_argument(
                "Model: this single-head version requires d_k == d_model "
                "(got d_k=" + std::to_string(d_k_) + ", d_model=" + std::to_string(d_model_) + ")");
        }
        for (int i = 0; i < num_layers_; i++) {
            blocks.emplace_back(d_model_, d_k_, d_hidden_);
        }
    }

    // Runs the full forward pass and returns logits for EVERY
    // position: [seq_len x vocab_size]. For generation we only
    // need the last row, but returning all rows keeps this
    // function reusable (e.g. for computing loss during training,
    // outside the scope of this project but good practice).
    Matrix forward(const std::vector<int>& token_ids) const {
        if (token_ids.empty()) {
            throw std::invalid_argument("Model::forward: token_ids is empty");
        }
        if (static_cast<int>(token_ids.size()) > max_seq_len) {
            throw std::invalid_argument(
                "Model::forward: sequence length " + std::to_string(token_ids.size()) +
                " exceeds max_seq_len " + std::to_string(max_seq_len));
        }

        Matrix x = token_emb.lookup_sequence(token_ids);
        x = pos_enc.add_to(x);

        for (const auto& block : blocks) {
            x = block.forward(x, /*causal=*/true);
        }

        x = final_ln.forward(x);
        Matrix logits = x.matmul(lm_head); // [seq_len x vocab_size]
        return logits;
    }

    // Greedy decoding: repeatedly picks the argmax next token and
    // appends it, until max_new_tokens is reached or max_seq_len hit.
    std::vector<int> generate(std::vector<int> prompt_ids, int max_new_tokens) const {
        if (prompt_ids.empty()) {
            throw std::invalid_argument("Model::generate: prompt_ids is empty");
        }
        if (max_new_tokens < 0) {
            throw std::invalid_argument("Model::generate: max_new_tokens must be >= 0");
        }

        std::vector<int> sequence = prompt_ids;

        for (int step = 0; step < max_new_tokens; step++) {
            if (static_cast<int>(sequence.size()) >= max_seq_len) {
                break; // hit context window limit, stop generating
            }

            Matrix logits = forward(sequence);          // [seq x vocab_size]
            const float* last_row = logits.row_ptr(logits.rows - 1);

            int best_id = 0;
            float best_score = last_row[0];
            for (int v = 1; v < vocab_size; v++) {
                if (last_row[v] > best_score) {
                    best_score = last_row[v];
                    best_id = v;
                }
            }
            sequence.push_back(best_id);
        }
        return sequence;
    }
};
