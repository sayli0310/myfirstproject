#pragma once
#include "matrix.hpp"
#include "attention.hpp"
#include "layernorm.hpp"
#include <stdexcept>

// ============================================================
// TransformerBlock: one full layer of the model. GPT-2 small
// stacks 12 of these.
//
//   x = x + Attention(LayerNorm(x))
//   x = x + FeedForward(LayerNorm(x))
//
// This is "Pre-LN" ordering (norm BEFORE the sublayer), which is
// what GPT-2 and most modern decoder-only models use because it
// trains more stably than the original "Post-LN" Transformer paper.
// ============================================================
class TransformerBlock {
public:
    int d_model;
    AttentionHead attn;
    FeedForward ffn;
    LayerNorm ln1; // before attention
    LayerNorm ln2; // before feed-forward

    TransformerBlock(int d_model_, int d_k_, int d_hidden_)
        : d_model(d_model_),
          attn(d_model_, d_k_),
          ffn(d_model_, d_hidden_),
          ln1(d_model_),
          ln2(d_model_) {}

    Matrix forward(const Matrix& x, bool causal = true) const {
        if (x.cols != d_model) {
            throw std::invalid_argument(
                "TransformerBlock::forward: d_model mismatch (" +
                std::to_string(x.cols) + " vs " + std::to_string(d_model) + ")");
        }

        // --- Attention sub-layer with residual ---
        Matrix normed1 = ln1.forward(x);
        Matrix attn_out = attn.forward(normed1, causal);
        // NOTE: attn_out has cols = d_k, not d_model, unless d_k == d_model.
        // In a full multi-head implementation you'd concatenate heads back
        // to d_model and apply an output projection here. For this
        // single-head teaching version we require d_k == d_model so the
        // residual add is shape-valid -- enforced in the constructor caller.
        Matrix x1 = x.add(attn_out);

        // --- Feed-forward sub-layer with residual ---
        Matrix normed2 = ln2.forward(x1);
        Matrix ffn_out = ffn.forward(normed2);
        Matrix x2 = x1.add(ffn_out);

        return x2;
    }
};
