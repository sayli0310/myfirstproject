// week3_test.cpp
// Verifies: softmax correctness + numerical stability, causal masking,
// full AttentionHead forward pass and its error handling.
//
// Compile:  g++ -std=c++17 -O2 -Iinclude src/week3_test.cpp -o week3_test
// Run:      ./week3_test

#include "../include/attention.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

void test_softmax_sums_to_one() {
    float row[] = {1.0f, 2.0f, 3.0f};
    softmax_row(row, 3);
    float sum = row[0] + row[1] + row[2];
    assert(std::fabs(sum - 1.0f) < 1e-5f);
    assert(row[2] > row[1] && row[1] > row[0]); // highest input -> highest probability
    std::cout << "[PASS] softmax sums to 1.0 and preserves order\n";
}

void test_softmax_numerical_stability() {
    // Without the subtract-max trick, expf(1000) would overflow to inf
    // and the result would be nan. With the trick, this must work cleanly.
    float row[] = {1000.0f, 1001.0f, 999.0f};
    softmax_row(row, 3);
    for (int i = 0; i < 3; i++) {
        assert(std::isfinite(row[i]));
    }
    std::cout << "[PASS] softmax stable on large inputs (no overflow)\n";
}

void test_causal_mask_zeroes_future() {
    Matrix scores(3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            scores.at(i, j) = 1.0f; // uniform scores before masking

    apply_causal_mask(scores);
    softmax_matrix_rows(scores);

    // row 0 (token 0) should only attend to itself -> weight ~1.0 at col 0
    assert(std::fabs(scores.at(0, 0) - 1.0f) < 1e-5f);
    assert(scores.at(0, 1) < 1e-5f);
    assert(scores.at(0, 2) < 1e-5f);

    // row 2 (token 2, last position) can see everything -> roughly uniform
    assert(std::fabs(scores.at(2, 0) - scores.at(2, 1)) < 1e-5f);
    std::cout << "[PASS] causal mask blocks future positions correctly\n";
}

void test_causal_mask_requires_square() {
    Matrix bad(3, 5);
    bool threw = false;
    try {
        apply_causal_mask(bad);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "[PASS] non-square mask input caught: " << e.what() << "\n";
    }
    assert(threw);
}

void test_attention_head_forward_shape() {
    int d_model = 8, d_k = 4, seq_len = 5;
    AttentionHead head(d_model, d_k);

    Matrix x(seq_len, d_model);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_model; j++)
            x.at(i, j) = 0.1f * (i + j);

    Matrix out = head.forward(x, /*causal=*/true);
    assert(out.rows == seq_len && out.cols == d_k);
    std::cout << "[PASS] AttentionHead forward produces correct output shape ["
              << out.rows << "x" << out.cols << "]\n";
}

void test_attention_head_dmodel_mismatch_throws() {
    AttentionHead head(8, 4);
    Matrix wrong_input(5, 16); // d_model should be 8, not 16
    bool threw = false;
    try {
        head.forward(wrong_input);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "[PASS] AttentionHead d_model mismatch caught: " << e.what() << "\n";
    }
    assert(threw);
}

int main() {
    std::cout << "=== Week 3: Softmax + Attention tests ===\n\n";
    try {
        test_softmax_sums_to_one();
        test_softmax_numerical_stability();
        test_causal_mask_zeroes_future();
        test_causal_mask_requires_square();
        test_attention_head_forward_shape();
        test_attention_head_dmodel_mismatch_throws();
    } catch (const std::exception& e) {
        std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\nAll week 3 tests passed.\n";
    return 0;
}
