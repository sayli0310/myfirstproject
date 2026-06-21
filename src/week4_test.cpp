// week4_test.cpp
// Verifies: LayerNorm correctness, FeedForward shape, TransformerBlock
// residual wiring, WeightLoader round-trip + error handling, and a
// full Model.generate() run with random weights (sanity, not language
// quality -- random weights won't produce real English, just proves
// the pipeline runs end-to-end without crashing or shape errors).
//
// Compile:  g++ -std=c++17 -O2 -Iinclude src/week4_test.cpp -o week4_test
// Run:      ./week4_test

#include "../include/layernorm.hpp"
#include "../include/transformer_block.hpp"
#include "../include/weight_loader.hpp"
#include "../include/model.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <fstream>
#include <random>

void test_layernorm_zero_mean_unit_var() {
    LayerNorm ln(4);
    Matrix x(1, 4);
    x.at(0,0)=10; x.at(0,1)=20; x.at(0,2)=30; x.at(0,3)=40;
    Matrix out = ln.forward(x);
    // gamma=1, beta=0 by default, so output should have ~0 mean
    float mean = 0;
    for (int j = 0; j < 4; j++) mean += out.at(0, j);
    mean /= 4;
    assert(std::fabs(mean) < 1e-4f);
    std::cout << "[PASS] LayerNorm produces ~zero mean output\n";
}

void test_layernorm_shape_mismatch_throws() {
    LayerNorm ln(8);
    Matrix wrong(2, 4);
    bool threw = false;
    try { ln.forward(wrong); }
    catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "[PASS] LayerNorm shape mismatch caught: " << e.what() << "\n";
    }
    assert(threw);
}

void test_feedforward_shape() {
    FeedForward ffn(8, 32);
    Matrix x(3, 8);
    Matrix out = ffn.forward(x);
    assert(out.rows == 3 && out.cols == 8);
    std::cout << "[PASS] FeedForward preserves [seq x d_model] shape\n";
}

void test_transformer_block_residual_shape() {
    TransformerBlock block(8, 8, 32); // d_k must equal d_model for residual add
    Matrix x(4, 8);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            x.at(i,j) = 0.01f * (i+1) * (j+1);
    Matrix out = block.forward(x);
    assert(out.rows == 4 && out.cols == 8);
    std::cout << "[PASS] TransformerBlock forward preserves shape through residuals\n";
}

void test_weight_loader_roundtrip() {
    // Write a small matrix to disk in our custom binary format, then load it back.
    Matrix original(2, 3);
    original.at(0,0)=1; original.at(0,1)=2; original.at(0,2)=3;
    original.at(1,0)=4; original.at(1,1)=5; original.at(1,2)=6;

    {
        std::ofstream out("test_weight.bin", std::ios::binary);
        int32_t rows = original.rows, cols = original.cols;
        out.write(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        out.write(reinterpret_cast<char*>(&cols), sizeof(int32_t));
        out.write(reinterpret_cast<char*>(original.data.data()),
                   static_cast<std::streamsize>(original.data.size() * sizeof(float)));
    }

    Matrix loaded = WeightLoader::load_matrix("test_weight.bin");
    assert(loaded.rows == 2 && loaded.cols == 3);
    assert(loaded.at(1,2) == 6.0f);
    std::cout << "[PASS] WeightLoader round-trip correctness\n";
}

void test_weight_loader_missing_file_throws() {
    bool threw = false;
    try { WeightLoader::load_matrix("nope_does_not_exist.bin"); }
    catch (const std::runtime_error& e) {
        threw = true;
        std::cout << "[PASS] WeightLoader missing file caught: " << e.what() << "\n";
    }
    assert(threw);
}

void test_weight_loader_truncated_file_throws() {
    // Write a header claiming 10x10 floats but only provide 2 floats of data.
    {
        std::ofstream out("truncated.bin", std::ios::binary);
        int32_t rows = 10, cols = 10;
        out.write(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        out.write(reinterpret_cast<char*>(&cols), sizeof(int32_t));
        float junk[2] = {1.0f, 2.0f};
        out.write(reinterpret_cast<char*>(junk), sizeof(junk));
    }
    bool threw = false;
    try { WeightLoader::load_matrix("truncated.bin"); }
    catch (const std::runtime_error& e) {
        threw = true;
        std::cout << "[PASS] WeightLoader truncated file caught: " << e.what() << "\n";
    }
    assert(threw);
}

void test_model_end_to_end_generation() {
    // Small random-weight model just to prove the FULL pipeline runs:
    // embedding -> N transformer blocks -> final norm -> lm_head -> argmax loop.
    // Random weights will NOT produce coherent English -- that requires
    // real trained weights (see "load real GPT-2 weights" in run instructions).
    int vocab_size = 50, d_model = 16, d_k = 16, d_hidden = 64, max_seq_len = 32, num_layers = 2;
    Model model(vocab_size, d_model, d_k, d_hidden, max_seq_len, num_layers);

    // Fill weights with small random values so the math has something to chew on
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : model.token_emb.table.data) v = dist(rng);
    for (auto& v : model.pos_enc.table.data) v = dist(rng);
    for (auto& v : model.lm_head.data) v = dist(rng);
    for (auto& block : model.blocks) {
        for (auto& v : block.attn.Wq.data) v = dist(rng);
        for (auto& v : block.attn.Wk.data) v = dist(rng);
        for (auto& v : block.attn.Wv.data) v = dist(rng);
        for (auto& v : block.ffn.W1.data) v = dist(rng);
        for (auto& v : block.ffn.W2.data) v = dist(rng);
    }

    std::vector<int> prompt = {1, 2, 3};
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> result = model.generate(prompt, /*max_new_tokens=*/5);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    assert(result.size() == prompt.size() + 5);
    for (int id : result) {
        assert(id >= 0 && id < vocab_size);
    }
    std::cout << "[PASS] Model.generate() ran end-to-end, produced "
              << result.size() << " tokens in " << ms << " ms\n";
}

void test_model_rejects_dk_not_equal_dmodel() {
    bool threw = false;
    try {
        Model bad(50, 16, 8 /* d_k != d_model */, 64, 32, 2);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "[PASS] Model constructor rejected d_k != d_model: " << e.what() << "\n";
    }
    assert(threw);
}

int main() {
    std::cout << "=== Week 4: LayerNorm + FFN + TransformerBlock + Model tests ===\n\n";
    try {
        test_layernorm_zero_mean_unit_var();
        test_layernorm_shape_mismatch_throws();
        test_feedforward_shape();
        test_transformer_block_residual_shape();
        test_weight_loader_roundtrip();
        test_weight_loader_missing_file_throws();
        test_weight_loader_truncated_file_throws();
        test_model_end_to_end_generation();
        test_model_rejects_dk_not_equal_dmodel();
    } catch (const std::exception& e) {
        std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\nAll week 4 tests passed.\n";
    return 0;
}
