// benchmark.cpp
// Produces the throughput/latency numbers for your README's
// benchmarks/ section -- this is what recruiters actually look at.
//
// Compile:  g++ -std=c++17 -O3 -Iinclude src/benchmark.cpp -o benchmark
// Run:      ./benchmark

#include "../include/matrix.hpp"
#include "../include/attention.hpp"
#include "../include/model.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& x : v) x = dist(rng);
}

int main() {
    std::mt19937 rng(42);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== Mini LLM inference engine -- benchmarks ===\n\n";

    // ---- 1. Raw matmul throughput at a few sizes ----
    std::cout << "-- matmul throughput --\n";
    for (int size : {64, 128, 256, 512}) {
        Matrix a(size, size), b(size, size);
        fill_random(a.data, rng);
        fill_random(b.data, rng);

        auto start = Clock::now();
        int iters = (size <= 128) ? 20 : 5;
        for (int i = 0; i < iters; i++) {
            Matrix c = a.matmul(b);
        }
        double total_ms = ms_since(start);
        double per_call = total_ms / iters;
        double flops = 2.0 * size * size * size; // multiply+add per element
        double gflops_per_sec = (flops / (per_call / 1000.0)) / 1e9;

        std::cout << "  " << size << "x" << size << " . " << size << "x" << size
                  << " : " << per_call << " ms/call, "
                  << gflops_per_sec << " GFLOP/s\n";
    }

    // ---- 2. Attention head latency vs sequence length ----
    std::cout << "\n-- attention head latency (d_model=64) --\n";
    for (int seq_len : {8, 32, 64, 128}) {
        AttentionHead head(64, 64);
        fill_random(head.Wq.data, rng);
        fill_random(head.Wk.data, rng);
        fill_random(head.Wv.data, rng);

        Matrix x(seq_len, 64);
        fill_random(x.data, rng);

        auto start = Clock::now();
        int iters = 50;
        for (int i = 0; i < iters; i++) {
            Matrix out = head.forward(x);
        }
        double per_call = ms_since(start) / iters;
        std::cout << "  seq_len=" << seq_len << " : " << per_call << " ms/call\n";
    }

    // ---- 3. Full model: end-to-end generation throughput ----
    std::cout << "\n-- full model generation throughput --\n";
    int vocab_size = 1000, d_model = 64, d_k = 64, d_hidden = 256, max_seq_len = 128, num_layers = 6;
    Model model(vocab_size, d_model, d_k, d_hidden, max_seq_len, num_layers);
    fill_random(model.token_emb.table.data, rng);
    fill_random(model.pos_enc.table.data, rng);
    fill_random(model.lm_head.data, rng);
    for (auto& block : model.blocks) {
        fill_random(block.attn.Wq.data, rng);
        fill_random(block.attn.Wk.data, rng);
        fill_random(block.attn.Wv.data, rng);
        fill_random(block.ffn.W1.data, rng);
        fill_random(block.ffn.W2.data, rng);
    }

    std::vector<int> prompt = {1, 2, 3, 4, 5};
    for (int new_tokens : {10, 30, 50}) {
        auto start = Clock::now();
        auto result = model.generate(prompt, new_tokens);
        double total_ms = ms_since(start);
        double tok_per_sec = new_tokens / (total_ms / 1000.0);
        std::cout << "  generate " << new_tokens << " tokens (6 layers, d_model=64): "
                  << total_ms << " ms total, " << tok_per_sec << " tokens/sec\n";
    }

    std::cout << "\nNote: numbers are from a teaching-scale model on this machine,\n"
                 "not representative of production model sizes. Re-run on your\n"
                 "target hardware and report exact specs in your README.\n";

    return 0;
}
