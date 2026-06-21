// main.cpp
// The CLI entry point: loads vocab + a random-initialized model
// (random weights, since training real GPT-2 weights is outside
// scope -- see README "load real GPT-2 weights" for the upgrade
// path), tokenizes a prompt, and runs greedy generation.
//
// Compile:  g++ -std=c++17 -O2 -Iinclude src/main.cpp -o llm_engine
// Run:      ./llm_engine "the cat sat on the"

#include "../include/tokenizer.hpp"
#include "../include/model.hpp"
#include <iostream>
#include <random>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " \"<prompt text>\"\n";
        std::cerr << "Example: " << argv[0] << " \"the cat sat on the\"\n";
        return 1;
    }
    std::string prompt_text = argv[1];

    try {
        // --- Load tokenizer ---
        // NOTE: replace these paths with real GPT-2 vocab.json/merges.txt
        // (converted to this tool's tab-separated format) once you've
        // followed the "load real GPT-2 weights" steps in the README.
        Tokenizer tok;
        tok.load_vocab("weights/vocab.tsv");
        tok.load_merges("weights/merges.txt");

        std::cout << "Loaded vocab: " << tok.vocab_size() << " tokens\n";

        // --- Build model (random weights for this teaching version) ---
        int vocab_size = static_cast<int>(tok.vocab_size());
        int d_model = 32, d_k = 32, d_hidden = 128, max_seq_len = 64, num_layers = 4;

        Model model(vocab_size, d_model, d_k, d_hidden, max_seq_len, num_layers);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        auto init = [&](std::vector<float>& v) { for (auto& x : v) x = dist(rng); };
        init(model.token_emb.table.data);
        init(model.pos_enc.table.data);
        init(model.lm_head.data);
        for (auto& block : model.blocks) {
            init(block.attn.Wq.data);
            init(block.attn.Wk.data);
            init(block.attn.Wv.data);
            init(block.ffn.W1.data);
            init(block.ffn.W2.data);
        }

        std::cout << "Model: " << num_layers << " layers, d_model=" << d_model << "\n";

        // --- Tokenize prompt ---
        std::vector<int> prompt_ids = tok.encode(prompt_text);
        if (prompt_ids.empty()) {
            std::cerr << "Error: prompt encoded to zero tokens\n";
            return 1;
        }
        std::cout << "Prompt tokens: " << prompt_ids.size() << "\n";

        // --- Generate with timing for benchmark numbers ---
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> result = model.generate(prompt_ids, /*max_new_tokens=*/10);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        int new_tokens = static_cast<int>(result.size() - prompt_ids.size());

        std::cout << "\nGenerated " << new_tokens << " tokens in " << ms << " ms"
                  << " (" << (new_tokens / (ms / 1000.0)) << " tokens/sec)\n";
        std::cout << "Output ids: ";
        for (int id : result) std::cout << id << " ";
        std::cout << "\n";

        // Decode only succeeds if every generated id is in vocab --
        // with random weights + small vocab this is just a sanity decode.
        try {
            std::cout << "Decoded: " << tok.decode(result) << "\n";
        } catch (const std::exception& e) {
            std::cout << "(decode skipped: " << e.what() << ")\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
