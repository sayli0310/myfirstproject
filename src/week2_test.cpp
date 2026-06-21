// week2_test.cpp
// Verifies: tokenizer loading + error handling, BPE merge logic,
// encode/decode round trip, embedding lookup, positional encoding.
//
// Compile:  g++ -std=c++17 -O2 -Iinclude src/week2_test.cpp -o week2_test
// Run:      ./week2_test

#include "../include/tokenizer.hpp"
#include "../include/embedding.hpp"
#include <cassert>
#include <iostream>
#include <fstream>

// Writes a tiny throwaway vocab + merges file so the test is self-contained.
void write_test_files() {
    std::ofstream vocab("test_vocab.tsv");
    // single chars first (BPE always falls back to these)
    const std::string chars = "cathsondm ";
    int id = 0;
    for (char c : chars) vocab << std::string(1, c) << "\t" << id++ << "\n";
    vocab << "<unk>\t" << id++ << "\n";
    // merged subwords that our merges.txt will produce
    vocab << "ca\t" << id++ << "\n";
    vocab << "cat\t" << id++ << "\n";
    vocab << "ma\t" << id++ << "\n";
    vocab << "mat\t" << id++ << "\n";
    vocab.close();

    std::ofstream merges("test_merges.txt");
    merges << "c a\n";   // c + a -> ca
    merges << "ca t\n";  // ca + t -> cat
    merges << "m a\n";   // m + a -> ma
    merges << "ma t\n";  // ma + t -> mat
    merges.close();
}

void test_load_missing_vocab_throws() {
    Tokenizer t;
    bool threw = false;
    try {
        t.load_vocab("this_file_does_not_exist.tsv");
    } catch (const std::runtime_error& e) {
        threw = true;
        std::cout << "[PASS] missing vocab file caught: " << e.what() << "\n";
    }
    assert(threw);
}

void test_bpe_merge_and_encode() {
    Tokenizer t;
    t.load_vocab("test_vocab.tsv");
    t.load_merges("test_merges.txt");

    auto pieces = t.bpe_merge_word("cat");
    assert(pieces.size() == 1 && pieces[0] == "cat");
    std::cout << "[PASS] BPE merged 'cat' -> single token\n";

    auto ids = t.encode("cat mat");
    assert(ids.size() == 2);
    std::cout << "[PASS] encode produced " << ids.size() << " token ids\n";

    std::string decoded = t.decode(ids);
    assert(decoded == "cat mat");
    std::cout << "[PASS] decode round-trip: \"" << decoded << "\"\n";
}

void test_decode_unknown_id_throws() {
    Tokenizer t;
    t.load_vocab("test_vocab.tsv");
    bool threw = false;
    try {
        t.decode({99999});
    } catch (const std::out_of_range& e) {
        threw = true;
        std::cout << "[PASS] decode unknown id caught: " << e.what() << "\n";
    }
    assert(threw);
}

void test_embedding_lookup() {
    Embedding emb(/*vocab_size=*/100, /*d_model=*/8);
    // manually set a known row so we can verify lookup correctness
    for (int j = 0; j < 8; j++) emb.table.at(5, j) = static_cast<float>(j);

    auto vec = emb.lookup(5);
    assert(vec.size() == 8 && vec[3] == 3.0f);
    std::cout << "[PASS] embedding lookup correctness\n";

    bool threw = false;
    try {
        emb.lookup(500); // out of range
    } catch (const std::out_of_range&) {
        threw = true;
        std::cout << "[PASS] embedding out-of-range id caught\n";
    }
    assert(threw);
}

void test_positional_encoding() {
    Embedding emb(100, 8);
    PositionalEncoding pos(/*max_seq_len=*/16, /*d_model=*/8);

    std::vector<int> ids = {1, 2, 3};
    Matrix tok_emb = emb.lookup_sequence(ids);
    Matrix combined = pos.add_to(tok_emb);
    assert(combined.rows == 3 && combined.cols == 8);
    std::cout << "[PASS] positional encoding shape correct\n";

    bool threw = false;
    try {
        std::vector<int> too_long(100, 1); // exceeds max_seq_len=16
        Matrix big = emb.lookup_sequence(too_long);
        pos.add_to(big);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "[PASS] sequence-too-long caught: " << e.what() << "\n";
    }
    assert(threw);
}

int main() {
    std::cout << "=== Week 2: Tokenizer + Embedding tests ===\n\n";
    write_test_files();
    try {
        test_load_missing_vocab_throws();
        test_bpe_merge_and_encode();
        test_decode_unknown_id_throws();
        test_embedding_lookup();
        test_positional_encoding();
    } catch (const std::exception& e) {
        std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\nAll week 2 tests passed.\n";
    return 0;
}
