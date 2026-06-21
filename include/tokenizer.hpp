#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <utility>

// ============================================================
// Tokenizer: a simplified byte-pair-encoding (BPE) tokenizer.
//
// Real GPT-2 BPE operates on UTF-8 bytes remapped to a printable
// alphabet. For learning purposes this version operates on
// whitespace-split "words" and merges character pairs within each
// word -- same *algorithm*, simplified alphabet, so the mechanics
// you implement here transfer directly to a full byte-level BPE
// (the upgrade path is noted in comments below).
// ============================================================
class Tokenizer {
public:
    // vocab: token string -> integer id
    std::unordered_map<std::string, int> vocab;
    // id -> token string (for decoding)
    std::unordered_map<int, std::string> id_to_token;
    // merge rules in priority order: (left, right) -> learned earlier = higher priority
    std::vector<std::pair<std::string, std::string>> merges;
    // quick lookup: merge pair -> its rank (lower rank = applied first)
    std::unordered_map<std::string, int> merge_rank;

    Tokenizer() = default;

    // ---- Load vocab.json-style file: "token<TAB>id" per line ----
    void load_vocab(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Tokenizer::load_vocab: cannot open file '" + path + "'");
        }
        std::string line;
        int line_no = 0;
        while (std::getline(file, line)) {
            line_no++;
            if (line.empty()) continue;
            size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                throw std::runtime_error(
                    "Tokenizer::load_vocab: malformed line " + std::to_string(line_no) +
                    " in '" + path + "' (expected 'token<TAB>id')");
            }
            std::string token = line.substr(0, tab);
            int id;
            try {
                id = std::stoi(line.substr(tab + 1));
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "Tokenizer::load_vocab: invalid id on line " + std::to_string(line_no));
            }
            vocab[token] = id;
            id_to_token[id] = token;
        }
        if (vocab.empty()) {
            throw std::runtime_error("Tokenizer::load_vocab: vocab file '" + path + "' was empty");
        }
    }

    // ---- Load merges.txt-style file: "left right" per line, in priority order ----
    void load_merges(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Tokenizer::load_merges: cannot open file '" + path + "'");
        }
        std::string line;
        int rank = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string left, right;
            if (!(iss >> left >> right)) continue; // skip malformed lines silently (header lines etc.)
            merges.emplace_back(left, right);
            merge_rank[left + " " + right] = rank++;
        }
    }

    // ---- Core BPE merge loop for a single word ----
    // Splits word into characters, then repeatedly merges the
    // highest-priority adjacent pair until no more merges apply.
    std::vector<std::string> bpe_merge_word(const std::string& word) const {
        if (word.empty()) return {};

        std::vector<std::string> symbols;
        for (char c : word) symbols.push_back(std::string(1, c));

        while (symbols.size() > 1) {
            int best_rank = -1;
            size_t best_idx = 0;
            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                std::string key = symbols[i] + " " + symbols[i + 1];
                auto it = merge_rank.find(key);
                if (it != merge_rank.end()) {
                    if (best_rank == -1 || it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = i;
                    }
                }
            }
            if (best_rank == -1) break; // no applicable merge left

            std::string merged = symbols[best_idx] + symbols[best_idx + 1];
            symbols[best_idx] = merged;
            symbols.erase(symbols.begin() + best_idx + 1);
        }
        return symbols;
    }

    // ---- Encode: text -> token ids ----
    std::vector<int> encode(const std::string& text) const {
        if (vocab.empty()) {
            throw std::runtime_error("Tokenizer::encode: vocab not loaded, call load_vocab() first");
        }
        std::vector<int> ids;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            auto pieces = bpe_merge_word(word);
            for (const auto& piece : pieces) {
                auto it = vocab.find(piece);
                if (it != vocab.end()) {
                    ids.push_back(it->second);
                } else {
                    // Unknown subword: fall back to an <unk> token if present,
                    // otherwise this is a hard error -- silent skipping would
                    // desync token positions from what the model expects.
                    auto unk = vocab.find("<unk>");
                    if (unk != vocab.end()) {
                        ids.push_back(unk->second);
                    } else {
                        throw std::runtime_error(
                            "Tokenizer::encode: unknown token piece '" + piece +
                            "' and no <unk> fallback in vocab");
                    }
                }
            }
        }
        return ids;
    }

    // ---- Decode: token ids -> text ----
    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (size_t i = 0; i < ids.size(); i++) {
            auto it = id_to_token.find(ids[i]);
            if (it == id_to_token.end()) {
                throw std::out_of_range(
                    "Tokenizer::decode: id " + std::to_string(ids[i]) + " not found in vocab");
            }
            if (i > 0) out += " "; // simplified word-level spacing
            out += it->second;
        }
        return out;
    }

    size_t vocab_size() const { return vocab.size(); }
};

// UPGRADE PATH (not required for this project, mentioned for completeness):
// real GPT-2 BPE remaps every byte 0-255 to a printable unicode char first,
// so it never hits an unknown-token error -- any byte sequence is encodable.
// That's a ~30 line addition once this version works.
