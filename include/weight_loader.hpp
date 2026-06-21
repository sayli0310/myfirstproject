#pragma once
#include "matrix.hpp"
#include <fstream>
#include <stdexcept>
#include <string>

// ============================================================
// WeightLoader: reads a simple custom binary format.
//
// File layout (all little-endian):
//   [int32] rows
//   [int32] cols
//   [float32 x rows*cols] data, row-major
//
// This is intentionally simple so YOU can export weights from
// Python with a five-line script (see the README / run instructions
// at the end of this lesson) without needing a full GGUF parser.
// The upgrade path to real GGUF is noted at the end of this file.
// ============================================================
class WeightLoader {
public:
    static Matrix load_matrix(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("WeightLoader::load_matrix: cannot open '" + path + "'");
        }

        int32_t rows = 0, cols = 0;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));

        if (!file.good()) {
            throw std::runtime_error("WeightLoader::load_matrix: failed reading header from '" + path + "'");
        }
        if (rows <= 0 || cols <= 0) {
            throw std::runtime_error(
                "WeightLoader::load_matrix: invalid shape (" + std::to_string(rows) +
                "x" + std::to_string(cols) + ") in '" + path + "'");
        }

        Matrix m(rows, cols);
        size_t expected_floats = static_cast<size_t>(rows) * cols;
        file.read(reinterpret_cast<char*>(m.data.data()),
                   static_cast<std::streamsize>(expected_floats * sizeof(float)));

        size_t actually_read = static_cast<size_t>(file.gcount()) / sizeof(float);
        if (actually_read != expected_floats) {
            throw std::runtime_error(
                "WeightLoader::load_matrix: truncated file '" + path + "' -- expected " +
                std::to_string(expected_floats) + " floats, got " + std::to_string(actually_read));
        }

        return m;
    }

    // Loads a flat vector (used for biases, gamma, beta)
    static std::vector<float> load_vector(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("WeightLoader::load_vector: cannot open '" + path + "'");
        }
        int32_t len = 0;
        file.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
        if (!file.good() || len <= 0) {
            throw std::runtime_error("WeightLoader::load_vector: invalid header in '" + path + "'");
        }
        std::vector<float> v(len);
        file.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(len * sizeof(float)));
        size_t actually_read = static_cast<size_t>(file.gcount()) / sizeof(float);
        if (actually_read != static_cast<size_t>(len)) {
            throw std::runtime_error(
                "WeightLoader::load_vector: truncated file '" + path + "'");
        }
        return v;
    }
};

// UPGRADE PATH (stretch goal): real GGUF files store a key-value
// metadata header (model hyperparameters, tokenizer vocab) followed
// by named, quantized tensors. To read real GGUF you'd replace this
// fixed two-int header with GGUF's magic-number + version + tensor
// table parsing. The Matrix/Attention/TransformerBlock code above
// does NOT need to change -- only this loader does. That separation
// is itself a good resume talking point: "decoupled tensor math from
// the weight format so the loader could be swapped independently."
