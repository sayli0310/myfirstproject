# Mini LLM Inference Engine (C++)

A transformer-based language model inference engine built from scratch in C++ —
no PyTorch, no TensorFlow, no ML framework dependency. Implements tokenization,
embeddings, scaled dot-product attention with causal masking, layer
normalization, and greedy autoregressive text generation entirely in raw C++.

This project mirrors the architecture of real edge-AI inference engines like
`llama.cpp` at a small, fully-understood scale.

## Why this exists

Production environments without GPUs (mobile devices, embedded systems,
on-premise inference) need models that run without a Python interpreter or
deep learning framework attached. This project implements that inference
layer: load trained weights, run the transformer forward pass, generate text
— in C++, with explicit memory layout and error handling throughout.

## Architecture

```
Input text
   │
   ▼
Tokenizer (BPE)  ──────────►  token ids
   │
   ▼
Embedding lookup + Positional encoding
   │
   ▼
┌─────────────────────────────┐
│  TransformerBlock  x N      │
│  x = x + Attention(LN(x))   │
│  x = x + FeedForward(LN(x)) │
└─────────────────────────────┘
   │
   ▼
Final LayerNorm
   │
   ▼
Linear projection → vocab logits
   │
   ▼
Greedy decode (argmax) ──► next token
```

## Build

Requires a C++17 compiler (g++ or clang++), no external dependencies.

```bash
g++ -std=c++17 -O2 -Iinclude src/main.cpp -o llm_engine
g++ -std=c++17 -O3 -Iinclude src/benchmark.cpp -o benchmark
```

## Run

First generate a small demo vocabulary (only needed once):

```bash
python3 generate_demo_vocab.py
```

Then run the engine on a prompt:

```bash
./llm_engine "the cat sat on the"
```

Expected output: token IDs, decoded text, and tokens/sec throughput. Since
this demo uses randomly-initialized weights (no real GPT-2 weights are
bundled), generated text will not be coherent English — this proves the
*engine* is correct, not that it has learned language. See "Loading real
weights" below to fix that.

## Run the test suite (one set of tests per week of development)

```bash
g++ -std=c++17 -O2 -Iinclude src/week1_test.cpp -o week1_test && ./week1_test
g++ -std=c++17 -O2 -Iinclude src/week2_test.cpp -o week2_test && ./week2_test
g++ -std=c++17 -O2 -Iinclude src/week3_test.cpp -o week3_test && ./week3_test
g++ -std=c++17 -O2 -Iinclude src/week4_test.cpp -o week4_test && ./week4_test
```

37 tests total across the 4 weeks, covering correctness and error handling
(shape mismatches, missing files, truncated files, out-of-range indices).

## Run benchmarks

```bash
./benchmark
```

See `benchmarks/results.md` for recorded numbers and methodology.

## Project structure

```
include/
  matrix.hpp            Core Matrix class: matmul, dot, transpose, add
  tokenizer.hpp          BPE tokenizer: encode/decode
  embedding.hpp           Token + positional embedding lookup
  attention.hpp            Softmax, causal masking, AttentionHead
  layernorm.hpp             LayerNorm, GELU, FeedForward (MLP)
  transformer_block.hpp      Full transformer block (attention + MLP + residuals)
  weight_loader.hpp            Custom binary weight format loader
  model.hpp                      Full model + generate() (greedy decoding)
src/
  week1_test.cpp ... week4_test.cpp   Test suite, one file per build week
  main.cpp                             CLI entry point
  benchmark.cpp                         Throughput/latency benchmarks
weights/
  vocab.tsv, merges.txt                 Tokenizer files
benchmarks/
  results.md                             Recorded benchmark output
```

## Loading real GPT-2 weights (extension)

This project ships with random weights so the pipeline is runnable
out-of-the-box. To load real GPT-2 weights:

1. Use a short Python script with `transformers` to load `gpt2` and dump
   each weight matrix to this project's binary format (two int32s for
   shape, followed by row-major float32 data) using `WeightLoader`'s format.
2. Convert GPT-2's `vocab.json`/`merges.txt` to this project's tab-separated
   format expected by `Tokenizer::load_vocab` / `load_merges`.
3. Set `d_model=768, num_layers=12, d_hidden=3072` to match GPT-2 small.
4. Note: this teaching version implements single-head attention requiring
   `d_k == d_model`. Real GPT-2 uses 12 heads of width 64 concatenated to
   768 — extending `AttentionHead` to `MultiHeadAttention` (running N heads
   on slices and concatenating outputs) is the natural next step, and is a
   strong "what I'd build next" talking point in an interview.

## What I'd build next (stretch goals)

- Multi-head attention (currently single-head)
- KV-caching to avoid recomputing attention for already-generated tokens
  (the single biggest real-world inference speedup)
- INT8 weight quantization for memory/speed
- Real GGUF file format parsing (see `weight_loader.hpp` comments for the
  upgrade path)
- Top-k / temperature sampling instead of pure greedy decoding
