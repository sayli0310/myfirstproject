# Benchmark results

Measured with `benchmark.exe` (compiled with `-O3`) on Windows.

## Hardware

```
CPU:              Intel(R) Core(TM) i5-10210U @ 1.60GHz (boost up to 4.20GHz)
Cores:            4 cores / 8 threads
OS:               Windows
Build flags:      -std=c++17 -O3
Threading:        single-threaded (no multithreading used)
SIMD:             none (no manual vectorization / intrinsics)
```

Note: the i5-10210U is a low-power "U-series" laptop chip (15W TDP),
base clock 1.6GHz. This explains the modest GFLOP/s figures below —
they reflect this specific power-efficient mobile CPU running fully
single-threaded code, not a ceiling on what the algorithm can achieve.
A desktop CPU or enabling multithreading would show meaningfully
higher throughput on identical code.

## Methodology

- Matmul: averaged over multiple iterations (20 for small sizes, 5 for
  large, to keep total run time reasonable) at each square size.
- Attention: single head, `d_model=64`, averaged over 50 forward passes.
- Full model generation: 6-layer model, `d_model=64`, `vocab_size=1000`,
  timed end-to-end including the full token-by-token greedy decode loop.
- All single-threaded, no SIMD intrinsics, `-O3` optimization only.

## Matmul throughput

| Size              | Time/call  | Throughput   |
|-------------------|------------|--------------|
| 64x64 . 64x64     | 1.170 ms   | 0.448 GFLOP/s |
| 128x128 . 128x128 | 5.279 ms   | 0.794 GFLOP/s |
| 256x256 . 256x256 | 17.570 ms  | 1.910 GFLOP/s |
| 512x512 . 512x512 | 147.814 ms | 1.816 GFLOP/s |

**Observation:** throughput climbs from 64x64 to 256x256 as larger matrices
better amortize loop overhead, then plateaus/slightly drops at 512x512 —
likely the working set exceeding L2 cache on this CPU. A natural next step
would be loop tiling/blocking to keep more of the computation cache-resident
at larger sizes.

## Attention head latency (d_model=64)

| Sequence length | Latency/call |
|------------------|--------------|
| 8                | 0.202 ms     |
| 32               | 0.858 ms     |
| 64               | 3.198 ms     |
| 128              | 7.455 ms     |

**Observation:** latency grows roughly quadratically with sequence length
(doubling seq_len roughly quadruples latency), which matches theory —
attention's `Q·Kᵀ` step is O(seq_len²) in the number of score computations.
This is the same bottleneck that motivates techniques like FlashAttention
in production engines.

## Full model generation throughput (6 layers, d_model=64)

| New tokens generated | Total time | Throughput      |
|-----------------------|-----------|-----------------|
| 10                     | 52.252 ms | 191.4 tokens/sec |
| 30                     | 344.427 ms| 87.1 tokens/sec  |
| 50                     | 910.435 ms| 54.9 tokens/sec  |

**Observation:** throughput roughly halves each time the generation length
increases — this is the clearest signal in these benchmarks. It happens
because this engine re-runs the full forward pass (including attention
over the *entire* sequence so far) on every new token, with no caching of
previously-computed keys/values. This is the single biggest optimization
gap versus production inference engines like `llama.cpp`, which implement
KV-caching specifically to avoid this quadratic-ish blowup. Implementing
KV-caching is the top item in the "what I'd build next" list.

## Scale context

This is a teaching-scale model, not a production-sized one:

| Parameter   | This benchmark | Real GPT-2 small |
|-------------|----------------|-------------------|
| d_model     | 64             | 768                |
| layers      | 6              | 12                 |
| vocab_size  | 1,000          | ~50,257            |

These numbers describe the *engine's* correctness and scaling behavior,
not what a production-sized model would achieve on this hardware.
