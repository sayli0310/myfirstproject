"""
generate_demo_vocab.py
Generates a tiny demo vocab.tsv + merges.txt so main.cpp is runnable
out of the box, without needing real GPT-2 files for a first test run.

Run: python3 generate_demo_vocab.py
"""
import itertools

words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "and", "slow"]
chars = sorted(set("".join(words) + " "))

vocab_lines = []
idx = 0
for c in chars:
    vocab_lines.append(f"{c}\t{idx}")
    idx += 1
vocab_lines.append(f"<unk>\t{idx}")
idx += 1

merges = []
for w in words:
    pieces = list(w)
    while len(pieces) > 1:
        merges.append((pieces[0], pieces[1]))
        merged = pieces[0] + pieces[1]
        pieces = [merged] + pieces[2:]
    vocab_lines.append(f"{w}\t{idx}")
    idx += 1

seen = set()
unique_merges = []
for m in merges:
    if m not in seen:
        seen.add(m)
        unique_merges.append(m)

with open("weights/vocab.tsv", "w") as f:
    f.write("\n".join(vocab_lines) + "\n")

with open("weights/merges.txt", "w") as f:
    for a, b in unique_merges:
        f.write(f"{a} {b}\n")

print(f"Wrote {len(vocab_lines)} vocab entries and {len(unique_merges)} merge rules to weights/")
