#!/usr/bin/env python3
"""Quantize a llama2.c legacy float32 checkpoint to 8, 6, 4, 2 or 1 bits per weight (QBRW format).

Usage:
  quantize_bits.py MODEL.bin BITS OUT.bin
  quantize_bits.py MODEL.bin --compare TOKENIZER.bin    quality table for every bit width

Each weight row is split into groups of 32 with one float32 scale per group. A code q in
[0, 2^bits - 1] stands for (q - (2^bits - 1) / 2) * scale; each group's scale is the one that
minimizes squared error. Rows whose length isn't a multiple of 32 end in a partial group.

Layout (little-endian, 4-byte aligned):
  b"QBRW", int32 version=1, int32 bits, int32 group=32, Config (7 x int32)
  then the same section order as quantize.py's Q8RW format, where each matrix block is
  float32 scales[d][groups_per_row] followed by codes[d][groups_per_row][4 * bits bytes],
  each group packed as a little-endian bitstream (code k in bits k*bits .. k*bits+bits-1).
"""
import struct
import sys

import numpy as np

import quantize as qz

MAGIC = b"QBRW"
VERSION = 1
GROUP = 32
BITS = (8, 6, 4, 2, 1)
MATRICES = ("emb", "wq", "wk", "wv", "wo", "w1", "w2", "w3")


def quant_groups(m, bits):
    """m: (d, n) -> scales (d, groups) float32, codes (d, groups, GROUP) uint8."""
    d, n = m.shape
    groups = -(-n // GROUP)
    g = np.zeros((d, groups * GROUP), np.float32)
    g[:, :n] = m
    g = g.reshape(d, groups, GROUP)
    valid = (np.arange(groups * GROUP) < n).reshape(groups, GROUP)
    top = 2 ** bits - 1
    half = top / 2
    amax = np.abs(g).max(axis=2)
    best_err = np.full(amax.shape, np.inf)
    best_s = np.ones(amax.shape, np.float32)
    for f in np.linspace(0.1, 1.0, 19):  # try clipping the range; low bit widths gain the most
        s = np.where(amax > 0, amax * f / half, 1.0).astype(np.float32)
        q = np.clip(np.round(g / s[..., None] + half), 0, top)
        err = ((((q - half) * s[..., None] - g) ** 2) * valid).sum(axis=2)
        better = err < best_err
        best_err[better] = err[better]
        best_s[better] = s[better]
    codes = np.clip(np.round(g / best_s[..., None] + half), 0, top).astype(np.uint8)
    return best_s, codes


def dequant(scales, codes, bits, n):
    half = (2 ** bits - 1) / 2
    vals = (codes.astype(np.float32) - half) * scales[..., None]
    return vals.reshape(vals.shape[0], -1)[:, :n]


def pack(codes, bits):
    """(d, groups, GROUP) codes -> (d, groups, 4 * bits) bytes, LSB-first bitstream per group."""
    bitplanes = (codes[..., None] >> np.arange(bits, dtype=np.uint8)) & 1
    stream = bitplanes.reshape(*codes.shape[:2], GROUP * bits)
    return np.packbits(stream, axis=-1, bitorder="little")


def block_bytes(m, bits):
    scales, codes = quant_groups(m, bits)
    return scales.tobytes() + pack(codes, bits).tobytes()


def write_qb(cfg, w, bits, path):
    out = [MAGIC, struct.pack("3i", VERSION, bits, GROUP), struct.pack("7i", *(cfg[k] for k in qz.CONFIG_FIELDS))]
    out.append(block_bytes(w["emb"], bits))
    out.append(w["rms_att"].tobytes())
    for name in ("wq", "wk", "wv", "wo"):
        out += [block_bytes(layer, bits) for layer in w[name]]
    out.append(w["rms_ffn"].tobytes())
    for name in ("w1", "w2", "w3"):
        out += [block_bytes(layer, bits) for layer in w[name]]
    out.append(w["rms_final"].tobytes())
    data = b"".join(out)
    open(path, "wb").write(data)
    return len(data)


def dequantized(w, bits):
    """The weights the board effectively uses at this bit width."""
    dq = dict(w)
    for name in MATRICES:
        m = w[name]
        if m.ndim == 2:  # embedding table
            dq[name] = dequant(*quant_groups(m, bits), bits, m.shape[1])
        else:  # one matrix per layer
            dq[name] = np.stack([dequant(*quant_groups(layer, bits), bits, layer.shape[1]) for layer in m])
    return dq


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def perplexity(cfg, w, stories):
    """exp(mean negative log-likelihood) of each story's tokens under weights w."""
    nll, count = 0.0, 0
    for tokens in stories:
        _, logits = qz.greedy(cfg, w, len(tokens), pick=lambda l, pos: tokens[pos])
        for pos, tok in enumerate(tokens):
            nll -= np.log(softmax(logits[pos])[tok])
            count += 1
    return float(np.exp(nll / count))


def compare(cfg, w, vocab):
    rng = np.random.default_rng(0)
    # Reference text: stories the float model writes itself (temperature 1), avoiding BOS.
    stories = []
    for _ in range(8):
        pairs, _ = qz.greedy(cfg, w, 64, pick=lambda l, pos: int(rng.choice(len(l), p=softmax(np.where(np.arange(len(l)) == 1, -1e9, l)))))
        stories.append([tok for _, tok in pairs])
    f_pairs, f_logits = qz.greedy(cfg, w, 64)
    print(f"{'model':8} {'perplexity':>10} {'same words':>10} {'drift':>6}  greedy story")
    print(f"{'float32':8} {perplexity(cfg, w, stories):10.2f} {'64/64':>10} {0:6.2f}  {qz.decode(f_pairs, vocab)[:70]}")
    for bits in BITS:
        dq = dequantized(w, bits)
        pairs, logits = qz.greedy(cfg, dq, 64)
        same = 0
        while same < min(len(f_pairs), len(pairs)) and f_pairs[same] == pairs[same]:
            same += 1
        drift = np.abs(f_logits[:same + 1] - logits[:same + 1]).max()
        print(f"{str(bits) + '-bit':8} {perplexity(cfg, dq, stories):10.2f} {f'{same}/{len(f_pairs)}':>10} {drift:6.2f}  {qz.decode(pairs, vocab)[:70]}")


def main():
    if len(sys.argv) == 4 and sys.argv[2] == "--compare":
        cfg, w = qz.load_float(sys.argv[1])
        compare(cfg, w, qz.load_vocab(sys.argv[3], cfg["vocab_size"]))
    elif len(sys.argv) == 4 and sys.argv[2].isdigit() and int(sys.argv[2]) in BITS:
        cfg, w = qz.load_float(sys.argv[1])
        size = write_qb(cfg, w, int(sys.argv[2]), sys.argv[3])
        print(f"wrote {sys.argv[3]}: {size} bytes")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
