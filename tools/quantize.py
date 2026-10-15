#!/usr/bin/env python3
"""Quantize a llama2.c legacy (v0) float32 checkpoint to int8 with one float scale per row.

Usage:
  quantize.py MODEL.bin OUT_q8.bin          write the int8 model
  quantize.py MODEL.bin OUT_q8.bin --check TOKENIZER.bin
                                            also compare float vs int8 greedy stories

Output layout (all little-endian, every section 4-byte aligned):
  b"Q8RW", int32 version=1, Config (7 x int32, vocab_size > 0 means shared classifier)
  embedding block                    (n=dim,    d=vocab)
  rms_att    float32[layers*dim]
  wq blocks per layer                (n=dim,    d=dim)
  wk blocks per layer                (n=dim,    d=kv_dim)
  wv blocks per layer                (n=dim,    d=kv_dim)
  wo blocks per layer                (n=dim,    d=dim)
  rms_ffn    float32[layers*dim]
  w1 blocks per layer                (n=dim,    d=hidden)
  w2 blocks per layer                (n=hidden, d=dim)
  w3 blocks per layer                (n=dim,    d=hidden)
  rms_final  float32[dim]
A block is float32 scales[d] followed by int8 q[d*n] (row i = output i), padded to 4 bytes.
"""
import struct
import sys

import numpy as np

MAGIC = b"Q8RW"
VERSION = 1
CONFIG_FIELDS = ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len")


def load_float(path):
    data = open(path, "rb").read()
    cfg = dict(zip(CONFIG_FIELDS, struct.unpack("7i", data[:28])))
    if cfg["vocab_size"] <= 0:
        sys.exit("unshared classifier weights are not supported")
    f = np.frombuffer(data, dtype=np.float32, offset=28)
    dim, hid, L = cfg["dim"], cfg["hidden_dim"], cfg["n_layers"]
    kv = dim * cfg["n_kv_heads"] // cfg["n_heads"]
    shapes = [
        ("emb", (cfg["vocab_size"], dim)),
        ("rms_att", (L, dim)),
        ("wq", (L, dim, dim)),
        ("wk", (L, kv, dim)),
        ("wv", (L, kv, dim)),
        ("wo", (L, dim, dim)),
        ("rms_ffn", (L, dim)),
        ("w1", (L, hid, dim)),
        ("w2", (L, dim, hid)),
        ("w3", (L, hid, dim)),
        ("rms_final", (dim,)),
    ]
    w, pos = {}, 0
    for name, shape in shapes:
        size = int(np.prod(shape))
        w[name] = f[pos:pos + size].reshape(shape)
        pos += size
    return cfg, w


def quantize_rows(m):
    """m: (d, n) -> (scales[d], q[d, n] int8) with m ~= q * scales[:, None]."""
    scales = np.abs(m).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.round(m / scales[:, None]), -127, 127).astype(np.int8)
    return scales.astype(np.float32), q


def block_bytes(m):
    scales, q = quantize_rows(m)
    raw = scales.tobytes() + q.tobytes()
    return raw + b"\0" * (-len(raw) % 4)


def write_q8(cfg, w, path):
    out = [MAGIC, struct.pack("i", VERSION), struct.pack("7i", *(cfg[k] for k in CONFIG_FIELDS))]
    out.append(block_bytes(w["emb"]))
    out.append(w["rms_att"].tobytes())
    for name in ("wq", "wk", "wv", "wo"):
        out += [block_bytes(layer) for layer in w[name]]
    out.append(w["rms_ffn"].tobytes())
    for name in ("w1", "w2", "w3"):
        out += [block_bytes(layer) for layer in w[name]]
    out.append(w["rms_final"].tobytes())
    data = b"".join(out)
    open(path, "wb").write(data)
    return len(data)


def dequantized(w):
    """The weights the board will effectively use."""
    dq = dict(w)
    for name in ("emb", "wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        m = w[name]
        flat = m.reshape(-1, m.shape[-1])
        scales, q = quantize_rows(flat)
        dq[name] = (q.astype(np.float32) * scales[:, None]).reshape(m.shape)
    return dq


def rmsnorm(x, weight):
    return weight * (x / np.sqrt(np.mean(x * x) + 1e-5))


def greedy(cfg, w, steps):
    """Same math as run.c's forward(), always taking the most likely token. Starts from BOS."""
    dim, L = cfg["dim"], cfg["n_layers"]
    heads, kv_heads = cfg["n_heads"], cfg["n_kv_heads"]
    hs = dim // heads
    kv_dim = dim * kv_heads // heads
    kcache = np.zeros((L, steps, kv_dim), np.float32)
    vcache = np.zeros((L, steps, kv_dim), np.float32)
    token, out, all_logits = 1, [], []
    for pos in range(steps):
        x = w["emb"][token].copy()
        for l in range(L):
            xb = rmsnorm(x, w["rms_att"][l])
            q, k, v = w["wq"][l] @ xb, w["wk"][l] @ xb, w["wv"][l] @ xb
            for i in range(0, dim, 2):
                freq = 1.0 / 10000.0 ** ((i % hs) / hs)
                c, s = np.cos(pos * freq), np.sin(pos * freq)
                for vec in ((q, k) if i < kv_dim else (q,)):
                    v0, v1 = vec[i], vec[i + 1]
                    vec[i], vec[i + 1] = v0 * c - v1 * s, v0 * s + v1 * c
            kcache[l, pos], vcache[l, pos] = k, v
            att_out = np.zeros(dim, np.float32)
            for h in range(heads):
                kh = h // (heads // kv_heads)
                keys = kcache[l, :pos + 1, kh * hs:(kh + 1) * hs]
                scores = keys @ q[h * hs:(h + 1) * hs] / np.sqrt(hs)
                a = np.exp(scores - scores.max())
                a /= a.sum()
                att_out[h * hs:(h + 1) * hs] = a @ vcache[l, :pos + 1, kh * hs:(kh + 1) * hs]
            x = x + w["wo"][l] @ att_out
            xb = rmsnorm(x, w["rms_ffn"][l])
            h1 = w["w1"][l] @ xb
            x = x + w["w2"][l] @ (h1 / (1 + np.exp(-h1)) * (w["w3"][l] @ xb))
        logits = w["emb"] @ rmsnorm(x, w["rms_final"])
        all_logits.append(logits)
        nxt = int(np.argmax(logits))
        if nxt == 1:
            break
        out.append((token, nxt))
        token = nxt
    return out, np.array(all_logits)


def load_vocab(path, vocab_size):
    data = open(path, "rb").read()
    pos, vocab = 4, []
    for _ in range(vocab_size):
        n = struct.unpack_from("i", data, pos + 4)[0]
        vocab.append(data[pos + 8:pos + 8 + n].decode("utf-8", "replace"))
        pos += 8 + n
    return vocab


def decode(pairs, vocab):
    text = ""
    for prev, tok in pairs:
        piece = vocab[tok]
        if prev == 1 and piece.startswith(" "):
            piece = piece[1:]
        if piece.startswith("<0x") and piece.endswith(">"):
            piece = chr(int(piece[3:-1], 16))
        text += piece
    return text


def main():
    if len(sys.argv) not in (3, 5) or (len(sys.argv) == 5 and sys.argv[3] != "--check"):
        sys.exit(__doc__)
    cfg, w = load_float(sys.argv[1])
    size = write_q8(cfg, w, sys.argv[2])
    print(f"config {cfg}")
    print(f"wrote {sys.argv[2]}: {size} bytes")
    if len(sys.argv) == 5:
        vocab = load_vocab(sys.argv[4], cfg["vocab_size"])
        steps = min(64, cfg["seq_len"])
        f_pairs, f_logits = greedy(cfg, w, steps)
        q_pairs, q_logits = greedy(cfg, dequantized(w), steps)
        same = 0
        while same < min(len(f_pairs), len(q_pairs)) and f_pairs[same] == q_pairs[same]:
            same += 1
        # only positions where both runs saw the same tokens are comparable
        diff = np.abs(f_logits[:same + 1] - q_logits[:same + 1]).max()
        spread = np.ptp(f_logits[:same + 1], axis=1).mean()
        print(f"float32: {decode(f_pairs, vocab)}")
        print(f"int8:    {decode(q_pairs, vocab)}")
        print(f"identical tokens before first difference: {same}/{len(f_pairs)}")
        print(f"max logit difference while identical: {diff:.3f} (typical logit range {spread:.1f})")


if __name__ == "__main__":
    main()
