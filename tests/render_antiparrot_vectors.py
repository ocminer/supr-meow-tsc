#!/usr/bin/env python3
"""Render the upstream anti-parrot mainnet windows (.npz + expected_bcore.json,
vendor/tensorcash-v1.2.2/.../tests/vectors/antiparrot_mainnet) into one flat
little-endian file that tests/precheck_test.cpp reads without numpy.

Layout: b"MAP1", u32 count, then per window:
  u32 height, u32 steps, u32 row_len, u32 n_prompt, u32 n_pad, u32 reason_len,
  reason bytes, 7 x u64 bcore metrics (braw, bcap, hard_run, maxrs_q32,
  reuse_cache_q32, n_states_eff_u, n_states_eff_id),
  f32 logits[steps*row_len], u32 ids[steps*row_len], u32 chosen[steps],
  u32 prompt[n_prompt], u8 pad[n_pad].
"""
import json
import os
import struct
import sys

import numpy as np

METRICS = ("braw", "bcap", "hard_run", "maxrs_q32", "reuse_cache_q32",
           "n_states_eff_u", "n_states_eff_id")


def main(src, out):
    proofs = json.load(open(os.path.join(src, "expected_bcore.json")))["proofs"]
    with open(out, "wb") as f:
        f.write(b"MAP1")
        f.write(struct.pack("<I", len(proofs)))
        for p in proofs:
            a = np.load(os.path.join(src, "h%06d.npz" % p["height"]), allow_pickle=False)
            logits = np.ascontiguousarray(a["logits"], dtype=np.float32)
            ids = np.ascontiguousarray(a["ids"], dtype=np.int64)
            chosen = np.ascontiguousarray(a["chosen"], dtype=np.int64)
            prompt = np.ascontiguousarray(a["prompt"], dtype=np.int64)
            pad = np.ascontiguousarray(a["pad_mask"], dtype=np.uint8)
            steps, row_len = logits.shape
            assert ids.shape == logits.shape and chosen.shape == (steps,)
            reason = p["reason"].encode()
            f.write(struct.pack("<6I", p["height"], steps, row_len, len(prompt), len(pad), len(reason)))
            f.write(reason)
            for m in METRICS:
                f.write(struct.pack("<Q", int(p[m])))
            f.write(logits.tobytes())
            f.write(ids.astype(np.uint32).tobytes())
            f.write(chosen.astype(np.uint32).tobytes())
            f.write(prompt.astype(np.uint32).tobytes())
            f.write(pad.tobytes())
    print("rendered %d anti-parrot mainnet windows -> %s" % (len(proofs), out))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: render_antiparrot_vectors.py <antiparrot_mainnet dir> <out.bin>")
    main(sys.argv[1], sys.argv[2])
