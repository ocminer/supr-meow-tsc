#!/usr/bin/env python3
"""Patch upstream proof-of-inference sources AFTER they are staged into the
build tree, so vendor/ stays an untouched submodule checkout.

Only genuine upstream defects belong here. Each patch states what it fixes and
fails loudly if the code it targets is gone — a silently skipped patch would
reintroduce the bug on the next submodule bump.
"""
import sys

PATCHES = [
    (
        "pow_utils.cpp",
        # adjusted_bits is the SHARE-mode threshold the verifier gates on, but
        # upstream derives it from `target` — the block target — so a sub-block
        # share is judged against a threshold it can never meet. The proof's
        # `target` must stay the block target (bcore validates solutions with
        # it), so the share threshold travels here instead.
        "                auto target_bytes = hex_to_bytes(target_it->second);\n",
        "                auto thr_it = is_solution ? target_it\n"
        "                                          : seq_pow_params.find(\"share_target\");\n"
        "                if (thr_it == seq_pow_params.end() || thr_it->second.empty())\n"
        "                    thr_it = target_it;\n"
        "                auto target_bytes = hex_to_bytes(thr_it->second);\n",
    ),
    (
        "pow_utils.cpp",
        # proof.fbs documents `timestamp` as a UNIX timestamp, and both
        # validation.fbs and blockheader.fbs narrow it to uint32 seconds. But
        # system_clock::time_since_epoch().count() is NANOSECONDS on libstdc++,
        # so every proof carries a value ~1e9 too large. The proof itself still
        # serializes (uint64), which is why this stays invisible until a
        # validator converts it and rejects the proof as malformed:
        #   "bad number 1785206392599723924 for type uint32"
        "         std::chrono::system_clock::now().time_since_epoch().count()\n",
        "         std::chrono::duration_cast<std::chrono::seconds>(\n"
        "             std::chrono::system_clock::now().time_since_epoch()).count()\n",
    ),
]

def main(stage_dir: str) -> int:
    for filename, old, new in PATCHES:
        path = f"{stage_dir}/{filename}"
        with open(path) as fh:
            text = fh.read()
        if new in text:
            continue                      # already patched by an earlier build
        if old not in text:
            print(f"error: patch target not found in {filename} — upstream "
                  f"changed; re-check the fix before removing it", file=sys.stderr)
            return 1
        with open(path, "w") as fh:
            fh.write(text.replace(old, new, 1))
        print(f"patched {filename}: proof timestamp is seconds, not nanoseconds")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
