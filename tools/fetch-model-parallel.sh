#!/bin/bash
# Parallel range download that needs ONE file's worth of disk, not two.
#
# The earlier version downloaded 8 parts and then `cat`-ed them into the
# output: that needs 2x16.4 GB on a box with 23 GB free, so the concatenation
# silently truncated and only the final sha256 caught it. Here each range is
# written straight into its offset in the output file, so peak usage is 16.4 GB.
#
# Per-chunk hashes are checked against the known-good local file, so a bad
# range is identified precisely instead of failing the whole 16 GB download.
set -u
URL=https://www.suprnova.cc/models/Qwen3-8B-9c925d64-bf16.gguf
SIZE=16388043744
SHA=fef5847c18f860086007cec0b08960f206c13c5cba69e3ed6292ee1e02ed7e44
OUT=/root/models/Qwen3-8B-9c925d64-bf16.gguf
N=8
LOG=/root/dl.log
chunk=$(( (SIZE + N - 1) / N ))

# Reference per-chunk sha256 (first 32 hex) of the model we have been mining.
REF=(043184ea0af2c8b95c76c14281d71c5a
     4f6fbad5089f50fabf412c34c7b3a775
     9fcb36bb7131e8ea0a5152947e32ab86
     7cca576b89cf91371614fcdd48f72002
     b5c28d4a1d7bae5466bdcd8e75002ba0
     55e3de3db96646479142aa3abcf8b6ba
     1f612f6534f3f35b961d73ae2d4f448b
     23a8e5e5ebe21f4745ee1a4d9f7c8745)

echo "[$(date -u +%H:%M:%S)] free before: $(df -h --output=avail / | tail -1)" > "$LOG"
rm -f "$OUT" /root/models/p[0-7] /root/models/q[0-7] /root/models/*.part
fallocate -l "$SIZE" "$OUT" 2>/dev/null || truncate -s "$SIZE" "$OUT"

fetch() {   # fetch <index>
  local i=$1 lo hi
  lo=$(( i * chunk )); hi=$(( lo + chunk - 1 ))
  [ "$hi" -ge "$SIZE" ] && hi=$(( SIZE - 1 ))
  curl -fsS --retry 5 --retry-delay 3 -r "${lo}-${hi}" "$URL" \
    | dd of="$OUT" bs=4M seek="$lo" oflag=seek_bytes conv=notrunc status=none
}

for i in $(seq 0 $((N-1))); do fetch "$i" & done
wait
echo "[$(date -u +%H:%M:%S)] transfers done, size $(stat -c %s "$OUT")" >> "$LOG"

bad=0
for i in $(seq 0 $((N-1))); do
  lo=$(( i * chunk ))
  got=$(tail -c +$((lo+1)) "$OUT" | head -c $chunk | sha256sum | cut -c1-32)
  if [ "$got" != "${REF[$i]}" ]; then
    echo "[$(date -u +%H:%M:%S)] chunk $i MISMATCH ($got != ${REF[$i]}) — refetching" >> "$LOG"
    fetch "$i"
    got=$(tail -c +$((lo+1)) "$OUT" | head -c $chunk | sha256sum | cut -c1-32)
    [ "$got" != "${REF[$i]}" ] && { echo "[$(date -u +%H:%M:%S)] chunk $i STILL BAD" >> "$LOG"; bad=1; }
  fi
done
[ "$bad" -eq 0 ] || { echo "[$(date -u +%H:%M:%S)] GIVING UP" >> "$LOG"; exit 1; }

got=$(sha256sum "$OUT" | cut -d' ' -f1)
if [ "$got" = "$SHA" ]; then
  echo "[$(date -u +%H:%M:%S)] sha256 OK, free after: $(df -h --output=avail / | tail -1)" >> "$LOG"
  {
    echo "[$(date -u +%H:%M:%S)] image OK"
    echo "[$(date -u +%H:%M:%S)] model downloaded"
    echo "[$(date -u +%H:%M:%S)] SETUP DONE"
  } > /root/setup.log
else
  echo "[$(date -u +%H:%M:%S)] WHOLE-FILE SHA MISMATCH got=$got" >> "$LOG"
  exit 1
fi
