#!/usr/bin/env bash
# Apply miner-owned patches to the pinned dependency; safe to repeat.
set -euo pipefail
patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
llama_dir="${1:?usage: prepare-llama.sh /path/to/pinned/llama.cpp}"
cd -- "$llama_dir"
for name in llama-gpu-logits llama-single-ubatch llama-q8-context-padding llama-f16-context-padding; do
  # The F16 addition changes the final line of the preceding Q8 hunk.
  if [[ $name == llama-q8-context-padding ]] &&
     patch --batch --silent -p1 --dry-run --reverse < "$patch_dir/llama-f16-context-padding.patch" >/dev/null 2>&1; then
    echo "$name already present beneath F16 extension"
    continue
  fi
  if patch --batch --silent -p1 --dry-run --forward < "$patch_dir/$name.patch"; then
    patch --batch -p1 --forward < "$patch_dir/$name.patch"
  elif patch --batch --silent -p1 --dry-run --reverse < "$patch_dir/$name.patch"; then
    echo "$name already applied"
  else
    echo "error: $name does not match this llama checkout" >&2
    exit 1
  fi
done
# Old source packages already raised this bound to 512. Normalize either
# pristine 256 or patched 512 before making the CMake definition authoritative.
python3 - <<'PY'
from pathlib import Path
p = Path('src/llama-cparams.h')
s = p.read_text()
if '#ifndef LLAMA_MAX_SEQ' not in s:
    import re
    s, count = re.subn(r'^#define LLAMA_MAX_SEQ (256|512)$',
        '#ifndef LLAMA_MAX_SEQ\n#define LLAMA_MAX_SEQ 512\n#endif', s, flags=re.M)
    if count != 1:
        raise SystemExit('unknown LLAMA_MAX_SEQ declaration')
    p.write_text(s)
PY
