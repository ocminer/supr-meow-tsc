#!/usr/bin/env bash
# Verify that vendor/tensorcash-v1.2.2/ is byte-identical to the upstream
# files recorded in its UPSTREAM.txt (TensorCash v1.2.2, commit 48d1344).
# Exit status is non-zero on any missing or edited file.
set -euo pipefail
cd "$(dirname "$0")/../vendor/tensorcash-v1.2.2"
sed -n '/^[0-9a-f]\{64\}  /p' UPSTREAM.txt > /tmp/meow-vendored-precheck.$$.sha256
trap 'rm -f /tmp/meow-vendored-precheck.$$.sha256' EXIT
n=$(wc -l < /tmp/meow-vendored-precheck.$$.sha256)
[ "$n" -gt 0 ] || { echo "check-vendored-precheck: no hashes in UPSTREAM.txt" >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum --quiet -c /tmp/meow-vendored-precheck.$$.sha256
else
    shasum -a 256 --quiet -c /tmp/meow-vendored-precheck.$$.sha256
fi
echo "check-vendored-precheck: $n files match TensorCash v1.2.2"
