#!/usr/bin/env bash
# Build the HiveOS custom-miner package and drop the tarball in dist/.
#
#   tools/build-hiveos-package.sh [version]
#
# The package must be built against Ubuntu 22.04 because that is what HiveOS
# runs; a binary from the normal 24.04 image fails there with
# "GLIBC_2.38 not found". See docker/Dockerfile.hiveos.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-$(grep -oP 'kVersion = "\K[^"]+' src/main.cpp)}"
NAME=supr-meow-tsc
TAG="${NAME}:hiveos-${VERSION}"
OUT="dist/${NAME}-${VERSION}.tar.gz"

echo "building HiveOS package ${VERSION} (Ubuntu 22.04 / CUDA 12.8)"
docker build -f docker/Dockerfile.hiveos --target package \
  --build-arg PKG_VERSION="${VERSION}" -t "${TAG}" .

mkdir -p dist
cid=$(docker create "${TAG}")
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
docker cp "${cid}:/${NAME}-${VERSION}.tar.gz" "${OUT}"

echo
echo "package: ${OUT}  ($(du -h "${OUT}" | cut -f1))"
sha256sum "${OUT}" | tee "${OUT}.sha256"
echo
echo "contents:"
# Capture the listing FIRST, then truncate the string. Two reasons, and the
# order matters:
#
#   `tar -tzf "$OUT" | head -20` makes tar take SIGPIPE when head exits, and
#   under `set -o pipefail` that is exit 141 — the script reported FAILURE on a
#   cosmetic listing, long after writing a valid package.
#
#   The obvious patch, appending `|| true`, swaps that bug for a worse one:
#   `tar -tzf` walks and decompresses the whole archive, so it is the only
#   integrity check here, and `|| true` would let a CORRUPT 774 MB tarball
#   publish silently. Verified: with `|| true` a CRC-damaged archive exits 0;
#   with the capture below it exits 2 and `set -e` stops the release.
#
# Command substitution reads tar to completion, so there is no SIGPIPE and a
# real failure still aborts. Only the harmless in-memory truncation is guarded.
listing=$(tar -tzf "${OUT}")
printf '%s\n' "${listing}" | head -20 || true
