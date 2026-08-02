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
tar -tzf "${OUT}" | head -20
