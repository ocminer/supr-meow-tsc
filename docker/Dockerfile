# supr-meow-tsc — deployable miner image.
#
# Two variants from one file:
#   docker build -t supr-meow-tsc:slim .                      (no model, ~1.5 GB)
#   docker build --build-arg BAKE_MODEL=/path/rel/to/ctx \
#                -t supr-meow-tsc:full .                      (model inside)
#
# The slim image fetches the GGUF at start from $MODEL_URL and caches it in
# /models, so mount a volume there if the host has persistent storage.
#
# MULTI-ARCH IS MANDATORY: rented rigs are not all Blackwell. sm_80 A100,
# sm_86 A10/3090, sm_89 4090, sm_90 H100, sm_120 5090. Building for one arch
# produces a binary that dies with "no kernel image is available" elsewhere.
#
# sm_100 (B200) IS DELIBERATELY ABSENT — DO NOT "FIX" THIS.
# The build embeds PTX as well as SASS (cuobjdump names the entries
# `*.sm_90.ptx`, NOT `compute_90` — grepping for the latter wrongly reports no
# PTX at all). A B200 therefore runs fine by JIT-compiling the sm_90 PTX
# forward, and forward JIT is the only direction that works: sm_120 PTX cannot
# serve sm_100 because 120 > 100.
#
# Adding 100 here was measured on 2026-08-03 on an 8x B200 box and is a 15%
# REGRESSION. Concurrent A/B, same box, same moment, all cards at 512 slots,
# four samples per arm with no overlap between them:
#     JIT from sm_90 PTX  56.2 w/s   (55.0 56.0 56.1 57.9)
#     native sm_100 SASS  49.0 w/s   (47.9 48.1 50.0 50.1)
# Reproduced three ways (mixed sweep, all-512 run, concurrent A/B). ggml-cuda's
# Blackwell-datacenter kernel selection is simply slower for this workload than
# its Hopper path retargeted by ptxas. Re-measure before adding sm_100 back;
# a future ggml may flip the result.

# ---------- build ----------
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
ARG CUDA_ARCHS="80;86;89;90;120"
# Empty = use every core the builder has. The CUDA compile is several hundred
# template instantiations across five architectures and is the whole cost of
# this image; the old fixed -j8 left 24 of 32 cores idle for over an hour.
ARG JOBS=
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libssl-dev libzmq3-dev cppzmq-dev libargon2-dev libgmp-dev \
        flatbuffers-compiler libflatbuffers-dev python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Copy ONLY what the compile consumes, never `COPY . /src`. The whole-repo copy
# meant that editing docker/entrypoint.sh, a README or a doc invalidated this
# layer and every layer after it — i.e. a one-line entrypoint change triggered
# the full ~70 minute CUDA rebuild. vendor/llama.cpp is deliberately absent: it
# is cloned at the pinned commit below, and copying the host checkout over the
# clone would defeat the `rm -rf` that keeps the fork pristine.
COPY tools/             /src/tools/
COPY vendor/nlohmann/   /src/vendor/nlohmann/
COPY vendor/tensorcash/ /src/vendor/tensorcash/

# The llama.cpp fork is fetched here rather than relying on a host checkout,
# and OUR PATCH IS APPLIED — without it the miner does not link
# (llama_get_logits_device / llama_set_skip_logits_copy are ours, and the
# GPU-resident-logits path that doubled throughput depends on them).
# Stock llama.cpp hard-codes `#define LLAMA_MAX_SEQ 256`, which caps slots per
# context regardless of free VRAM — on an H100 that was the real ceiling, not
# the card (256->37.6 w/s, 512->40.5). Set as a build arg rather than a patch
# file so a bigger card can be tried without a new patch to maintain. The
# constant sizes std::bitset<LLAMA_MAX_SEQ> and per-batch arrays, so ALWAYS
# re-check reject counts after raising it: a regression corrupts transcripts
# rather than crashing.
ARG LLAMA_MAX_SEQ=512
# github.com/tensorcash/llama.cpp vanished on 2026-08-02 (clone began
# prompting for credentials) and the pinned commit is fork-specific — it is NOT
# in ggml-org/llama.cpp, so there was no upstream to fall back on and every
# clean build broke. The fork was recovered from the Docker cache and rehosted
# at github.com/ocminer/llama.cpp, which we control; verified fetchable
# anonymously. vendor/llama-src.tar.gz (already patched, gitignored) is kept as
# an offline fallback and wins when present.
ARG LLAMA_REPO=https://github.com/ocminer/llama.cpp
ARG LLAMA_COMMIT=966fb8619648da1b2033dd78ee12756cbc7ff3b0
COPY vendor/llama-src.tar.gz* /tmp/
RUN rm -rf /src/vendor/llama.cpp && \
    if [ -f /tmp/llama-src.tar.gz ]; then \
      echo "using vendored llama fork" && \
      tar -xzf /tmp/llama-src.tar.gz -C /src/vendor/ && rm -f /tmp/llama-src.tar.gz; \
    else \
      echo "vendored copy absent — cloning ${LLAMA_REPO}" && \
      git clone --filter=blob:none "${LLAMA_REPO}" /src/vendor/llama.cpp && \
      cd /src/vendor/llama.cpp && git checkout -q "${LLAMA_COMMIT}" && \
      git apply --verbose /src/tools/llama-gpu-logits.patch; \
    fi && \
    cd /src/vendor/llama.cpp && \
    sed -i "s/#define LLAMA_MAX_SEQ .*/#define LLAMA_MAX_SEQ ${LLAMA_MAX_SEQ}/" src/llama-cparams.h && \
    grep -q "#define LLAMA_MAX_SEQ ${LLAMA_MAX_SEQ}" src/llama-cparams.h && \
    grep -q llama_get_logits_device include/llama.h || \
      (echo "FATAL: llama patch did not apply" && exit 1)

# A container has no NVIDIA driver, so libcuda.so exists only as a link-time
# STUB. ggml-cuda calls the driver API (cuMemMap, cuMemRelease, ...) and the
# link fails with undefined references unless the stub dir is on the linker
# path. The stub is never loaded at runtime — the real driver library is
# injected by the container runtime on the mining host.
ENV LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

# Our own sources land last: they change most often, and everything above
# (apt, the llama clone and patches) then stays cached across edits.
COPY CMakeLists.txt /src/
COPY src/           /src/src/

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DMEOW_LLAMA_DIR=/src/vendor/llama.cpp \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs -lcuda" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs -lcuda" \
 && cmake --build build -j"${JOBS:-$(nproc)}" --target supr-meow-tsc

# ---------- runtime ----------
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libzmq5 libargon2-1 libgmp10 libgomp1 libssl3t64 \
        ca-certificates curl \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/build/supr-meow-tsc      /app/
COPY --from=build /src/build/bin/*.so*          /app/lib/
COPY docker/entrypoint.sh                       /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENV LD_LIBRARY_PATH=/app/lib

# Optional: bake a GGUF into the image (path relative to the build context).
# Slim by default: no model baked in, fetched at container start (MODEL_URL).
ARG BAKE_MODEL=""
RUN mkdir -p /models
VOLUME ["/models"]

# Local stats API for rig managers (HiveOS-style). Off by default in Docker
# unless API_BIND is set to 0.0.0.0:21550.
EXPOSE 21550
# OctaSpace gives a rented rig NO shell unless the image itself ships an SSH
# server — it is not a platform feature. Without it a rig is a black box you
# cannot debug. sshd starts ONLY when SSH_PUBKEY is provided; no key, no
# daemon, no password auth ever.
EXPOSE 22

ENTRYPOINT ["/app/entrypoint.sh"]
