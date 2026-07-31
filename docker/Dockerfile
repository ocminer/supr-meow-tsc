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

# ---------- build ----------
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
ARG CUDA_ARCHS="80;86;89;90;120"
ARG JOBS=8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libssl-dev libzmq3-dev cppzmq-dev libargon2-dev libgmp-dev \
        flatbuffers-compiler libflatbuffers-dev python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# The llama.cpp fork is fetched here rather than relying on a host checkout,
# and OUR PATCH IS APPLIED — without it the miner does not link
# (llama_get_logits_device / llama_set_skip_logits_copy are ours, and the
# GPU-resident-logits path that doubled throughput depends on them).
ARG LLAMA_REPO=https://github.com/tensorcash/llama.cpp
ARG LLAMA_COMMIT=966fb8619648da1b2033dd78ee12756cbc7ff3b0
RUN rm -rf /src/vendor/llama.cpp && \
    git clone --filter=blob:none "${LLAMA_REPO}" /src/vendor/llama.cpp && \
    cd /src/vendor/llama.cpp && git checkout -q "${LLAMA_COMMIT}" && \
    git apply --verbose /src/tools/llama-gpu-logits.patch && \
    grep -q llama_get_logits_device include/llama.h || \
      (echo "FATAL: llama patch did not apply" && exit 1)

# A container has no NVIDIA driver, so libcuda.so exists only as a link-time
# STUB. ggml-cuda calls the driver API (cuMemMap, cuMemRelease, ...) and the
# link fails with undefined references unless the stub dir is on the linker
# path. The stub is never loaded at runtime — the real driver library is
# injected by the container runtime on the mining host.
ENV LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DMEOW_LLAMA_DIR=/src/vendor/llama.cpp \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs -lcuda" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs -lcuda" \
 && cmake --build build -j"${JOBS}" --target supr-meow-tsc

# ---------- runtime ----------
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libzmq5 libargon2-1 libgmp10 libgomp1 libssl3t64 \
        ca-certificates curl \
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

ENTRYPOINT ["/app/entrypoint.sh"]
