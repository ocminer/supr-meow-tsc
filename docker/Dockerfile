
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
ARG CUDA_ARCHS="80;86;89;90;120"
ARG JOBS=
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git patch ca-certificates \
        libssl-dev libzmq3-dev cppzmq-dev libargon2-dev libgmp-dev \
        flatbuffers-compiler libflatbuffers-dev python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY tools/             /src/tools/
COPY vendor/nlohmann/   /src/vendor/nlohmann/
COPY vendor/tensorcash/ /src/vendor/tensorcash/

ARG LLAMA_MAX_SEQ=512
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
    bash /src/tools/prepare-llama.sh /src/vendor/llama.cpp

ENV LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

COPY CMakeLists.txt /src/
COPY src/           /src/src/

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DMEOW_LLAMA_DIR=/src/vendor/llama.cpp \
      -DMEOW_MAX_SEQUENCES="${LLAMA_MAX_SEQ}" \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs -lcuda" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs -lcuda" \
 && cmake --build build -j"${JOBS:-$(nproc)}" --target supr-meow-tsc

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
COPY tools/q8-profile.sh                      /app/q8-profile.sh
COPY docker/entrypoint.sh                       /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENV LD_LIBRARY_PATH=/app/lib

ARG BAKE_MODEL=""
RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 21550
EXPOSE 22

ENTRYPOINT ["/app/entrypoint.sh"]
