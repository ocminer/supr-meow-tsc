# supr-meow-tsc vLLM backend — SLIM production miner image (v0.5.1).
#
#   docker run -d --gpus all --shm-size=4g \
#     -v meow-vllm-models:/models \
#     -e WALLET=tc1q... -e WORKER=rig1 \
#     -e POOL=stratum+tcp://tsc.suprnova.cc:3310 \
#     ocminersupr/supr-meow-tsc-vllm:latest
#
# Identical runtime behaviour to 0.5.0 (same wheel + same PoW overlay, already
# pool-audited), on the CUDA -runtime base instead of -devel. The devel base
# only added nvcc + multi-GB CUDA static libs that the running miner never
# touches: triton bundles its OWN cuda headers + ptxas (backends/nvidia/) and
# torch ships its own libcudart, so triton's first-boot JIT is self-contained.
# We still install gcc/g++ + python3.12-dev because triton compiles a small
# cuda_utils.c driver stub at first boot (needs a C compiler + Python.h).
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv python3-pip curl ca-certificates \
      gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# vLLM 0.19.0 (bundles torch+cu128) plus the overlay's runtime deps.
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir vllm==0.19.0 flatbuffers pyzmq numpy
ENV PATH=/opt/venv/bin:$PATH

# ── PoW overlay: the COMPLETE set of forked pure-Python files onto the stock
# wheel (sampler + serving/protocol layer that accepts the pow xarg + runner
# hooks + support modules). Same audited topk-opt / compact-CDF / inf-fixed set
# as 0.5.0 — byte-for-byte identical overlay.
ARG SITE=/opt/venv/lib/python3.12/site-packages
COPY overlay/vllm/  ${SITE}/vllm/
COPY overlay/proof/ ${SITE}/proof/

# ── miner: launcher, PP/TP auto-select, hardened bridge, config, entrypoint ──
COPY scripts/  /app/scripts/
COPY bridge/   /app/bridge/
COPY config/   /app/config/
COPY config/miner.env /app/miner.env
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN cp /app/scripts/run-miner.sh /app/run-miner.sh \
    && chmod +x /app/entrypoint.sh /app/run-miner.sh /app/scripts/run-miner.sh

VOLUME /models
ENV HF_HUB_CACHE=/models POW_USE_ASYNC_PROCESSING=1
WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
