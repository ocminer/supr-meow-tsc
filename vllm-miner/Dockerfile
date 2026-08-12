# supr-meow-tsc vLLM backend — production miner image for the sub-24 GB
# multi-GPU tier (two matched ~16 GB cards) and 24 GB+ single-GPU vLLM.
#
#   docker run -d --gpus all --shm-size=4g \
#     -v meow-vllm-models:/models \
#     -e WALLET=tc1q... -e WORKER=rig1 \
#     -e POOL=stratum+tcp://tsc.suprnova.cc:3310 \
#     ocminersupr/supr-meow-tsc-vllm:latest
#
# The vLLM 0.19.0 wheel ships native kernels for sm_80/89/90/100/120, so ONE
# image serves Ampere through Blackwell (incl. RTX 5080/5090 consumer Blackwell,
# sm_120) with NO per-arch source build. Our PoW additions are ALL pure-Python,
# overlaid onto the stock wheel — no recompilation.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

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
# hooks + support modules). This is the topk-opt / compact-CDF / inf-fixed
# build that passed the pool full-tier audit (see vllm-miner/docs).
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
