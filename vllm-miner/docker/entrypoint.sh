#!/bin/bash
# supr-meow-tsc vLLM miner entrypoint.
#   Required: -e WALLET=tc1q...   -e POOL=stratum+tcp://host:port
#   Optional: -e WORKER=rig1  -e GPUS=all|0|2,3  -e PARALLEL=auto|pp|tp|single
#             -e MAX_NUM_SEQS=256  -e GPU_MEM_UTIL=0.84 ...
set -uo pipefail
: "${WALLET:?set -e WALLET=tc1q...}"
: "${POOL:?set -e POOL=stratum+tcp://host:port}"
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# --- parse POOL (stratum+tcp://host:port or host:port) into HOST/PORT ---------
_pp="${POOL#stratum+tcp://}"; _pp="${_pp#stratum+ssl://}"; _pp="${_pp#stratum+tls://}"
export POOL_HOST="${_pp%%:*}"
export POOL_PORT="${_pp##*:}"
export WORKER_SUFFIX="${WORKER:-${LABEL:-vllmrig0}}"
export WALLET

# --- load defaults (container env already exported wins via `:=`) ------------
export VENV=/opt/venv PKG_DIR=/app HF_MODEL_CACHE=/models
# shellcheck disable=SC1091
. /app/miner.env

SNAP="$HF_MODEL_CACHE/models--${MINING_MODEL_NAME/\//--}/snapshots/$MINING_MODEL_COMMIT"

# --- fetch + cache the model once (sha-pinned by revision) -------------------
if [ ! -d "$SNAP" ]; then
  say "model not cached — downloading $MINING_MODEL_NAME @$MINING_MODEL_COMMIT (10-30 min first boot)"
  HF_HUB_CACHE="$HF_MODEL_CACHE" "$VENV/bin/python" - <<PY || { say "model download FAILED"; exit 1; }
from huggingface_hub import snapshot_download
snapshot_download("${MINING_MODEL_NAME}", revision="${MINING_MODEL_COMMIT}",
                  cache_dir="${HF_MODEL_CACHE}",
                  allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model"])
print("model cached")
PY
  say "model cached at $SNAP"
else
  say "model present in $HF_MODEL_CACHE"
fi

say "starting vLLM miner: worker=$WALLET.$WORKER_SUFFIX pool=$POOL_HOST:$POOL_PORT gpus=$GPUS parallel=$PARALLEL mns=$MAX_NUM_SEQS"

# --- supervise: run-miner.sh starts engine+bridge; restart the pair on exit --
trap 'pkill -f "vllm serve"; pkill -f "vllm-stratum-bridge.py"; exit 0' TERM INT
while true; do
  bash /app/run-miner.sh /app/miner.env
  say "miner exited (rc=$?); cleaning up + restarting in 8s"
  pkill -f "vllm serve" 2>/dev/null; pkill -f "vllm-stratum-bridge.py" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 8
done
