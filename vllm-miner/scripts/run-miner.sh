#!/bin/bash
# TensorCash vLLM multi-GPU miner launcher (round-5 GREEN build).
# Config-driven: edit config/miner.env. Auto-selects PP vs TP by interconnect,
# starts the inf-fixed vLLM engine + the hardened stratum bridge, both under a
# supervised loop. One command; no per-run flags.
set -u
PKG="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:-$PKG/config/miner.env}"
[ -f "$CFG" ] || { echo "config not found: $CFG" >&2; exit 1; }
# shellcheck disable=SC1090
. "$CFG"

GPU_ARR=(${GPUS//,/ }); N=${#GPU_ARR[@]}

# --- parallelism selection --------------------------------------------------
case "${PARALLEL:-auto}" in
  single) PAR="--tensor-parallel-size 1" ;;
  tp)     PAR="--tensor-parallel-size $N" ;;
  pp)     PAR="--tensor-parallel-size 1 --pipeline-parallel-size $N --disable-custom-all-reduce" ;;
  auto)
    MODE=$("$VENV/bin/python" "$PKG/scripts/gpu_parallel_select.py" "${GPU_ARR[@]}" 2>/dev/null | awk '{print $1}')
    if [ "$MODE" = tensor ]; then PAR="--tensor-parallel-size $N"
    else PAR="--tensor-parallel-size 1 --pipeline-parallel-size $N --disable-custom-all-reduce"; fi
    echo "[run-miner] auto-select: $MODE -> $PAR" >&2 ;;
  *) echo "bad PARALLEL=$PARALLEL" >&2; exit 1 ;;
esac
[ "$N" -eq 1 ] && PAR="--tensor-parallel-size 1"

SNAP="$HF_MODEL_CACHE/models--${MINING_MODEL_NAME/\//--}/snapshots/$MINING_MODEL_COMMIT/"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE="$HF_MODEL_CACHE" VLLM_HOST_IP=127.0.0.1
export VLLM_ENABLE_POW=1 POW_PROOF_VERSION="$POW_PROOF_VERSION" POW_PROCESSOR_MODE="$POW_PROCESSOR_MODE"
export POW_FAST_SETUP="${POW_FAST_SETUP:-1}"
export POW_EGRESS_MODE=broker POW_PROXY_ENABLE=false ZMQ_PUSH_HOST=127.0.0.1 ZMQ_PUSH_PORT="$ZMQ_PUSH_PORT"

_engine() {
  cd "$TC_VLLM"
  exec "$VENV/bin/vllm" serve "$SNAP" \
    --served-model-name "$MINING_MODEL_NAME" --revision "$MINING_MODEL_COMMIT" \
    $PAR --dtype "$MINING_DTYPE" --load-format safetensors \
    --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prompt-tokens-details --generation-config vllm \
    --host 127.0.0.1 --port 8000 --api-key "$VLLM_API_KEY"
}

_bridge() {
  exec "$VENV/bin/python" "$PKG/bridge/vllm-stratum-bridge.py" \
    --host "$POOL_HOST" --port "$POOL_PORT" \
    --user "$WALLET.$WORKER_SUFFIX" \
    --vllm-port 8000 --vllm-key "$VLLM_API_KEY" \
    --zmq-port "$ZMQ_PUSH_PORT" --concurrency "$CONCURRENCY" --b-floor "$B_FLOOR" \
    --dump-proof "$PKG/last-submitted-proof.bin"
}

case "${MODE_RUN:-both}" in
  engine) _engine ;;
  bridge) _bridge ;;
  both)
    echo "[run-miner] starting engine ($PAR) on GPUs $GPUS ..." >&2
    ( _engine ) > "$PKG/engine.log" 2>&1 &
    EPID=$!
    # wait for engine health, then bridge
    for _ in $(seq 1 90); do
      grep -qE "Application startup complete|Uvicorn running" "$PKG/engine.log" 2>/dev/null && break
      kill -0 "$EPID" 2>/dev/null || { echo "[run-miner] engine died; see engine.log" >&2; exit 1; }
      sleep 4
    done
    echo "[run-miner] engine up; starting bridge -> $POOL_HOST:$POOL_PORT worker $WALLET.$WORKER_SUFFIX" >&2
    ( _bridge ) > "$PKG/bridge.log" 2>&1 &
    BPID=$!
    trap 'kill -9 $BPID $EPID 2>/dev/null; pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "VLLM::Worker"' TERM INT
    wait "$BPID"
    ;;
esac
