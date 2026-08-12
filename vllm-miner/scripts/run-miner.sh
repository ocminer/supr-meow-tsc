#!/bin/bash
# TensorCash vLLM multi-GPU miner launcher (round-5 GREEN build).
# Config-driven: edit config/miner.env. Auto-selects PP vs TP by interconnect,
# starts the inf-fixed vLLM engine + the hardened stratum bridge, both under a
# supervised loop. One command; no per-run flags.
set -u
PKG="${PKG_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
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
# expandable_segments avoids fragmentation OOM at high batch (part of the
# audited high-batch bundle)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

_engine() {
  exec "$VENV/bin/vllm" serve "$SNAP" \
    --served-model-name "$MINING_MODEL_NAME" --revision "$MINING_MODEL_COMMIT" \
    $PAR --dtype "$MINING_DTYPE" --load-format safetensors \
    --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --compilation-config "{\"max_cudagraph_capture_size\": $MAX_NUM_SEQS}" \
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

# vLLM emits engine + EngineCore subprocess logs to its own stdout/stderr; keep
# them ON the container's streams (do NOT swallow into a file) so crashes are
# visible in `docker logs`. Persist the compile/cudagraph cache to the model
# volume so the slow cold boot (CUDA graph capture) happens ONCE.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HF_MODEL_CACHE/.vllm_cache}"
mkdir -p "$VLLM_CACHE_ROOT" 2>/dev/null || true

case "${MODE_RUN:-both}" in
  engine) _engine ;;
  bridge) _bridge ;;
  both)
    echo "[run-miner] starting engine ($PAR) on GPUs $GPUS ..." >&2
    _engine &
    EPID=$!
    trap 'kill $EPID 2>/dev/null; pkill -f "vllm-stratum-bridge.py"; pkill -9 -f "EngineCore"' TERM INT EXIT
    # Health-poll (not log-grep): cold first boot captures CUDA graphs for up to
    # ~15 min. Fail only if the engine PROCESS actually exits.
    ready=0
    for _ in $(seq 1 200); do   # 200 x 5s = ~16 min
      if curl -sf -o /dev/null -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8000/health 2>/dev/null; then
        ready=1; break
      fi
      kill -0 "$EPID" 2>/dev/null || { echo "[run-miner] engine process exited during startup" >&2; exit 1; }
      sleep 5
    done
    [ "$ready" = 1 ] || { echo "[run-miner] engine did not become healthy in time" >&2; exit 1; }
    echo "[run-miner] engine healthy; starting bridge -> $POOL_HOST:$POOL_PORT worker $WALLET.$WORKER_SUFFIX" >&2
    _bridge &
    BPID=$!
    # exit (and let the entrypoint restart the pair) if EITHER dies
    while kill -0 "$EPID" 2>/dev/null && kill -0 "$BPID" 2>/dev/null; do sleep 5; done
    echo "[run-miner] engine or bridge exited; cycling" >&2
    ;;
esac
