#!/usr/bin/env bash
# Env-var front end for supr-meow-tsc. Everything a rented rig needs is
# settable without touching the command line.
#
# REQUIRED
#   POOL_URL    stratum+tcp://host:port   (repeatable via POOL_URL2, POOL_URL3)
#   WALLET      payout address
#
# MODEL (one of)
#   MODEL_PATH  path inside the container (use with a mounted volume / baked image)
#   MODEL_URL   fetched to $MODEL_DIR on first start, then cached
#   MODEL_SHA256  optional integrity check for MODEL_URL
#
# OPTIONAL (defaults are the values measured on an RTX 5090 / Qwen3-8B)
#   WORKER, PASSWORD, DEVICES, SLOTS, GROUPS, CTX, API_BIND, LOG_INTERVAL,
#   MODEL_DIR, EXTRA_ARGS
#   DOUBLE_BUFFER=0|1   PROMPT_STYLE=0|1
set -euo pipefail

die() { echo "error: $*" >&2; exit 2; }
[[ -n "${POOL_URL:-}" ]] || die "POOL_URL is required (stratum+tcp://host:port)"
[[ -n "${WALLET:-}"   ]] || die "WALLET is required"

MODEL_DIR="${MODEL_DIR:-/models}"
mkdir -p "$MODEL_DIR"

# ---- model resolution -------------------------------------------------
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -n "${MODEL_URL:-}" ]]; then
    fname="$(basename "${MODEL_URL%%\?*}")"
    MODEL_PATH="$MODEL_DIR/$fname"
    if [[ ! -s "$MODEL_PATH" ]]; then
      echo "[entrypoint] downloading model: $MODEL_URL"
      curl -fL --retry 5 --retry-delay 5 -o "$MODEL_PATH.part" "$MODEL_URL" \
        || die "model download failed"
      mv "$MODEL_PATH.part" "$MODEL_PATH"
    else
      echo "[entrypoint] using cached model: $MODEL_PATH"
    fi
    if [[ -n "${MODEL_SHA256:-}" ]]; then
      echo "[entrypoint] verifying sha256..."
      echo "${MODEL_SHA256}  ${MODEL_PATH}" | sha256sum -c - \
        || die "model checksum mismatch — refusing to mine with an unverified model"
    fi
  else
    # last resort: a single .gguf already present in MODEL_DIR
    found=$(find "$MODEL_DIR" -maxdepth 1 -name '*.gguf' | head -1 || true)
    [[ -n "$found" ]] || die "no model: set MODEL_PATH or MODEL_URL, or mount a .gguf into $MODEL_DIR"
    MODEL_PATH="$found"
    echo "[entrypoint] found model: $MODEL_PATH"
  fi
fi
[[ -s "$MODEL_PATH" ]] || die "model not readable: $MODEL_PATH"

# ---- user string ------------------------------------------------------
USER_ARG="$WALLET"
[[ -n "${WORKER:-}" ]] && USER_ARG="${WALLET}.${WORKER}"

# ---- assemble ---------------------------------------------------------
args=( -o "$POOL_URL" )
[[ -n "${POOL_URL2:-}" ]] && args+=( -o "$POOL_URL2" )
[[ -n "${POOL_URL3:-}" ]] && args+=( -o "$POOL_URL3" )
args+=( -u "$USER_ARG" -p "${PASSWORD:-x}" --model "$MODEL_PATH" --no-color )
[[ -n "${DEVICES:-}"      ]] && args+=( -d "$DEVICES" )
args+=( --slots "${SLOTS:-128}" --groups "${GROUPS:-12}" )
[[ -n "${CTX:-}"          ]] && args+=( --ctx "$CTX" )
args+=( --api-bind "${API_BIND:-off}" )
[[ -n "${LOG_INTERVAL:-}" ]] && args+=( --log-interval "$LOG_INTERVAL" )
[[ -n "${EXTRA_ARGS:-}"   ]] && args+=( ${EXTRA_ARGS} )

# Tuning defaults measured on RTX 5090 + Qwen3-8B: single-buffered decode
# (8B is bandwidth-bound, so double-buffering pays the weight-read twice) and
# open-ended prompts (the chain's reuse-entropy guard rejects predictable
# continuations; a factual prompt measured 3.9% honest rejects).
export MEOW_DOUBLE_BUFFER="${DOUBLE_BUFFER:-0}"
export POW_PROMPT_STYLE="${PROMPT_STYLE:-1}"

echo "[entrypoint] pool=$POOL_URL user=$USER_ARG model=$(basename "$MODEL_PATH")"
echo "[entrypoint] slots=${SLOTS:-128} groups=${GROUPS:-12} double_buffer=$MEOW_DOUBLE_BUFFER prompt_style=$POW_PROMPT_STYLE"
exec /app/supr-meow-tsc "${args[@]}"
