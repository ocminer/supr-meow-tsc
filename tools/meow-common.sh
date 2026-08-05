#!/usr/bin/env bash
# Shared launch logic for the mining-OS packages (HiveOS / MMPOS / SMOS).
#
# Each OS wrapper sets POOL_URL, WALLET and MODEL_DIR from its own config
# format, then execs this. Keeping one copy means a fix to the model fetch or
# the argument mapping reaches every platform instead of three drifting forks.
set -u

MODEL_DIR="${MODEL_DIR:-$(pwd)/models}"
if [[ -z ${MODEL_PATH:-} ]]; then
  if [[ -n ${MODEL_URL:-} ]]; then
    mkdir -p "$MODEL_DIR"
    MODEL_PATH="$MODEL_DIR/$(basename "${MODEL_URL%%\?*}")"
    if [[ ! -s $MODEL_PATH ]]; then
      echo "[supr-meow-tsc] downloading model (16.4 GB, one time): $MODEL_URL"
      # -L is MANDATORY: Hugging Face resolve/ URLs answer 302 and curl -f does
      # NOT treat 302 as an error, so without it every range "succeeds" with a
      # ~1 KB redirect body and the file stays zeros.
      SIZE=$(curl -fsSLI "$MODEL_URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
      if [[ -n $SIZE && $SIZE -gt 1048576 ]]; then
        truncate -s "$SIZE" "$MODEL_PATH.part" 2>/dev/null || fallocate -l "$SIZE" "$MODEL_PATH.part"
        N=8; chunk=$(( (SIZE + N - 1) / N ))
        for i in $(seq 0 $((N-1))); do
          lo=$(( i * chunk )); hi=$(( lo + chunk - 1 )); [[ $hi -ge $SIZE ]] && hi=$((SIZE-1))
          curl -fsSL --retry 5 --retry-delay 3 -r "${lo}-${hi}" "$MODEL_URL" \
            | dd of="$MODEL_PATH.part" bs=4M seek="$lo" oflag=seek_bytes conv=notrunc status=none &
        done
        wait
      else
        curl -fL --retry 5 --no-progress-meter -o "$MODEL_PATH.part" "$MODEL_URL" || exit 1
      fi
      mv "$MODEL_PATH.part" "$MODEL_PATH"
    fi
    if [[ -n ${MODEL_SHA256:-} ]]; then
      echo "[supr-meow-tsc] verifying model checksum..."
      echo "${MODEL_SHA256}  ${MODEL_PATH}" | sha256sum -c - || {
        echo "[supr-meow-tsc] CHECKSUM MISMATCH — refusing to mine with an unverified model"
        rm -f "$MODEL_PATH"; exit 1; }
    fi
  else
    MODEL_PATH=$(find "$MODEL_DIR" -maxdepth 1 -name '*.gguf' 2>/dev/null | head -1)
  fi
fi
[[ -s ${MODEL_PATH:-} ]] || { echo "[supr-meow-tsc] no model — set MODEL_URL or MODEL_PATH"; exit 1; }

args=( -o "$POOL_URL" -u "$WALLET" -p "${PASSWORD:-x}" --model "$MODEL_PATH" --no-color )
args+=( --api-bind "127.0.0.1:${API_PORT:-21550}" )
# Leave slots/groups unset so the per-GPU auto-tuning applies.
# NEVER read $GROUPS: it is a bash built-in array and always arrives as "0".
[[ -n ${SLOTS:-}       ]] && args+=( --slots  "$SLOTS" )
[[ -n ${MEOW_GROUPS:-} ]] && args+=( --groups "$MEOW_GROUPS" )
[[ -n ${CTX:-}         ]] && args+=( --ctx    "$CTX" )
[[ -n ${DEVICES:-}     ]] && args+=( -d       "$DEVICES" )
[[ ${SPLIT_MODEL:-0} == 1 ]] && args+=( --split-model )
[[ -n ${EXTRA_ARGS:-}  ]] && args+=( ${EXTRA_ARGS} )

echo "[supr-meow-tsc] pool=$POOL_URL model=$(basename "$MODEL_PATH")"
exec ./supr-meow-tsc "${args[@]}"
