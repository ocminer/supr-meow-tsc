#!/usr/bin/env bash
# Launch the miner. cwd is the package directory; HiveOS captures stdout.
cd "$(dirname "$0")" || exit 1
. h-manifest.conf
[[ -f $CUSTOM_CONFIG_FILENAME ]] && . "$CUSTOM_CONFIG_FILENAME"

# Our bundled CUDA runtime and llama libs. The DRIVER (libcuda, libnvidia-ml)
# comes from HiveOS itself and must keep winning, so our dir goes last.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/lib"

[[ -z $POOL_URL ]] && echo "POOL_URL missing — set the pool in the flight sheet" && exit 1
[[ -z $WALLET   ]] && echo "WALLET missing — set the wallet template in the flight sheet" && exit 1

# ---- model: 16.4 GB, fetched once and cached -------------------------------
MODEL_DIR="${MODEL_DIR:-/hive-config/models}"
if [[ -z $MODEL_PATH ]]; then
  if [[ -n $MODEL_URL ]]; then
    mkdir -p "$MODEL_DIR"
    MODEL_PATH="$MODEL_DIR/$(basename "${MODEL_URL%%\?*}")"
    if [[ ! -s $MODEL_PATH ]]; then
      echo "[supr-meow-tsc] downloading model (16.4 GB, one time): $MODEL_URL"
      # Eight parallel ranges: a single stream measured 21 Mbit/s decaying to 3
      # from some hosts, which is 12 hours for this file. Written straight to
      # offsets so peak disk is one copy, not two.
      SIZE=$(curl -fsSLI "$MODEL_URL" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -1)
      if [[ -n $SIZE && $SIZE -gt 0 ]]; then
        truncate -s "$SIZE" "$MODEL_PATH.part" 2>/dev/null || fallocate -l "$SIZE" "$MODEL_PATH.part"
        N=8; chunk=$(( (SIZE + N - 1) / N ))
        for i in $(seq 0 $((N-1))); do
          lo=$(( i * chunk )); hi=$(( lo + chunk - 1 )); [[ $hi -ge $SIZE ]] && hi=$((SIZE-1))
          curl -fsS --retry 5 --retry-delay 3 -r "${lo}-${hi}" "$MODEL_URL" \
            | dd of="$MODEL_PATH.part" bs=4M seek="$lo" oflag=seek_bytes conv=notrunc status=none &
        done
        wait
      else
        curl -fL --retry 5 --no-progress-meter -o "$MODEL_PATH.part" "$MODEL_URL" || exit 1
      fi
      mv "$MODEL_PATH.part" "$MODEL_PATH"
    fi
    if [[ -n $MODEL_SHA256 ]]; then
      echo "[supr-meow-tsc] verifying model checksum..."
      echo "${MODEL_SHA256}  ${MODEL_PATH}" | sha256sum -c - || {
        echo "[supr-meow-tsc] CHECKSUM MISMATCH — refusing to mine with an unverified model"
        rm -f "$MODEL_PATH"; exit 1; }
    fi
  else
    MODEL_PATH=$(find "$MODEL_DIR" -maxdepth 1 -name '*.gguf' 2>/dev/null | head -1)
  fi
fi
[[ -s $MODEL_PATH ]] || { echo "[supr-meow-tsc] no model — set MODEL_URL or MODEL_PATH in the flight sheet's extra config"; exit 1; }

args=( -o "$POOL_URL" -u "$WALLET" -p "${PASSWORD:-x}" --model "$MODEL_PATH" --no-color )
args+=( --api-bind "127.0.0.1:${API_PORT:-21550}" )
# Unset by default so per-GPU auto-tuning applies. NEVER read $GROUPS here:
# it is a bash built-in array and always arrives as "0".
[[ -n $SLOTS       ]] && args+=( --slots  "$SLOTS" )
[[ -n $MEOW_GROUPS ]] && args+=( --groups "$MEOW_GROUPS" )
[[ -n $CTX         ]] && args+=( --ctx    "$CTX" )
[[ -n $DEVICES     ]] && args+=( -d       "$DEVICES" )
[[ -n $EXTRA_ARGS  ]] && args+=( ${EXTRA_ARGS} )

export MEOW_DOUBLE_BUFFER="${DOUBLE_BUFFER:-0}"
export POW_PROMPT_STYLE="${PROMPT_STYLE:-1}"
exec ./supr-meow-tsc "${args[@]}"
