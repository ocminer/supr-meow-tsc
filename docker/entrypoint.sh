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
# OPTIONAL — SLOTS/GROUPS default to the miner's per-GPU auto-tuning; set them
# only to override it.
#   WORKER, PASSWORD, DEVICES, SLOTS, MEOW_GROUPS, CTX, API_BIND, LOG_INTERVAL,
#   MODEL_DIR, EXTRA_ARGS
#   DOUBLE_BUFFER=0|1   PROMPT_STYLE=0|1
#
# SHELL ACCESS (sshd starts only if at least one of these is set)
#   SSH_PUBKEY    authorized_keys line — preferred, no password to guess
#   SSH_PASSWORD  root password; enables PasswordAuthentication
#   SSH_PORT      default 22
set -euo pipefail

die() { echo "error: $*" >&2; exit 2; }

start_sshd() {
  mkdir -p /root/.ssh /run/sshd
  chmod 700 /root/.ssh
  ssh-keygen -A >/dev/null 2>&1

  local pw_auth=no root_login=prohibit-password mode=
  if [[ -n "${SSH_PUBKEY:-}" ]]; then
    echo "$SSH_PUBKEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    mode="key"
  fi
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    echo "root:${SSH_PASSWORD}" | chpasswd
    pw_auth=yes; root_login=yes
    mode="${mode:+$mode+}password"
  fi

  sed -i \
    -e "s/^#\?PasswordAuthentication.*/PasswordAuthentication ${pw_auth}/" \
    -e "s/^#\?PermitRootLogin.*/PermitRootLogin ${root_login}/" \
    -e "s/^#\?UsePAM.*/UsePAM yes/" \
    /etc/ssh/sshd_config
  # Ubuntu ships drop-ins under sshd_config.d that override the main file —
  # cloud images in particular disable password auth there. Without this the
  # settings above look applied but do nothing.
  mkdir -p /etc/ssh/sshd_config.d
  printf 'PasswordAuthentication %s\nPermitRootLogin %s\n' "$pw_auth" "$root_login" \
    > /etc/ssh/sshd_config.d/zz-supr-meow.conf

  if /usr/sbin/sshd -p "${SSH_PORT:-22}"; then
    echo "[entrypoint] sshd listening on ${SSH_PORT:-22} (${mode} auth)"
    [[ $pw_auth == yes ]] && echo "[entrypoint] WARNING: root password login is enabled — anyone who reaches this port can try to guess it. Use a long random SSH_PASSWORD, or prefer SSH_PUBKEY."
  else
    echo "[entrypoint] WARNING: sshd failed to start — continuing without a shell" >&2
  fi
}
[[ -n "${POOL_URL:-}" ]] || die "POOL_URL is required (stratum+tcp://host:port)"
[[ -n "${WALLET:-}"   ]] || die "WALLET is required"

MODEL_DIR="${MODEL_DIR:-/models}"
mkdir -p "$MODEL_DIR"

# ---- optional sshd (OctaSpace has no shell unless the image provides one) --
# Deliberately BEFORE the model download: a rig that cannot fetch the 15 GB
# model is exactly when a shell is needed, and starting sshd afterwards left
# precisely those rigs unreachable. Set SSH_PUBKEY (keys, preferred),
# SSH_PASSWORD (root password), or both. Neither set = no daemon, as before.
if [[ -n "${SSH_PUBKEY:-}" || -n "${SSH_PASSWORD:-}" ]]; then
  start_sshd
fi

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
# Leave these UNSET by default so the miner's own auto-tuning picks them from
# the card it actually got (src/tuning.cpp). Hardcoding the 5090 numbers here
# defeated that on every other GPU: 128 slots on a card that cannot hold their
# KV cache leaves no VRAM for the sampler scratch, and every window fails.
[[ -n "${SLOTS:-}"  ]] && args+=( --slots "$SLOTS" )
# NOT "GROUPS": that is a bash BUILT-IN array holding the caller's group ids,
# so bash overwrites any env var of that name and `${GROUPS:-12}` expands to
# "0" for root. Every containerised run therefore passed `--groups 0`, which
# made each worker group sort all the slots at once — above the GPU sort's
# 64-stream limit, so every window failed and the rig mined nothing at all.
[[ -n "${MEOW_GROUPS:-}" ]] && args+=( --groups "$MEOW_GROUPS" )
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
echo "[entrypoint] slots=${SLOTS:-auto} groups=${MEOW_GROUPS:-auto} double_buffer=$MEOW_DOUBLE_BUFFER prompt_style=$POW_PROMPT_STYLE"
exec /app/supr-meow-tsc "${args[@]}"
