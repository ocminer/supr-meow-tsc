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
#   MODEL_URL2  mirror, tried only if MODEL_URL fails
#   MODEL_SHA256  optional integrity check, applied whichever URL was used
#   MODEL_DL_JOBS parallel byte ranges for the fetch (default 8; 1 = single stream)
#
# OPTIONAL — SLOTS/GROUPS default to the miner's per-GPU auto-tuning; set them
# only to override it.
#   WORKER, PASSWORD, DEVICES, SLOTS, MEOW_GROUPS, CTX, API_BIND, LOG_INTERVAL,
#   SPLIT_MODEL=1  one model across all DEVICES (cards too small to hold it)
#   MODEL_DIR, EXTRA_ARGS
#   DOUBLE_BUFFER=0|1   PROMPT_STYLE=0|1
#
# SHELL ACCESS (sshd starts only if at least one of these is set)
#   SSH_PUBKEY    authorized_keys line — preferred, no password to guess
#   SSH_PASSWORD  root password; enables PasswordAuthentication
#   SSH_PORT      default 22
set -euo pipefail

SSHD_UP=0

# Fatal config error. If — and only if — the operator asked for a shell and we
# managed to start one, HOLD THE CONTAINER OPEN instead of exiting, so they can
# log in and fix the env. Exiting would take sshd down with us (it is a child
# of this process), which is why simply starting sshd earlier is not enough on
# its own: the one failure you most need to inspect is the one that used to
# guarantee the container was gone before you could.
#
# Without SSH the behaviour is unchanged — exit 2, crash-loop, stay visible.
# This deliberately does not paper over failures: it never holds open silently,
# and it never applies to a rig that did not ask for a shell.
die() {
  echo "error: $*" >&2
  if [[ $SSHD_UP == 1 ]]; then
    echo "[entrypoint] NOT MINING — config error above. sshd is up; log in and fix the environment, then restart the container." >&2
    while true; do
      echo "[entrypoint] idle: $* (waiting for you to fix it — this rig is NOT earning)" >&2
      sleep 300
    done
  fi
  exit 2
}

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
    SSHD_UP=1
    echo "[entrypoint] sshd listening on ${SSH_PORT:-22} (${mode} auth)"
    # MUST be `if`, never `[[ ... ]] && echo`. As the last statement of this
    # function that becomes its return value, and with key-only auth pw_auth is
    # "no", so the list returns 1, `start_sshd` returns 1, and `set -e` killed
    # the container right after printing "sshd listening". SSH_PUBKEY — the
    # option this very warning recommends — was therefore fatal, while
    # SSH_PASSWORD worked, which is why the fleet never hit it.
    if [[ $pw_auth == yes ]]; then
      echo "[entrypoint] WARNING: root password login is enabled — anyone who reaches this port can try to guess it. Use a long random SSH_PASSWORD, or prefer SSH_PUBKEY."
    fi
  else
    echo "[entrypoint] WARNING: sshd failed to start — continuing without a shell" >&2
  fi
  return 0   # never let this function's status abort the miner
}
# Parallel ranged model fetch.
#
# A single-stream curl measured 7 MiB/s on an OctaSpace rig — 29 minutes for
# the 16.4 GB model, with the GPU sitting idle the whole time. The same file
# over 8 ranges lands in about 7. Every rig downloads independently, so this is
# ~22 minutes off EVERY cold start, which dwarfs any tuning gain in the miner.
#
# Ranges are written straight to their offset in the destination, so peak disk
# is ONE file, not two. An earlier parallel fetcher wrote 8 parts and cat-ed
# them, needing 2x16.4 GB on a 23 GB box; the concatenation silently truncated
# and only the final sha256 caught it.
#
# Degrades to a single stream whenever anything is unclear — no Content-Length,
# no Accept-Ranges, a failed part. A slow download is an annoyance; a wrong one
# is a rig mining garbage, and MODEL_SHA256 is checked either way.
fetch_model() {   # <url> <destination> [mirror]
  # Ranges are split across BOTH sources CONCURRENTLY, not primary-then-fallback.
  # Measured 2026-08-06: Hugging Face opens at ~21 MB/s and decays to 0.6-5 MB/s
  # under sustained parallel load (per-source throttling), while the suprnova
  # mirror holds ~6.8 MB/s flat. Pulling from both at once adds their bandwidth
  # instead of queueing behind the slower one. On churn-heavy spot instances the
  # 16.4 GB fetch is pure unpaid setup time, so this is the highest-value knob.
  local url="$1" out="$2" mirror="${3:-}" jobs="${MODEL_DL_JOBS:-8}"
  local size=""

  single_stream() {
    echo "[entrypoint] downloading model (single stream): $1"
    rm -f "$2.part"
    curl -fL --retry 5 --retry-delay 5 -o "$2.part" "$1" || return 1
    mv "$2.part" "$2"
  }

  # Probe EVERY source independently. A single failed HEAD must not cost the
  # 8-way path: measured 2026-08-06, one flaky HEAD dropped the fetch to a
  # single ~1 MB/s stream — a 4.5 HOUR download for a file the ranged path
  # pulls in minutes. On a spot box that churns in ~20 minutes that is the
  # difference between earning and never starting. So the size may come from
  # whichever source answers; they are cross-checked below.
  probe_size() {   # <url> -> "<size> <ranges>" on stdout, empty on failure
    local h
    h=$(curl -fsSLI --retry 3 --retry-delay 2 --max-time 45 "$1" 2>/dev/null) || return 1
    printf '%s %s' \
      "$(printf '%s' "$h" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')" \
      "$(printf '%s' "$h" | tr -d '\r' | awk 'tolower($1)=="accept-ranges:"{v=tolower($2)} END{print v}')"
  }

  local -a srcs=() cand=("$url"); [[ -n "$mirror" ]] && cand+=("$mirror")
  local u psize pranges
  for u in "${cand[@]}"; do
    read -r psize pranges <<<"$(probe_size "$u")"
    [[ -z "$psize" || ! "$psize" =~ ^[0-9]+$ || "$psize" -lt 1048576 || "$pranges" != "bytes" ]] && {
      echo "[entrypoint] no usable byte-range support from $u"; continue; }
    if [[ -z "$size" ]]; then
      size="$psize"; srcs+=("$u")
    elif [[ "$psize" == "$size" ]]; then
      # Identical length is the ONLY admission test. Splicing ranges from two
      # different builds yields a file corrupt in a way nothing but the final
      # sha256 would catch, after the whole download has been paid for.
      srcs+=("$u")
    else
      echo "[entrypoint] size mismatch from $u ($psize vs $size) — excluded"
    fi
  done

  if (( ${#srcs[@]} == 0 )) || (( jobs < 2 )); then
    echo "[entrypoint] no ranged source available — falling back to a single stream"
    single_stream "$url" "$out"; return $?
  fi
  (( ${#srcs[@]} > 1 )) && echo "[entrypoint] ${#srcs[@]} sources agree on size — fetching in parallel"

  echo "[entrypoint] downloading model: $((size/1048576)) MiB over $jobs ranges from ${#srcs[@]} source(s)"
  rm -f "$out.part"
  fallocate -l "$size" "$out.part" 2>/dev/null || truncate -s "$size" "$out.part" || { single_stream "$url" "$out"; return $?; }

  local chunk=$(( (size + jobs - 1) / jobs )) i lo hi rc=0
  local -a pids=()
  for (( i=0; i<jobs; i++ )); do
    lo=$(( i * chunk )); hi=$(( lo + chunk - 1 ))
    (( hi >= size )) && hi=$(( size - 1 ))
    # Alternate ranges across sources so both links are saturated at once.
    local pri="${srcs[i % ${#srcs[@]}]}" alt=""
    (( ${#srcs[@]} > 1 )) && alt="${srcs[(i + 1) % ${#srcs[@]}]}"
    (
      # pipefail is REQUIRED: without it the subshell reports dd's status, and
      # dd happily "succeeds" on curl's empty output. The old size check could
      # never catch that either, because .part is pre-allocated to the full
      # length — so a dead range slipped through to the sha256 and killed the
      # whole fetch. With pipefail a failed range is seen here and retried on
      # the OTHER source, which is exactly what a second source is for.
      set -o pipefail
      # -L is MANDATORY: Hugging Face resolve/ URLs answer 302 to the CDN, and
      # -f does NOT treat 302 as an error — without -L every range "succeeds"
      # instantly with a ~1 KB redirect body and the file stays zeros.
      curl -fsSL --retry 3 --retry-delay 3 --max-time 3600 -r "${lo}-${hi}" "$pri" \
        | dd of="$out.part" bs=4M seek="$lo" oflag=seek_bytes conv=notrunc status=none && exit 0
      [[ -n "$alt" ]] || exit 1
      echo "[entrypoint] range ${lo}-${hi} failed on primary — retrying on mirror" >&2
      curl -fsSL --retry 3 --retry-delay 3 --max-time 3600 -r "${lo}-${hi}" "$alt" \
        | dd of="$out.part" bs=4M seek="$lo" oflag=seek_bytes conv=notrunc status=none
    ) &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done

  if (( rc != 0 )); then
    echo "[entrypoint] ranged fetch incomplete — falling back to a single stream"
    single_stream "$url" "$out"; return $?
  fi
  mv "$out.part" "$out"
}

# ---- optional sshd (OctaSpace has no shell unless the image provides one) --
# FIRST, before ANY validation or the model download. Two reasons, both learned
# the hard way:
#   * a rig that cannot fetch the 15 GB model is exactly when a shell is
#     needed, and starting sshd afterwards left precisely those rigs
#     unreachable;
#   * a typo in POOL_URL or WALLET used to `die` here, killing the container
#     with no shell — so the one misconfiguration you most need to log in and
#     inspect was the one that guaranteed you could not. On a platform where
#     env vars are the only config channel, that is the difference between a
#     30-second fix and destroying the instance.
# Set SSH_PUBKEY (keys, preferred), SSH_PASSWORD (root password), or both.
# Neither set = no daemon, as before.
if [[ -n "${SSH_PUBKEY:-}" || -n "${SSH_PASSWORD:-}" ]]; then
  start_sshd
fi

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
      # MODEL_URL2 is passed IN so its ranges run concurrently with the
      # primary's, not merely as a retry after the primary has already
      # burned minutes failing. The whole-mirror retry stays as the last
      # resort for the case where the primary is unusable outright.
      fetch_model "$MODEL_URL" "$MODEL_PATH" "${MODEL_URL2:-}" \
        || { [[ -n "${MODEL_URL2:-}" ]] \
               && echo "[entrypoint] primary failed — trying MODEL_URL2 alone" \
               && fetch_model "$MODEL_URL2" "$MODEL_PATH"; } \
        || die "model download failed"
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
# Small cards: spread ONE model over every selected GPU and mine as a single
# worker, instead of a full copy per card. For GPUs that cannot hold the
# 15.3 GB model alone (2x12, 4x8, 8x6 GB). Aggregates VRAM, does not add
# throughput, and the cards should be identical. Do NOT also run one container
# per GPU — they would fight over the same cards.
[[ "${SPLIT_MODEL:-0}" == "1" ]] && args+=( --split-model )
[[ "${SPLIT_ROWS:-0}"  == "1" ]] && args+=( --split-rows )
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
