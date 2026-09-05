#!/usr/bin/env bash
# Sourced by launchers only when Q8_PROFILE=1. The miner independently verifies
# this same SHA-256, including when MODEL_PATH points at a pre-existing file.
MODEL_DIR="${MODEL_DIR:-./models}"
if [[ -z ${MODEL_PATH:-} ]]; then
  MODEL_PATH="$MODEL_DIR/Qwen3-8B-Q8_0-full-v2.gguf"
  if [[ ! -s $MODEL_PATH ]]; then
    (
      set -euo pipefail
      mkdir -p -- "$MODEL_DIR" || exit 1
      q8_part=$(mktemp "$MODEL_DIR/.q8-download.XXXXXX") || exit 1
      trap 'rm -f -- "$q8_part"' EXIT
      q8_url="${MODEL_URL:-https://huggingface.co/luckypoolio/Qwen3-8B-Q8_0-full-v2.gguf/resolve/850bbb3a59d5c855c5d7b78831ff823ca1791cfb/Qwen3-8B-Q8_0-full-v2.gguf}"
      curl --fail --location --proto '=https' --proto-redir '=https' --retry 5 \
        --output "$q8_part" "$q8_url" || exit 1
      printf '%s  %s\n' '30f3a9df384a08453ff0bf5715489e174e986cd033af2a9430e4ddb039cd73c7' "$q8_part" | sha256sum --check - || exit 1
      mv -- "$q8_part" "$MODEL_PATH" || exit 1
    ) || return 1
  fi
fi
