#!/usr/bin/env bash
# MMPOS custom miner launcher for supr-meow-tsc.
#
# MMPOS passes miner settings as environment variables and captures stdout,
# same shape as HiveOS, so this is the HiveOS launcher with the two
# platform-specific bits changed: the config file it sources and the default
# model cache directory (MMPOS keeps persistent data under /hive-config on
# some images and /mmpos on others — both are probed).
cd "$(dirname "$0")" || exit 1
[[ -f miner.conf ]] && . ./miner.conf
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/lib"

[[ -z $POOL_URL ]] && { echo "POOL_URL missing — set it in the MMPOS miner config"; exit 1; }
[[ -z $WALLET   ]] && { echo "WALLET missing — set the wallet in the MMPOS miner config"; exit 1; }

if [[ -z $MODEL_DIR ]]; then
  for d in /mmpos/models /hive-config/models "$(pwd)/models"; do
    [[ -d $(dirname "$d") ]] && MODEL_DIR="$d" && break
  done
  MODEL_DIR="${MODEL_DIR:-$(pwd)/models}"
fi
exec ./meow-common.sh
