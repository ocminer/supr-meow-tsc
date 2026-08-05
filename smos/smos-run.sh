#!/usr/bin/env bash
# SimpleMining (SMOS) custom miner launcher for supr-meow-tsc.
#
# SMOS runs custom miners from /home/miner/custom/<name>/ and reads its
# settings from the rig config exported into the environment. Persistent data
# belongs under /home/miner, which is why the model cache defaults there —
# the package directory itself can be wiped on miner updates.
cd "$(dirname "$0")" || exit 1
[[ -f miner.conf ]] && . ./miner.conf
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/lib"

[[ -z $POOL_URL ]] && { echo "POOL_URL missing — set it in the SMOS miner config"; exit 1; }
[[ -z $WALLET   ]] && { echo "WALLET missing — set the wallet in the SMOS miner config"; exit 1; }

MODEL_DIR="${MODEL_DIR:-/home/miner/models}"
exec ./meow-common.sh
