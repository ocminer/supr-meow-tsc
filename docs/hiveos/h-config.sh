#!/usr/bin/env bash
# Generates the miner command line from the HiveOS flight sheet.
# Flight-sheet fields map as:
#   Pool URL      -> %URL%     ($CUSTOM_URL)
#   Wallet/worker -> %WAL%     ($CUSTOM_TEMPLATE)
#   Extra config  -> pass-through arguments, e.g.:
#       --model /path/to/Qwen3-0.6B-bf16.gguf --slots 48 --groups 8
[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${RED}CUSTOM_TEMPLATE (wallet) is empty${NOCOLOR}" && return 1
[[ -z $CUSTOM_URL ]]      && echo -e "${RED}CUSTOM_URL (pool) is empty${NOCOLOR}" && return 1

conf="-o $CUSTOM_URL -u $CUSTOM_TEMPLATE --no-color --api-bind 127.0.0.1:${CUSTOM_API_PORT:-21550}"
[[ -n $CUSTOM_PASS ]] && conf+=" -p $CUSTOM_PASS"
[[ -n $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf" > "$CUSTOM_CONFIG_FILENAME"
