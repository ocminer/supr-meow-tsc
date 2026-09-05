#!/usr/bin/env bash
# Report stats to the HiveOS agent. Sourced, so it must SET $khs and $stats
# rather than print them, and must never exit.
#
# UNITS: this miner's rate is proof-of-inference windows per second, not
# hashes. HiveOS only understands khs, so w/s is reported there directly — a
# rig showing "26 kH/s" is doing 26 windows/s. The true figure is also in
# $stats as poi_s.
. /hive/miners/custom/supr-meow-tsc/h-manifest.conf 2>/dev/null
API_PORT="${API_PORT:-21550}"
[[ -f ${CUSTOM_CONFIG_FILENAME} ]] && API_PORT=$(grep -m1 '^API_PORT=' "$CUSTOM_CONFIG_FILENAME" 2>/dev/null | cut -d= -f2)
API_PORT="${API_PORT:-21550}"

khs=0
stats=""

raw=$(curl -s --connect-timeout 2 --max-time 4 "http://127.0.0.1:${API_PORT}/" 2>/dev/null)

# Miner not up yet, or mid-restart: report zero rather than garbage. `return`
# works because HiveOS sources this; the `exit` is for running it by hand.
if [[ -z $raw ]] || ! jq -e . >/dev/null 2>&1 <<< "$raw"; then
  return 0 2>/dev/null || exit 0
fi

khs=$(jq -r '.total_poi_s // 0' <<< "$raw")

# HiveOS wants bus ids as hex "01:00.0"; the API reports the decimal bus.
stats=$(jq -c \
  --argjson up "$(jq -r '.uptime // 0' <<< "$raw")" \
  '{
     hs:        [ .gpus[]? | .poi_s ],
     hs_units:  "khs",
     temp:      [ .gpus[]? | .temp ],
     fan:       [ .gpus[]? | .fan ],
     uptime:    $up,
     ver:       .ver,
     algo:      .algo,
     ar:        [ (.accepted // 0), (.rejected // 0), (.stale // 0) ],
     bus_numbers: [ .gpus[]? | .bus_id ]
   }' <<< "$raw")
