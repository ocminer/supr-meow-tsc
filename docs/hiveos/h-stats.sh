#!/usr/bin/env bash
# Maps the miner's local JSON stats API onto the variables the HiveOS agent
# expects: $khs (total hashrate in kH/s) and $stats (JSON).
#
# TSC rates are PoI/s (proofs per second), single/double digits — they are
# reported through the standard fields with hs_units="hs" so the dashboard
# shows the true number.
source /hive/miners/custom/supr-meow-tsc/h-manifest.conf 2>/dev/null

api="http://127.0.0.1:${CUSTOM_API_PORT:-21550}/"
j=$(curl -m 3 -s "$api")

if [[ -z $j ]]; then
  khs=0
  stats="null"
else
  khs=$(jq -r '(.total_poi_s // 0) / 1000' <<< "$j")
  stats=$(jq -c '{
      hs:        [ .gpus[].poi_s ],
      hs_units:  "hs",
      temp:      [ .gpus[].temp ],
      fan:       [ .gpus[].fan ],
      uptime:    .uptime,
      ver:       .ver,
      ar:        [ .accepted, .rejected ],
      algo:      .algo,
      bus_numbers: [ .gpus[].bus_id ]
    }' <<< "$j")
fi

[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats="null"
