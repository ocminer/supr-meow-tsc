# HiveOS integration

`supr-meow-tsc` exposes a local JSON stats API (default `127.0.0.1:21550`,
`--api-bind` to change, `--api-bind off` to disable). Any GET returns:

```json
{
  "name": "supr-meow-tsc", "ver": "0.2.0", "algo": "tsc-poi",
  "uptime": 3600, "model": "Qwen/Qwen3-0.6B@c1899de289a0",
  "pool": "stratum+tcp://pool:3307", "connected": true,
  "accepted": 1234, "rejected": 0, "stale": 4, "total_poi_s": 90.2,
  "gpus": [{
    "id": 1, "name": "NVIDIA GeForce RTX 5090", "bus_id": 7,
    "poi_s": 90.2, "windows": 320000,
    "accepted": 1234, "rejected": 0, "stale": 4,
    "temp": 62, "fan": 40, "power": 294,
    "core_mhz": 2500, "mem_mhz": 13801, "util": 82
  }]
}
```

## Packaging for HiveOS

```
supr-meow-tsc/           <- directory name must equal CUSTOM_NAME
  supr-meow-tsc          <- the binary
  h-manifest.conf
  h-config.sh
  h-run.sh
  h-stats.sh
```

Archive it as `supr-meow-tsc-0.2.0.tar.gz` and either place it on a web
server and point the flight sheet's "Installation URL" at it, or drop the
directory into `/hive/miners/custom/` by hand.

Flight sheet:
- **Miner**: Custom, name `supr-meow-tsc`
- **Pool URL**: `stratum+tcp://host:port`
- **Wallet and worker template**: `%WAL%.%WORKER_NAME%`
- **Extra config arguments**: everything else, at minimum the model path:
  `--model /path/to/model.gguf --slots 48 --groups 8`

Notes:
- Rates are **PoI/s** (proofs per second), not hashes; `h-stats.sh` reports
  them with `hs_units:"hs"` so the dashboard shows the real number.
- `h-stats.sh` needs `jq` and `curl` (both ship with HiveOS).
- The miner needs the chain-registered GGUF model on disk; ~1.5 GB (testnet
  0.6B) to ~16 GB (mainnet 8B) — put it on the rig's SSD, not on USB.
