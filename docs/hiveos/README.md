# HiveOS custom miner setup

Use the release asset named `supr-meow-tsc-0.6.0.tar.gz` as the Custom miner
installation URL. Do not add a `-hiveos` suffix or rename the archive.

Set the pool URL and wallet/worker template in the flight sheet. The launcher
downloads and verifies the required model automatically. Keep its cache under
`MODEL_DIR` (default `/hive-config/models`).

For two matching cards sharing one model, add these extra configuration lines:

```ini
DEVICES=0,1
SPLIT_MODEL=1
```

Each card must have at least 8 GB. For two matching pairs use `DEVICES=0,1,2,3`.
Do not run an additional miner on either card of a selected pair. Different GPU
types should use separate processes with distinct device lists and API ports.

Remove old `SLOTS`, `MEOW_GROUPS`, `CTX`, `EXTRA_ARGS`, `MODEL_URL` and
`MODEL_PATH` overrides on upgrade. Never use `GROUPS` as a configuration name;
it is reserved by Bash. Allow the first download and model load to complete
before expecting a mining rate.

The package includes its runtime libraries. The rig must provide a compatible
NVIDIA driver; Ampere or newer GPUs are supported. HiveOS reports the miner's
windows/s through its `khs` field, so its displayed unit is a UI convention.

See [release notes](../../RELEASE-v0.6.0.md) for the other platforms and
[operation notes](../OPERATIONS.md) for GPU numbering on mixed rigs.
