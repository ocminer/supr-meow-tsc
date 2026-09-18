# supr-meow-tsc v0.7.0

Adds support for **proof v4**, the format required by the TensorCash network
upgrade. Update to this version to keep producing valid work: after the v3
sunset height, the network no longer accepts proofs from older miners.

## What changed

- **Proof v4 support (required).** This version emits the new proof format the
  upgraded network verifies. It is the only change miners need for the upgrade;
  pool URL, wallet and device options are unchanged from v0.6.0.
- **Native Windows x64 package**, built through GitHub CI, with model download and launch scripts.
- Packaged launchers default to the standard Q8 profile and load their bundled libraries.

**You must upgrade before the v3 sunset.** A miner older than v0.7.0 keeps
running but its work stops being accepted once the network completes the switch.

## Downloads

| Platform | File |
|---|---|
| HiveOS | `supr-meow-tsc-0.7.0.tar.gz` — do not rename |
| MMPOS | `supr-meow-tsc-0.7.0-mmpos.tar.gz` |
| SimpleMining | `supr-meow-tsc-0.7.0-smos.tar.gz` |
| Linux x86-64 | `supr-meow-tsc-0.7.0-linux-x86_64.tar.gz` |
| Windows x64 | `supr-meow-tsc-0.7.0-windows-x86_64.zip` |
| Docker | `ocminersupr/supr-meow-tsc:0.7.0` |
| Checksums | `supr-meow-tsc-0.7.0.sha256` |

Use an NVIDIA Ampere or newer GPU and a compatible current NVIDIA driver.
The first start downloads the required model; keep the model cache between runs.

## Linux

Extract the Linux package and run from its directory:

```sh
POOL_URL='stratum+tcp://tsc.suprnova.cc:3310' \
WALLET='YOUR_WALLET.rig1' DEVICES=0 ./meow-common.sh
```

For a shared model across two matching cards with at least 8 GB each:

```sh
POOL_URL='stratum+tcp://tsc.suprnova.cc:3310' \
WALLET='YOUR_WALLET.rig1' DEVICES=0,1 SPLIT_MODEL=1 ./meow-common.sh
```

Select complete matching pairs: `0,1` for one pair or `0,1,2,3` for two pairs.
Each pair shares a model. Do not launch a separate mining process on each card
of a shared pair. On mixed rigs, use separate processes for different GPU types
and give each process its own selected devices.

## HiveOS, MMPOS and SimpleMining

Set the pool and wallet in the miner configuration. For shared-card mining,
add these extra configuration lines:

```ini
DEVICES=0,1
SPLIT_MODEL=1
```

Remove old `Q8_PROFILE`, `SLOTS`, `MEOW_GROUPS`, `CTX`, `EXTRA_ARGS`, `MODEL_URL` and
`MODEL_PATH` overrides when upgrading, so the new defaults and required model
are used. `MODEL_DIR` can select a persistent cache directory. For HiveOS, use
the exact unsuffixed archive name shown above as the installation URL.

## Native Windows

Extract the complete Windows ZIP, edit `start.cmd` to set your own TSC wallet,
and run it. Or use PowerShell:

```powershell
.\start.ps1 -Wallet YOUR_TSC_WALLET -Worker rig1 -Devices 0
```

Use `-Devices '0,1' -SplitModel` for a matching pair. The package is native
Windows x64 and requires an NVIDIA driver compatible with CUDA 13.3. The first
start downloads and verifies the pinned Q8 model. WSL and Docker are not required.

## Docker

Install and configure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host first.

```sh
docker run --rm --gpus '"device=0,1"' \
  -v meow-models:/models \
  -e POOL_URL='stratum+tcp://tsc.suprnova.cc:3310' \
  -e WALLET='YOUR_WALLET' -e WORKER=rig1 \
  -e DEVICES=0,1 -e SPLIT_MODEL=1 \
  ocminersupr/supr-meow-tsc:0.7.0
```

For one card, use `--gpus '"device=0"'`, `DEVICES=0` and omit `SPLIT_MODEL`.
Device numbers inside a restricted container refer to its visible GPUs.
On hosts using CDI, replace `--gpus` with `--device=nvidia.com/gpu=0`
(and `--device=nvidia.com/gpu=1` for the second card).

## Upgrade and pool identification

Stop the old miner before starting this version. The standard mining profile
submits user agent **`supr-meow-tsc/0.7.0`**.

## Validation

The Linux miner has produced accepted shares with live proof-v4 verification.
Native Windows CI checks proof-v4 roots, deterministic sampler arithmetic,
VDF proof binding and serialization against Linux, and HTTP/Stratum sockets.
The hosted Windows runner has no NVIDIA GPU, so Windows GPU mining is not
hardware-tested by CI. Individual proofs can still fail network content gates.

The v3 sunset is **block 27615**; upgrade before that height. Standard pool
settings continue to work, including `stratum+tcp://tsc.suprnova.cc:3310`.
