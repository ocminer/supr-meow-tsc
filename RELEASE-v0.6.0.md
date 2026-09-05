# supr-meow-tsc v0.6.0

Faster mining and support for sharing one model across matching GPUs with
smaller memory. This release also improves device selection and shutdown.

| Hardware | Mining rate |
|---|---:|
| RTX 5090 32 GB | 34.0 windows/s |
| 2 × RTX 3070 8 GB, shared model | 7.2 windows/s combined |

Rates vary with hardware, clocks and operating conditions.

## Downloads

| Platform | File |
|---|---|
| HiveOS | `supr-meow-tsc-0.6.0.tar.gz` — do not rename |
| MMPOS | `supr-meow-tsc-0.6.0-mmpos.tar.gz` |
| SimpleMining | `supr-meow-tsc-0.6.0-smos.tar.gz` |
| Linux x86-64 | `supr-meow-tsc-0.6.0-linux-x86_64.tar.gz` |
| Docker | `ocminersupr/supr-meow-tsc:0.6.0` |
| Checksums | `supr-meow-tsc-0.6.0.sha256` |

Use an NVIDIA Ampere or newer GPU and a compatible current NVIDIA driver.
The first start downloads the required model; keep the model cache between runs.

## Linux

Extract the Linux package and run from its directory:

```sh
POOL_URL='stratum+tcp://YOUR_POOL:PORT' \
WALLET='YOUR_WALLET.rig1' DEVICES=0 ./meow-common.sh
```

For a shared model across two matching cards with at least 8 GB each:

```sh
POOL_URL='stratum+tcp://YOUR_POOL:PORT' \
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

Remove old `SLOTS`, `MEOW_GROUPS`, `CTX`, `EXTRA_ARGS`, `MODEL_URL` and
`MODEL_PATH` overrides when upgrading, so the new defaults and required model
are used. `MODEL_DIR` can select a persistent cache directory. For HiveOS, use
the exact unsuffixed archive name shown above as the installation URL.

## Docker

Install and configure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host first.

```sh
docker run --rm --gpus '"device=0,1"' \
  -v meow-models:/models \
  -e POOL_URL='stratum+tcp://YOUR_POOL:PORT' \
  -e WALLET='YOUR_WALLET' -e WORKER=rig1 \
  -e DEVICES=0,1 -e SPLIT_MODEL=1 \
  ocminersupr/supr-meow-tsc:0.6.0
```

For one card, use `--gpus '"device=0"'`, `DEVICES=0` and omit `SPLIT_MODEL`.
Device numbers inside a restricted container refer to its visible GPUs.
On hosts using CDI, replace `--gpus` with `--device=nvidia.com/gpu=0`
(and `--device=nvidia.com/gpu=1` for the second card).

## Upgrade and pool identification

Stop the old miner before starting this version. Users of the older vLLM image
should switch to the image or packages above. The standard mining profile
submits user agent **`supr-meow-tsc/0.6.0`**.
