# Docker deployment

Use `ocminersupr/supr-meow-tsc:0.6.0`. Install and configure the NVIDIA
Container Toolkit on the host. Keep a persistent model cache; the required
model downloads automatically on first start.

```sh
docker run -d --name meow --gpus '"device=0"' --restart unless-stopped \
  -v meow-models:/models \
  -e POOL_URL='stratum+tcp://YOUR_POOL:PORT' \
  -e WALLET='YOUR_WALLET' -e WORKER=rig1 -e DEVICES=0 \
  ocminersupr/supr-meow-tsc:0.6.0
```

For two matching cards with at least 8 GB each, select both GPUs with
`--gpus '"device=0,1"'`, set `DEVICES=0,1` and add `SPLIT_MODEL=1`.
Use complete matching pairs. Device numbers inside the container refer to
its visible GPUs. Do not run another miner on a shared pair.

On CDI hosts, use `--device=nvidia.com/gpu=0` instead of `--gpus`, adding
`--device=nvidia.com/gpu=1` for a pair.

Set `POOL_URL`, `WALLET` and `WORKER` for your deployment. `PASSWORD` defaults
to `x`. `MODEL_DIR` defaults to `/models`; `MODEL_PATH` can select a verified
existing copy of the required model. The statistics API is disabled by
default; `API_BIND=127.0.0.1:21550` enables it inside the container.

Remove old model, slot, context and extra-argument overrides when upgrading.
See [release instructions](../RELEASE-v0.6.0.md) and
[operations](../docs/OPERATIONS.md) for package and mixed-rig setup.
