# Mining operation

Use the packaged launchers to download and verify the required model. Keep its
cache on persistent storage with `MODEL_DIR`. If supplying `MODEL_PATH`, use the
exact model required by this version; incompatible files are rejected.

For a matching pair, select both cards in one process:

```ini
DEVICES=0,1
SPLIT_MODEL=1
```

For four matching cards, `DEVICES=0,1,2,3 SPLIT_MODEL=1` forms two pairs.
The cards within each adjacent pair must match. Work and shares from a pair are
reported against its first card; a zero work counter on the second does not
mean that it is idle.

CUDA and system GPU numbering can differ on mixed rigs. To share a rig with
another miner, use GPU UUIDs from `nvidia-smi -L` in `CUDA_VISIBLE_DEVICES`, then
select `DEVICES=0,1` within that visible list. Give separate processes distinct
`API_PORT` values. Do not select a card already used by another workload.

On upgrade, remove old performance and model overrides, then let the new
version choose defaults. Stop the old process before launching the replacement.
For normal shutdown, send SIGINT and wait for the process to exit.
