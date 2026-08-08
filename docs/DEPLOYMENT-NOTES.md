# Deployment notes

Practical lessons from running this miner across a lot of rented multi-GPU
boxes. Each one below cost real hashrate before it was understood, and most of
them are **silent** — the miner looks healthy while something is wrong. If you
are deploying more than one or two rigs, this page is worth ten minutes.

For hardware requirements see [COMPATIBILITY.md](COMPATIBILITY.md); for tuning
and measured curves see [BENCHMARKS.md](BENCHMARKS.md).

---

## 1. Pin the image tag. Never deploy `:latest`

**This is the single most expensive mistake on this page.**

`:latest` is resolved **once**, when the container is created, and that container
keeps the image it got — forever. Two containers created thirty minutes apart can
therefore run **different builds** of the miner, indefinitely, with no indication
in `docker ps`.

Seen in practice: on one 8-GPU box, six containers were created from an explicit
version tag and two from `:latest` half an hour later, at a moment when `:latest`
still pointed at the previous release. Those two ran a build from *before* a
reject fix and produced ~2.3% rejected shares for two days, while their six
siblings on identical hardware produced ~0%. It looked exactly like two failing
GPUs — temperatures, power draw, PCIe link, ECC counters and throttle reasons
were all clean, and the "bad" cards were actually clocking *higher*.

**Do:**

```bash
docker run -d --name meow0 ... ocminersupr/supr-meow-tsc:0.4.0
```

**When a single container misbehaves, check what it is actually running before
you suspect the hardware:**

```bash
docker logs meow0 2>&1 | grep -oE "supr-meow-tsc [0-9]+\.[0-9]+\.[0-9]+" | tail -1
docker inspect meow0 --format '{{.Config.Image}} {{.Image}}'
```

Compare the **image ID**, not just the tag — two containers can show the same
tag string and different underlying images.

## 2. Verify the model by checksum, never by file size

If your fetcher preallocates the output (`fallocate`/`truncate`, which parallel
range downloaders normally do), then **a failed download leaves a full-size
file**. `ls -l`, `stat`, and even allocated block counts all report a complete
16.4 GB model that is partly garbage.

The dangerous consequence is a deploy script that guards its download with an
existence test:

```bash
if [ ! -s "$MODEL" ]; then fetch; fi     # WRONG after a failed fetch
```

On a retry this skips the download **and skips verification**, then starts mining
on a corrupt model. The miner cannot detect this — it will happily submit shares.

```bash
# RIGHT
model_ok(){ [ -s "$MODEL" ] && [ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$SHA" ]; }
if ! model_ok; then fetch; model_ok || exit 1; fi
```

The Docker entrypoint already checksum-verifies and refuses to mine on a
mismatch. This applies if you roll your own fetching.

## 3. `curl --retry` does not fire on a stalled transfer

`--retry` reacts to **errors**. A connection that stays open and simply stops
delivering bytes is not an error, so curl waits forever and your deploy hangs
with no failure message.

```bash
curl -fsSL --retry 5 \
     --speed-limit 1048576 --speed-time 30 \      # abort under 1 MB/s for 30 s
     -o "$OUT" "$URL"
```

Without the speed guard, several rigs sat at ~0.1 MB/s for ten minutes and then
failed the whole deploy. With it, the transfer aborts and the retry path works.

## 4. Deploying many rigs at once: copy the model from a rig you already have

Public model hosts rate-limit aggressive parallel range requests. Fetching with
8 parallel ranges from several fresh boxes simultaneously throttled to
**0–2 MB/s**, and consistently failed on the same chunk.

Copying between two rigs at the same provider ran **40–120 MB/s** — a 16.4 GB
model in under three minutes.

On a rig that already has a verified model:

```bash
python3 -m http.server 8899 --directory /path/to/models
```

On the new rig:

```bash
curl -fsS --speed-limit 1048576 --speed-time 30 \
     -o model.gguf http://<source-rig>:8899/model.gguf
sha256sum model.gguf            # ALWAYS verify before mining
```

**Stop the server afterwards** — and note Python's `http.server` has no Range
support, so it serves whole files only.

## 5. One container per GPU — and the logs cannot prove you did it

Give each container exactly one GPU:

```bash
docker run -d --name meow0 --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /path/to/models:/models:ro ...
```

Every container sees only its own card, so **every one of them reports "GPU 0"**
in its logs. A misconfiguration that puts two containers on the same physical GPU
looks completely normal in the logs. Check from the host:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader
```

You should see one distinct UUID per container. Mount the model read-only
(`:ro`) — the miner never writes to it.

`--gpus` may fail on some hosts with "failed to discover GPU vendor from CDI";
`--runtime=nvidia` with `NVIDIA_VISIBLE_DEVICES` is the portable form.

## 6. The displayed rate is quantized — never judge from one reading

The `TOTAL: x PoI/s` line is *windows completed since the last report ÷ interval*.
Windows complete in whole batches, so the printed value can only ever be an
integer multiple of one batch per interval. A card genuinely doing 21 PoI/s will
alternate between readings like 17.05 and 34.10, and **a single sample can be
~50% off in either direction**.

Average the log history instead:

```bash
docker logs meow0 2>&1 | grep -oE "TOTAL: +[0-9.]+ PoI/s" | grep -oE "[0-9.]+" \
  | awk '{n++; s+=$1} END {printf "n=%d mean=%.2f PoI/s\n", n, s/n}'
```

Thirty samples is plenty. The quantum scales with the tuned slot count, so it
differs per card.

## 7. Confirm the auto-tune actually applied

The miner auto-tunes per card when you do not pass `--slots`/`--groups`:

```bash
docker logs meow0 2>&1 | grep "auto-tuned for"
# auto-tuned for A100 80GB: --slots 512 --groups 12
```

A config line in a log proves a value was *computed*, not that it was *used* —
this exact distinction hid a real bug once, where the tuned profile was printed
but the engine kept its defaults. If throughput is far below
[BENCHMARKS.md](BENCHMARKS.md) for your card, suspect that before the hardware.

## 8. Disk, RAM and rented-box housekeeping

- **Disk:** the model is 16.4 GB and the image ~5.5 GB unpacked. On a 50 GB
  volume that leaves little room — keep **one** copy of the model on the host and
  bind-mount it into every container. Do not bake it into an image there, and do
  not keep two copies (a naive "download parts then concatenate" needs 2× the
  space and will silently truncate).
- **RAM:** allow roughly 4 GB of host RAM per GPU. Under-provisioned boxes swap
  and collapse in a way that looks like a thermal or driver problem.
- **Log rotation:** long-running containers with unbounded JSON logs will fill
  the disk on their own. Set `--log-opt max-size`/`max-file`.
- **Spot instances recycle IP addresses.** If SSH suddenly reports
  `Permission denied (publickey)` rather than timing out, the address has
  probably been reassigned to a different machine — that is not your rig
  refusing you. Clear the stale host key (`ssh-keygen -R <ip>`) before
  concluding anything, and be aware a box can re-provision mid-deploy and change
  its key a second time.

## 9. Blackwell datacenter cards (B200): do not add `sm_100`

The published image is built for `80;86;89;90;120` and **deliberately omits
`sm_100`**. A B200 JIT-compiles the embedded `sm_90` PTX forward, and in a
concurrent A/B on one 8× B200 box that was **faster than native `sm_100` SASS**:

```
JIT from sm_90 PTX   56.2 PoI/s
native sm_100 SASS   49.0 PoI/s      (15% regression)
```

Forward JIT is also the only direction that works — `sm_120` PTX cannot serve
`sm_100`. Expect a slower first container start on these cards while the JIT
runs; that is not a hang. Re-measure before changing this, since a future ggml
may flip the result.

---

## Quick pre-flight checklist

```
[ ] image tag pinned (not :latest), and identical across all containers
[ ] model sha256 verified AFTER download and BEFORE the first container starts
[ ] one distinct GPU UUID per container (nvidia-smi, not the container logs)
[ ] model mounted read-only
[ ] auto-tune line present in each container's log
[ ] rate averaged over ~30 samples, compared against BENCHMARKS.md
[ ] log rotation configured
[ ] no --split-model anywhere (the miner refuses it; see COMPATIBILITY.md)
```
