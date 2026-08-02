# Operating supr-meow-tsc on rented GPUs

Everything here is measured. It is the public, identifier-free half of a
working doc — the private half (pool endpoints, payout address, rig
addresses) stays out of this repo deliberately.

## Deploying a box

One command per box, ~12 minutes end to end, most of it the model download:

```bash
wait-and-deploy.sh <ip> <worker-prefix> <ngpus>
```

It waits out provisioning, copies `tools/fetch-model-parallel.sh` and a
bootstrap script, downloads and **sha256-verifies** the model, then starts one
container per GPU on auto-tune and reports the rates.

Leave `SLOTS` and `MEOW_GROUPS` **unset** — the miner reads compute capability
and VRAM and applies a measured profile. **Never set `GROUPS`**: it is a bash
built-in array, so it always arrives as `0` (see below).

## Rented-hardware gotchas, all of which cost real time

**Spot instances churn constantly.** Expect roughly half of what you start to
be reclaimed within hours. Commit findings immediately — the profile that is
now worth +37% on every RTX PRO 6000 came from a box reclaimed the same day.

**An open port 22 does not mean your box survived.** Spot IPs get recycled to
other tenants. A **changed host key** is the real signal: `ssh-keygen -R <ip>`,
then confirm your key still authenticates. One reclaimed box was only noticed
because a newly rented card was handed the dead box's address.

**A 50 GB volume cannot hold the model, the image and a Docker build.** The
model is 16.4 GB and the image ~5.5 GB. Building on the box fills the disk.

**Download the model in parallel.** A single stream measured 21 Mbit/s from one
host and decayed to 3 — twelve hours for 16.4 GB, billed by the hour. Eight
ranges in parallel gave ~300 Mbit/s. `tools/fetch-model-parallel.sh` also
writes each range straight to its offset rather than downloading parts and
concatenating them, because that needs 2x the file size and **silently
truncates** when the disk fills.

**`--gpus all` fails on some hosts** with "failed to discover GPU vendor from
CDI". Use `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<n>`.

**`failed to fulfil mount request: open /run/nvidia-persistenced/socket`** is a
host misconfiguration, not an image problem — it fails at `runc create`, before
any container code runs. Any GPU container on that node fails identically.

## GROUPS is a bash built-in

`GROUPS` holds the caller's group ids, so bash populates it when the
environment does not — `${GROUPS:-12}` expands to `"0"` and never reaches its
fallback. Every containerised run therefore passed `--groups 0`, collapsing to
one worker group sorting all the slots at once, above the GPU sort's 64-stream
limit. Every window failed, silently, and one rig mined nothing for over seven
hours. The variable is now `MEOW_GROUPS`; `--groups 0` means "auto"; and the
miner raises the group count when `slots/groups` would exceed 64.

## Tuning rules that do not generalise

Three separate times an optimum turned out to depend on something other than
the GPU model. Check before assuming:

- **Slot count depends on the memory regime, not the card.** VRAM-bound cards
  gain from slots monotonically to their ceiling; bandwidth-bound cards plateau
  and then regress. And VRAM class alone does not decide it — two 48 GB cards
  behaved oppositely.
- **Group count depends on host CPU count.** An 80-core host measured groups
  flat; a 176-core one gained clearly from 24.
- **The CPU-tail tuning's value tracks GPU idle.** A card at 78% utilisation
  gained +6.2%; one already at 88-100% gained +1.3%. Check
  `nvidia-smi --query-gpu=utilization.gpu` first.

And `MEOW_UBATCH=4096` **costs VRAM** — it OOMs outright on a card sitting near
its slot ceiling. Never enable it there.

Full measured numbers, curves and cost-per-GPU-hour: [BENCHMARKS.md](BENCHMARKS.md).
