# supr-meow-tsc v0.7.4

A compatibility release: the Linux package now runs on **Ubuntu 20.04 and
newer**, which covers current HiveOS, MMPOS and SMOS images. Mining, proofs and
the pre-submit check are unchanged from v0.7.3.

## What changed

- **Runs on older mining OS images.** v0.7.3 was built on Ubuntu 22.04 and
  needed glibc 2.34, so it did not start on Ubuntu 20.04-based rigs
  (`version 'GLIBC_2.34' not found`). v0.7.4 is built on Ubuntu 20.04
  (glibc 2.31) with CUDA 12.8; every library the miner needs beyond the base
  system and the NVIDIA driver ships in `lib/`.
- **Model download works with older curl.** The custom-URL download path no
  longer passes `--no-progress-meter`, which curl versions before 7.67 reject,
  and the HiveOS ranged download now follows redirects (needed for Hugging
  Face links). The default Q8 model download is unchanged and checksum-verified.

## Scope and limits

- Proof format (v4), mining kernels, the ruleset-5 pre-check, flight sheets,
  wallets and command lines are unchanged from v0.7.3.
- The native Windows/MSVC build still compiles the pre-check out (no 128-bit
  integers in MSVC); Windows miners keep submitting the near-pinned windows the
  network rejects after height 31,400. Use a Linux build for the withholding.
- Requires an NVIDIA driver with CUDA 12.8 support (R570 or newer).

## Upgrading

Replace the files of v0.7.3 with this package. The pool identifies the miner as
`supr-meow-tsc/0.7.4`. Use the standard Q8/F16 profile; the model downloads on
first start if it is not cached.
