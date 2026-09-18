# supr-meow-tsc

A GPU miner for TensorCash (TSC), with Linux, native Windows, HiveOS, MMPOS, SimpleMining and
Docker packages. Supports independent GPUs and shared-model mining across
matching GPU pairs.

See [v0.7.0 release notes](RELEASE-v0.7.0.md) for downloads and
upgrade instructions, and [hardware compatibility](docs/COMPATIBILITY.md) for
requirements.

Version **0.7.0 emits proof v4**, required for the TensorCash network upgrade.
Upgrade before height **27615**, when v3 proofs stop being accepted.
Use the standard Q8/F16 profile so the pool receives `supr-meow-tsc/0.7.0`.

## Quick start

From an extracted Linux package:

```sh
POOL_URL='stratum+tcp://tsc.suprnova.cc:3310' \
WALLET='YOUR_WALLET.rig1' DEVICES=0 ./meow-common.sh
```

For two matching cards sharing one model, use `DEVICES=0,1 SPLIT_MODEL=1`.
Each card must have at least 8 GB. Select complete adjacent pairs for larger
rigs. The launchers download and verify the required model automatically.
Keep `MODEL_DIR` on persistent storage.

For an existing compatible model and a direct binary launch:

```sh
LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
./supr-meow-tsc --q8 --model /models/Qwen3-8B-Q8_0-full-v2.gguf \
  -d 0 -o stratum+tcp://tsc.suprnova.cc:3310 -u YOUR_WALLET.rig1 -p x
```

Add `--split-model -d 0,1` for a shared pair. Use `--help` for device selection,
logging, API and tuning options. Explicit tuning settings override defaults.
See [operation notes](docs/OPERATIONS.md) for sharing a rig with other workloads.

## Native Windows

Download and extract `supr-meow-tsc-0.7.0-windows-x86_64.zip`, edit `start.cmd`
with your wallet, and run it. This is a native executable built in GitHub CI;
WSL and Docker are not required. See [Windows instructions](windows/README.txt).
The first launch downloads the same verified Q8 model used by Linux.

## Build

Use a CUDA toolkit supporting the target GPUs, CMake, a C++ compiler, OpenSSL,
ZeroMQ/cppzmq, Argon2, GMP and matching FlatBuffers compiler/headers. Docker
builds install these dependencies automatically.

Do not clone recursively: this miner uses only selected dependencies, while
nested upstream submodules can be unavailable.

```sh
git clone https://github.com/ocminer/supr-meow-tsc.git
cd supr-meow-tsc
git submodule update --init vendor/llama.cpp
bash tools/prepare-llama.sh vendor/llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target supr-meow-tsc -j
```

Build the container or the portable mining-OS package:

```sh
docker build -f docker/Dockerfile -t supr-meow-tsc:0.7.0 .
bash tools/build-hiveos-package.sh 0.7.0
```

The portable package is built against Ubuntu 22.04 for compatibility with
mining distributions. Do not substitute a binary built against a newer glibc.

The [TSC Stratum specification](docs/STRATUM-TSC.md) documents pool integration.
The older vLLM backend remains in `vllm-miner/` for compatibility; the v0.7.0
packages and image above are the current release path.

The [Windows CI workflow](.github/workflows/windows.yml) builds with MSVC and CUDA,
checks proof vectors and native sockets, and packages the executable and DLLs.
Hosted Windows CI has no NVIDIA GPU; its checks do not replace a GPU mining test.
