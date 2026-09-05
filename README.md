# supr-meow-tsc

A GPU miner for TensorCash (TSC), with Linux, HiveOS, MMPOS, SimpleMining and
Docker packages. Supports independent GPUs and shared-model mining across
matching GPU pairs.

See [v0.6.0 release notes](RELEASE-v0.6.0.md) for downloads, measured rates and
upgrade instructions, and [hardware compatibility](docs/COMPATIBILITY.md) for
requirements.

## Quick start

From an extracted Linux package:

```sh
POOL_URL='stratum+tcp://YOUR_POOL:PORT' \
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
  -d 0 -o stratum+tcp://YOUR_POOL:PORT -u YOUR_WALLET.rig1 -p x
```

Add `--split-model -d 0,1` for a shared pair. Use `--help` for device selection,
logging, API and tuning options. Explicit tuning settings override defaults.
See [operation notes](docs/OPERATIONS.md) for sharing a rig with other workloads.

## Build

Use a CUDA toolkit supporting the target GPUs, CMake, a C++ compiler, OpenSSL,
ZeroMQ/cppzmq, Argon2, GMP and matching FlatBuffers compiler/headers. Docker
builds install these dependencies automatically.

Do not clone recursively: this miner uses only selected dependencies, while
nested upstream submodules can be unavailable.

```sh
git clone https://github.com/ocminer/supr-meow-tsc.git
cd supr-meow-tsc
git submodule update --init vendor/tensorcash vendor/llama.cpp
bash tools/prepare-llama.sh vendor/llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target supr-meow-tsc -j
```

Build the container or the portable mining-OS package:

```sh
docker build -f docker/Dockerfile -t supr-meow-tsc:0.6.0 .
bash tools/build-hiveos-package.sh 0.6.0
```

The portable package is built against Ubuntu 22.04 for compatibility with
mining distributions. Do not substitute a binary built against a newer glibc.

The [TSC Stratum specification](docs/STRATUM-TSC.md) documents pool integration.
The older vLLM backend remains in `vllm-miner/` for compatibility; the v0.6.0
packages and image above are the current release path.
