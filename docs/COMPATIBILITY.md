# Hardware compatibility

Use Linux x86-64 and an NVIDIA Ampere or newer GPU with a compatible driver.
Turing and older GPUs, AMD, Intel and Apple GPUs are not supported by this release.

- Single-card mining: at least 12 GB of GPU memory; additional memory can
  improve performance. RTX 5090 32 GB was tested for this release.
- Shared-model mining: matching pairs with at least 8 GB on each card.
  A pair of RTX 3070 8 GB cards was tested for this release.
- Multiple independent cards: each card must hold its own model.
- Mixed rigs: launch different GPU types separately and explicitly select
  their devices. Each shared pair must contain the same GPU model and capacity.

Keep other workloads off the selected cards. The model cache also needs disk
space; allow at least 10 GB for the model download, plus package and temporary
storage. Leave memory settings at their defaults when upgrading.

See the [release notes](../RELEASE-v0.6.0.md) for shared-card commands and
[mining operation](OPERATIONS.md) for device numbering.
