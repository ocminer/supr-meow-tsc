# Hardware compatibility

See the repository's main [docs/COMPATIBILITY.md](../../docs/COMPATIBILITY.md)
— it covers both engines (llama.cpp and this vLLM backend) in one place.

Short version for this backend: **two matched sub-24 GB GPUs** (validated on
2x RTX 5080 16 GB over PCIe) mine ~3.9 w/s per pair via pipeline-parallel,
producing full-tier-verified proofs. Single GPUs >= 24 GB should use the
llama.cpp miner instead — it is ~3x faster on that tier.
