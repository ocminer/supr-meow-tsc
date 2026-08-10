# Hardware compatibility

See the repository's main [docs/COMPATIBILITY.md](../../docs/COMPATIBILITY.md)
— it covers both engines (llama.cpp and this vLLM backend) in one place.

Short version for this backend: **two matched sub-24 GB GPUs** (validated on
2x RTX 5080 16 GB over PCIe) mine ~11.4 w/s per pair via pipeline-parallel (high-batch defaults),
producing full-tier-verified proofs. Single GPUs >= 24 GB: the engines are
now comparable (17.5 vs 19.5 w/s on a 5090); llama.cpp remains recommended
there — slightly faster and far lighter to deploy.
