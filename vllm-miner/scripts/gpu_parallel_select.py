#!/usr/bin/env python3
"""Auto-select vLLM parallelism by GPU interconnect.

Rule (measured on a 2x RTX 5080 PCIe rig: PP=2 3.63 w/s vs TP=2 1.49 — wrong choice ~2.4x):
  all selected GPUs NVLink-connected  -> tensor-parallel  (cheap all-reduce)
  otherwise (PCIe/PHB/PXB/SYS/...)    -> pipeline-parallel (minimal comms)

Detection: parse `nvidia-smi topo -m`. A pairwise cell containing "NV" (NV1,
NV2, ...) means an NVLink connects that pair; anything else (PHB/PXB/PIX/NODE/
SYS) is PCIe/host. NVLink is chosen only if EVERY pair among the selected GPUs
is NVLink-connected.

Usage:
  gpu_parallel_select.py 2 3            # decide for physical GPUs 2 and 3
  gpu_parallel_select.py 2 3 --flag     # print just the vLLM flag
Prints (default): "<mode> <n>" e.g. "pipeline 2".  --flag: "--pipeline-parallel-size 2"
"""
import subprocess, sys, re


def topo_matrix():
    out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        m = re.match(r"\s*GPU(\d+)\s+(.*)", line)
        if not m:
            continue
        gpu = int(m.group(1))
        # cells before the CPU-affinity columns; keep tokens that look like link types
        cells = m.group(2).split()
        link_cells = [c for c in cells if c == "X" or re.match(r"^(NV\d+|PHB|PXB|PIX|NODE|SYS|S/A)$", c)]
        rows[gpu] = link_cells
    return rows


def all_nvlink(gpus):
    rows = topo_matrix()
    for a in gpus:
        if a not in rows:
            return False, f"GPU{a} not in topo"
        for b in gpus:
            if a == b:
                continue
            if b >= len(rows[a]):
                return False, f"no cell GPU{a}->GPU{b}"
            cell = rows[a][b]
            if not cell.startswith("NV"):
                return False, f"GPU{a}<->GPU{b} = {cell} (not NVLink)"
    return True, "all pairs NVLink"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flag_only = "--flag" in sys.argv
    gpus = [int(x) for x in args] if args else [0, 1]
    n = len(gpus)
    nvlink, reason = all_nvlink(gpus)
    mode = "tensor" if nvlink else "pipeline"
    if flag_only:
        print(f"--{mode}-parallel-size {n}")
    else:
        print(f"{mode} {n}")
        print(f"# gpus={gpus} nvlink={nvlink} ({reason})", file=sys.stderr)


if __name__ == "__main__":
    main()
