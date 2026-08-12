# SPDX-License-Identifier: Apache-2.0
"""Shared Proof-of-Work utilities for both V0 and V1 sampling."""

import torch
import hashlib
from dataclasses import dataclass, field
import math
import os
import time
import uuid
import json
from typing import Optional, Dict, Any
import flatbuffers
import sys
# Import generated FlatBuffers modules. Prefer environment/PYTHONPATH; if missing,
# fall back to repo-relative fb-schemas so local runs don’t depend on absolute paths.
try:
    from proof import Proof, FloatArray, UIntArray
except Exception:
    from pathlib import Path
    here = Path(__file__).resolve()
    for parent in here.parents:
        fb = parent / "fb-schemas"
        if fb.exists():
            sys.path.insert(0, str(fb))
            break
    from proof import Proof, FloatArray, UIntArray
import base64
import numpy as np
from collections import deque
from pprint import pformat
import binascii

# V3 prompt-binding / admission helpers (TIP-0003). pow_v3.py is
# deployed next to this file everywhere pow_utils.py is copied (top-level in
# pow-utils, utils/ package in verification-api), hence the dual import.
try:
    import pow_v3
except ImportError:
    from . import pow_v3


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean env var. Anything unrecognised falls back to default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off", ""):
        return False
    return default


def proof_snapshot(proof_logits, logsumexp_full, device):
    """Pre-temperature proof ring-buffer snapshot from full-vocab logits.

    Returns (extended_logits[B,70], extended_indices[B,70], mean_tensor[B,6]):
    top-50 logits/indices + 20 fixed-stride probes, and the 6 logsumexp stats
    (mean[0]=logsumexp_full, mean[1:5]=top50/50:500/500:2000/2000:+ bucket means,
    mean[5]=mean-all). Full-vocab stable descending sort.
    """
    B, V = proof_logits.shape
    probe_step = V // 20
    probe_indices_list = torch.arange(0, V, probe_step, device=device)[:20]
    probe_logits = proof_logits.gather(
        1, probe_indices_list.unsqueeze(0).expand(B, -1))
    probe_idx_row = probe_indices_list.unsqueeze(0).expand(B, -1).to(torch.int32)

    sl, si = torch.sort(proof_logits, dim=-1, descending=True, stable=True)
    topk_vals = sl[:, :50]
    topk_idx = si[:, :50]
    mean_tensor = torch.zeros((B, 6), dtype=torch.float32, device=device)
    mean_tensor[:, 0] = logsumexp_full
    if V >= 50:
        mean_tensor[:, 1] = torch.mean(sl[:, :50], dim=-1)
    if V >= 500:
        mean_tensor[:, 2] = torch.mean(sl[:, 50:500], dim=-1)
    if V >= 2000:
        mean_tensor[:, 3] = torch.mean(sl[:, 500:2000], dim=-1)
    if V > 2000:
        mean_tensor[:, 4] = torch.mean(sl[:, 2000:], dim=-1)
    mean_tensor[:, 5] = torch.mean(sl, dim=-1)

    extended_logits = torch.cat([topk_vals, probe_logits], dim=1)
    extended_indices = torch.cat([topk_idx, probe_idx_row], dim=1)
    return extended_logits, extended_indices, mean_tensor


def apply_topk_topp_mask(logits, k, p, fast=None, _topk_cap=50):
    """PoW top-k/top-p masking. Strict exclusion: keep logits STRICTLY greater than
    the k-th largest (mask `logits <= kth`), then nucleus for rows with p<1.

    Two k-mask implementations, producing a BYTE-IDENTICAL masked tensor:
      legacy: full-vocab ascending sort + gather(kth) + scatter-back.
      fast  : topk(cap) for the per-row k-th-largest threshold, then
              `logits.masked_fill(logits <= threshold)` — NO full sort, NO scatter.
              PoW invariant top_k <= 50 makes `cap` sufficient (proof support is top-50).

    `fast=None` (default, production): use the fast path for the top_p == 1.0 case
    (p None, or no row with p < 1.0); fall back to the legacy sort when any p < 1.0
    is present (kept until the nucleus path is separately cleaned up). `fast=True/False`
    forces a path (used by the equivalence test).
    """
    if k is None and p is None:
        return logits

    if fast is None:
        use_fast = (p is None) or (not bool((p.reshape(-1) < 1.0).any()))
    else:
        use_fast = fast

    if k is not None:
        if use_fast:
            cap = min(_topk_cap, logits.size(-1))
            topk_vals, _ = logits.topk(cap, dim=-1)                       # (B,cap) desc
            idx = (k.to(torch.long) - 1).clamp_(0, cap - 1).unsqueeze(1)  # k-th largest slot
            threshold = topk_vals.gather(1, idx)                         # (B,1)
            masked = logits.masked_fill(logits <= threshold, -float("inf"))
        else:
            logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
            top_k_count = logits_sort.size(-1) - k.to(torch.long)
            top_k_mask = logits_sort.gather(1, top_k_count.unsqueeze(dim=1))
            logits_sort.masked_fill_(logits_sort <= top_k_mask, -float("inf"))
            masked = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    else:
        masked = logits.clone()

    no_survivor = torch.isneginf(masked).all(dim=-1)              # (B,) bool
    masked.masked_fill_(no_survivor.unsqueeze(-1), float("-inf"))
    argmax = logits.argmax(dim=-1)                                # (B,) first max
    ar = torch.arange(masked.size(0), device=masked.device)
    masked[ar, argmax] = torch.where(no_survivor,
                                     logits[ar, argmax],
                                     masked[ar, argmax])

    if p is not None:
        p_flat = p.to(device=logits.device, dtype=torch.float32).reshape(-1)
        if p_flat.numel() == 1:
            p_flat = p_flat.expand(logits.shape[0])

        top_p_rows = torch.nonzero(p_flat < 1.0, as_tuple=False).flatten()
        for row_tensor in top_p_rows:
            row = int(row_tensor.item())
            support = torch.isfinite(masked[row]).nonzero(as_tuple=False).flatten()
            if support.numel() <= 1:
                continue
            if support.numel() > 50:
                raise ValueError(
                    "PoW top_p < 1.0 requires top_k <= 50 so the "
                    "verifier can replay from the fixed proof support")

            support_logits = masked[row, support]
            order = torch.argsort(-support_logits, stable=True)
            sorted_support = support[order]
            sorted_logits = support_logits[order]
            probs_sort = sorted_logits.softmax(dim=-1)
            prev_cum = torch.cumsum(probs_sort, dim=-1) - probs_sort
            keep = prev_cum < p_flat[row]
            keep[0] = True
            masked[row, sorted_support[~keep]] = -float("inf")

    return masked


try:
    import atexit
    import functools

    # Global registry for all profiled methods
    _pow_profile_data: dict[str, dict[str, float]] = {}

    def pow_profiler(fn):
        """
        Decorator to profile how many times fn is called and 
        its total/min/max elapsed time.
        
        Attach it to any method with @profiler.
        """
        qual = fn.__qualname__
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            
            stats = _pow_profile_data.setdefault(qual, {
                "count": 0,
                "total": 0.0,
                "min": float("inf"),
                "max": 0.0,
            })
            stats["count"] += 1
            stats["total"] += dt
            stats["min"] = min(stats["min"], dt)
            stats["max"] = max(stats["max"], dt)
            
            return result
        return wrapper

    def _pow_print_profile_summary():
        if not _pow_profile_data:
            print("No profiling data collected.")
            return
        print("\n=== Profiling Summary ===")
        for qual, s in _pow_profile_data.items():
            avg = s["total"] / s["count"]
            print(
                f"{qual:50s} "
                f"calls={s['count']:5d}  "
                f"avg={avg*1e3:6.2f} ms  "
                f"min={s['min']*1e3:6.2f} ms  "
                f"max={s['max']*1e3:6.2f} ms"
            )
        print("=========================\n")

    # print automatically on normal exit
    atexit.register(_pow_print_profile_summary)
except:
    def profiler(fn):
        return fn    

POW_WINDOW_SIZE = 256          # length (tokens) of the sliding window

@dataclass
class SequenceCache:
    """All state that lives as long as a seq_id is alive."""
    # grow-only archive, CPU side
    archive: list[int] = field(default_factory=list)      # full prompt + gens
    # fast rolling window, GPU side
    ring: torch.Tensor | None = None       # (POW_WINDOW_SIZE,) int32 CUDA
    ring_pos: int = 0                      # write cursor 0…POW_WINDOW_SIZE-1

@dataclass
class PowState:
    # immutable --------------------------------------------------------
    target: torch.ByteTensor            # (32,) uint8 - 256-bit target
    h_b:   torch.ByteTensor            # (32,) uint8 - block hash (legacy)
    v:     torch.ByteTensor            # (32,) uint8 - VDF
    T:     int                         # python int - tick
    header_prefix: torch.ByteTensor = None  # (76,) uint8 - block header prefix (optional)

    # rolling 256-step ring buffers  ----------------------------------
    topk_logits:   torch.FloatTensor = None   # (256, B, 50) - fp32 for reproducibility
    topk_indices:  torch.IntTensor  = None   # (256, B, 50)
    chosen_probs:  torch.FloatTensor = None   # (256, B)
    chosen_tokens: torch.LongTensor  = None   # (256, B)
    attention_mask:torch.BoolTensor = None   # (256, B)
    steps:         torch.IntTensor  = None   # (B,)
    window_pos:    int              = 0      # python int

    # For probability reconstruction -----------------------
    sampling_u:    torch.FloatTensor = None   # (256, B) - random values used for sampling

    # These are scalar per sequence, not per token - storing once when updated
    temperature_by_seq: dict[int, float] = field(default_factory=dict)
    top_p_by_seq: dict[int, float] = field(default_factory=dict)
    top_k_by_seq: dict[int, int] = field(default_factory=dict)
    rep_penalty_by_seq: dict[int, float] = field(default_factory=dict)
    softmax_normalizers: torch.FloatTensor = None # (256, B) - softmax denominators (fp32)
    seq_cache: dict[int, SequenceCache] = field(default_factory=dict)

# Helper for logging
class Logger:
    def __init__(self, log_dir=None):
        if not log_dir:
            log_dir = os.environ.get("POW_LOG_DIR", "/data/miner_logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, "pow_sampler.log")

    # @pow_profiler
    def log(self, message, level="INFO"):
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file_path, "a") as f:
                f.write(f"[{timestamp}] [{level}] {message}\n")
        except Exception:
            pass  # Silent failure

# Row manager for efficient assignment of buffer rows to sequences
class RowManager:
    def __init__(self, max_rows):
        self.max_rows = max_rows
        self.seqid_to_row = {}
        # deque gives O(1) pops on both ends
        self.free_rows = deque(range(max_rows))
        # Track allocation order for FIFO tie-breaking
        self.allocation_order = {}
        self.next_allocation_id = 0

    def get_row(self, seq_id):
        """Return the row currently assigned to seq_id, or None."""
        return self.seqid_to_row.get(seq_id)

    def allocate_row(self, seq_id):
        """
        Allocate a row for seq_id.
        - If already allocated, returns existing row.
        - If no free rows remain, returns None.
        - Otherwise, pops from the left (FIFO) and assigns it.
        """
        if seq_id in self.seqid_to_row:
            return self.seqid_to_row[seq_id]

        if not self.free_rows:
            return None  # out of slots

        row = self.free_rows.popleft()  # FIFO: oldest freed row
        self.seqid_to_row[seq_id] = row
        # Track allocation order for FIFO eviction
        self.allocation_order[seq_id] = self.next_allocation_id
        self.next_allocation_id += 1
        return row

    def free_row(self, seq_id):
        """
        Release the row for seq_id (if any), returning it to the free pool.
        Returns the freed row index, or None if seq_id was not allocated.
        """
        row = self.seqid_to_row.pop(seq_id, None)
        if row is not None:
            self.free_rows.append(row)  # goes to the right, newest free
            # Clean up allocation order tracking
            self.allocation_order.pop(seq_id, None)
        return row
    
    @pow_profiler
    def get_oldest_sequence(self, steps):
        if not self.seqid_to_row:
            return None, None
        
        # Get all rows and their steps as tensors
        seq_ids = list(self.seqid_to_row.keys())
        rows = [self.seqid_to_row[sid] for sid in seq_ids]
        
        # Get steps for all rows at once
        rows_tensor = torch.tensor(rows, device=steps.device)
        steps_values = steps[rows_tensor]
        
        # Find max using torch operations
        max_steps = steps_values.max()
        
        # Find all sequences with max steps
        max_mask = steps_values == max_steps
        
        # Among ties, find earliest allocation
        tied_indices = torch.where(max_mask)[0]
        
        # Get allocation orders for tied sequences
        tied_allocs = torch.tensor([
            self.allocation_order[seq_ids[i]] for i in tied_indices.cpu().numpy()
        ])
        
        # Find minimum allocation (earliest)
        earliest_idx = tied_indices[tied_allocs.argmin()]
        
        return seq_ids[earliest_idx], rows[earliest_idx]

# GPU-indexed pow params per row - eliminates CPU-side grouping and repeated hex decoding
class RingBuffers:
    def __init__(self, window_size, max_rows, device="cuda"):
        self.window_size = window_size
        self.max_rows = max_rows
        self.device = device

        # --- packed storage (same idea you implemented) ---
        # floats: [topk_logits:70 | chosen_prob:1 | sampling_u:1 | logZ:1 | lse_stats:6] = 79
        self._float_block = torch.zeros(
            (window_size, max_rows, 79), dtype=torch.float32, device=device
        )
        # ints: [topk_indices:70]
        self._int_block = torch.zeros(
            (window_size, max_rows, 70), dtype=torch.int32, device=device
        )
        # keep mask as bool to avoid surprises elsewhere
        self.attention_mask = torch.zeros(
            (window_size, max_rows), dtype=torch.bool, device=device
        )
        # IMPORTANT: keep chosen tokens as int64 to preserve byte layout for _tok_le_bytes
        self.chosen_tokens = torch.zeros(
            (window_size, max_rows), dtype=torch.int64, device=device
        )

        # steps stays separate (per-row)
        self.steps = torch.zeros(max_rows, dtype=torch.int32, device=device)
        # Optional host mirror of `steps`, so check_pow_solutions can find
        # proof-window boundaries without a GPU->CPU sync on every decode
        # step. OFF by default and gated on POW_STEPS_HOST_FASTPATH=1: it is
        # a pure-perf change, and a mirror that silently drifts out of sync
        # with the device tensor would SKIP proof windows (lost shares), a
        # far worse failure than the sync it removes.
        #
        # Every writer of self.steps must keep the mirror in step or clear
        # steps_host_ok. Current writers: increment_steps() (V1 sampler),
        # clear_row(), clear_rows(), CommonSamplerHelper's full reset, and
        # the V0 sampler's direct `ring_buffers.steps[rows] += 1` — which
        # routes through increment_steps as of this change.
        self.steps_host_fastpath = _env_flag("POW_STEPS_HOST_FASTPATH", False)
        self.steps_host = [0] * max_rows
        # Only ever True while the fast path is enabled AND no writer has
        # invalidated the mirror. Consumers must treat False as "use the
        # device tensor".
        self.steps_host_ok = self.steps_host_fastpath

        # --- POW PARAMS per row (written once when sequence is allocated) ---
        # These are indexed by row, not by window position
        # Eliminates CPU-side grouping and repeated hex→bytes conversion
        self.pow_tick = torch.zeros(max_rows, dtype=torch.int64, device=device)
        self.pow_request_id = torch.zeros(max_rows, dtype=torch.int64, device=device)
        self.pow_header = torch.zeros((max_rows, 76), dtype=torch.uint8, device=device)  # 76-byte header prefix
        self.pow_vdf = torch.zeros((max_rows, 1024), dtype=torch.uint8, device=device)   # VDF can be up to 1024 bytes
        self.pow_vdf_len = torch.zeros(max_rows, dtype=torch.int32, device=device)       # actual VDF length
        self.pow_target = torch.zeros((max_rows, 32), dtype=torch.uint8, device=device)  # 32-byte target
        # Slice 11.4 — share target (model-adjusted, derived by the
        # broker / worker from base_share_target * N / D). Numerically
        # LARGER (easier) than ``pow_target``; mining-tier proofs that
        # meet share but not block are sub-block share emissions.
        # All-zero == "share emission disabled for this row"; the
        # sampler falls back to the legacy block-only check then.
        self.pow_share_target = torch.zeros((max_rows, 32), dtype=torch.uint8, device=device)
        self.pow_block_hash = torch.zeros((max_rows, 32), dtype=torch.uint8, device=device)  # 32-byte block hash
        self.pow_header_len = torch.zeros(max_rows, dtype=torch.int32, device=device)    # actual header length (32 or 76)
        self.pow_valid = torch.zeros(max_rows, dtype=torch.bool, device=device)          # whether pow params are set
        # Host mirrors for per-row message-width metadata. These are written at
        # row allocation alongside the GPU tensors and let the hot sampling path
        # avoid header_len.unique()/item() GPU->CPU syncs every decode step.
        self.pow_header_len_host = [0] * max_rows
        self.pow_vdf_len_host = [0] * max_rows

        # --- V3 admission state per row (TIP-0003) ---
        # The sampler grinds the Argon2id admission nonce at prefill / each
        # 256-step window boundary (native admission_grind, GIL released),
        # stores the selected 32 bytes here BEFORE the window's first sampled
        # token, and batch_sample_tokens appends them to every v3 step
        # preimage while pow_admission_valid[row] is set. Rows mined for the
        # free tier keep the flag False and the legacy v2 message shape —
        # the miner commits to with/without before decoding (§7). Host
        # mirror keeps the hot path free of GPU->CPU syncs.
        self.pow_admission_nonce = torch.zeros((max_rows, 32), dtype=torch.uint8, device=device)
        self.pow_admission_valid = torch.zeros(max_rows, dtype=torch.bool, device=device)
        self.pow_admission_valid_host = [False] * max_rows

        # --- legacy-compatible views on top of packed blocks ---
        # float views
        self.topk_logits         = self._float_block[..., :70]                # (W, R, 70) fp32
        self.chosen_probs        = self._float_block[..., 70]                 # (W, R)     fp32
        self.sampling_u          = self._float_block[..., 71]                 # (W, R)     fp32
        self.softmax_normalizers = self._float_block[..., 72]                 # (W, R)     fp32
        self.logsumexp_stats     = self._float_block[..., 73:79]              # (W, R, 6)  fp32

        # int views
        self.topk_indices        = self._int_block[..., :70]                  # (W, R, 70) int32

    def get_positions(self, rows):
        return self.steps[rows] % self.window_size

    def clear_row(self, row):
        if row is None:
            return
        self.topk_logits[:, row].zero_()
        self.topk_indices[:, row].zero_()
        self.chosen_probs[:, row].zero_()
        self.chosen_tokens[:, row].zero_()     # int64 legacy buffer
        self.attention_mask[:, row].zero_()
        self.sampling_u[:, row].zero_()
        self.softmax_normalizers[:, row].zero_()
        self.logsumexp_stats[:, row].zero_()
        self.steps[row] = 0
        # Host mirror of steps clears with the tensor (see increment_steps).
        self.steps_host[int(row)] = 0
        # Clear pow params for this row
        self.pow_tick[row] = 0
        self.pow_request_id[row] = 0
        self.pow_header[row].zero_()
        self.pow_vdf[row].zero_()
        self.pow_vdf_len[row] = 0
        self.pow_target[row].zero_()
        self.pow_share_target[row].zero_()
        self.pow_block_hash[row].zero_()
        self.pow_header_len[row] = 0
        self.pow_header_len_host[row] = 0
        self.pow_vdf_len_host[row] = 0
        self.pow_valid[row] = False
        self.pow_admission_nonce[row].zero_()
        self.pow_admission_valid[row] = False
        self.pow_admission_valid_host[row] = False

    def write_admission_nonce(self, row: int, nonce: "bytes | None"):
        """Store (or clear, with None) the row's v3 admission nonce.

        Must be called BEFORE the first sampled token of the window the nonce
        admits (TIP-0003: admission is pre-decode — the nonce
        enters every step u of the window).
        """
        if nonce is None:
            self.pow_admission_nonce[row].zero_()
            self.pow_admission_valid[row] = False
            self.pow_admission_valid_host[row] = False
            return
        nonce = bytes(nonce)
        if len(nonce) != 32:
            raise ValueError("admission nonce must be exactly 32 bytes")
        self.pow_admission_nonce[row] = torch.frombuffer(
            bytearray(nonce), dtype=torch.uint8).to(self.device)
        self.pow_admission_valid[row] = True
        self.pow_admission_valid_host[row] = True

    def write_pow_params(self, row: int, pow_snapshot: dict):
        """Write pow params to GPU arrays for a specific row.

        Called once when sequence is allocated. Params are then read
        via row indexing in batch_sample_tokens - no more CPU-side grouping.

        Args:
            row: The row index to write to
            pow_snapshot: Dict with tick, header_prefix, vdf, block_hash, target, request_id
        """
        if pow_snapshot is None:
            self.pow_valid[row] = False
            self.pow_header_len_host[row] = 0
            self.pow_vdf_len_host[row] = 0
            return

        vdf_hex = pow_snapshot.get("vdf", "")
        self.pow_tick[row] = pow_snapshot["tick"]
        self.pow_request_id[row] = pow_snapshot.get("request_id", 0)
        self.pow_header_len_host[row] = 0
        self.pow_vdf_len_host[row] = 0

        # Decode and write header
        header_hex = pow_snapshot.get("header_prefix", "")
        if header_hex:
            header_bytes = hex_to_bytes_tensor(header_hex, device=self.device)
            hlen = min(header_bytes.numel(), 76)
            self.pow_header[row, :hlen] = header_bytes[:hlen]
            self.pow_header_len[row] = hlen
            self.pow_header_len_host[row] = int(hlen)
        else:
            self.pow_header_len[row] = 0

        # Decode and write VDF (can be up to 1024 bytes - commonly 200/341 bytes)
        # vdf_hex already fetched above for debug print
        if vdf_hex:
            vdf_bytes = hex_to_bytes_tensor(vdf_hex, device=self.device)
            vlen = min(vdf_bytes.numel(), 1024)  # VDF can be up to 1024 bytes
            self.pow_vdf[row, :vlen] = vdf_bytes[:vlen]
            self.pow_vdf_len[row] = vlen
            self.pow_vdf_len_host[row] = int(vlen)
        else:
            self.pow_vdf_len[row] = 0

        # Decode and write target (32 bytes)
        target_hex = pow_snapshot.get("target", "")
        if target_hex:
            # Pad target to 32 bytes if needed
            if len(target_hex) < 64:
                target_hex = "0" * (64 - len(target_hex)) + target_hex
            elif len(target_hex) > 64:
                target_hex = target_hex[-64:]
            target_bytes = hex_to_bytes_tensor(target_hex, device=self.device)
            tlen = min(target_bytes.numel(), 32)
            self.pow_target[row, :tlen] = target_bytes[:tlen]

        # Slice 11.4 — share target (model-adjusted). Optional; rows
        # without a share_target keep ``pow_share_target`` zeroed and
        # the sampler falls back to block-only emission.
        share_target_hex = pow_snapshot.get("share_target", "") or ""
        if share_target_hex:
            if len(share_target_hex) < 64:
                share_target_hex = "0" * (64 - len(share_target_hex)) + share_target_hex
            elif len(share_target_hex) > 64:
                share_target_hex = share_target_hex[-64:]
            share_bytes = hex_to_bytes_tensor(share_target_hex, device=self.device)
            slen = min(share_bytes.numel(), 32)
            self.pow_share_target[row, :slen] = share_bytes[:slen]

        # Decode and write block_hash (32 bytes)
        block_hash_hex = pow_snapshot.get("block_hash", "")
        if block_hash_hex:
            bh_bytes = hex_to_bytes_tensor(block_hash_hex, device=self.device)
            bhlen = min(bh_bytes.numel(), 32)
            self.pow_block_hash[row, :bhlen] = bh_bytes[:bhlen]

        # If no header_prefix, use block_hash as header
        if self.pow_header_len_host[row] == 0 and block_hash_hex:
            self.pow_header[row, :bhlen] = self.pow_block_hash[row, :bhlen]
            self.pow_header_len[row] = bhlen
            self.pow_header_len_host[row] = int(bhlen)

        self.pow_valid[row] = True

    def clear_rows(self, rows):
        if not rows:
            return
        r = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        # NOTE: advanced indexing (tensor index) returns a COPY, so
        # `buf[r].zero_()` / `buf[:, r].zero_()` silently zeroes the copy and
        # leaves the buffer stale. Assignments dispatch to index_put_ and DO
        # write through — use only those here.
        self.topk_logits[:, r] = 0
        self.topk_indices[:, r] = 0
        self.chosen_probs[:, r] = 0
        self.chosen_tokens[:, r] = 0
        self.attention_mask[:, r] = False
        self.sampling_u[:, r] = 0
        self.softmax_normalizers[:, r] = 0
        self.logsumexp_stats[:, r] = 0
        self.steps[r] = 0
        # Clear pow params for these rows
        self.pow_tick[r] = 0
        self.pow_request_id[r] = 0
        self.pow_header[r] = 0
        self.pow_vdf[r] = 0
        self.pow_vdf_len[r] = 0
        self.pow_target[r] = 0
        # Share target must clear with the row too: a recycled row keeping a
        # stale share_target would emit shares against a target from its
        # PRIOR occupant (matches clear_row).
        self.pow_share_target[r] = 0
        self.pow_block_hash[r] = 0
        self.pow_header_len[r] = 0
        # A cleared row must not read back as valid (matches clear_row).
        self.pow_valid[r] = False
        # V3 admission state must clear with the row (a stale nonce leaking
        # into a reallocated row would corrupt every u of its first window).
        self.pow_admission_nonce[r] = 0
        self.pow_admission_valid[r] = False
        if isinstance(rows, torch.Tensor):
            rows_iter = rows.detach().cpu().tolist()
        else:
            rows_iter = rows
        for row in rows_iter:
            self.pow_header_len_host[int(row)] = 0
            # Host mirror of the admission flag clears with the tensor.
            self.pow_admission_valid_host[int(row)] = False
            self.pow_vdf_len_host[int(row)] = 0
            self.pow_admission_valid_host[int(row)] = False
            # Host mirror of steps clears with the tensor (see
            # increment_steps); a recycled row keeping its predecessor's
            # step count would fire proof windows at the wrong offsets.
            self.steps_host[int(row)] = 0
        self.pow_valid[r] = False

    def _write_buffers(self, pos, rows, logits, indices, tokens, probs, u, mask, normalizers, stats):
        # pos, rows are 1D tensors on device (your caller already does this)
        # tokens must be int64 (caller already ensures this); others are fp32/i32/bool
        self.chosen_tokens[pos, rows]         = tokens                      # int64
        self.chosen_probs[pos, rows]          = probs                       # fp32
        self.topk_logits[pos, rows]           = logits                      # fp32[...70]
        self.sampling_u[pos, rows]            = u                           # fp32
        self.topk_indices[pos, rows]          = indices.to(torch.int32)     # int32[...70]
        self.softmax_normalizers[pos, rows]   = normalizers                 # fp32
        self.logsumexp_stats[pos, rows]       = stats                       # fp32[...6]
        self.attention_mask[pos, rows]        = mask                        # bool

    def increment_steps(self, rows, rows_host=None):
        """Advance the per-row step counter.

        ``rows_host`` is the same row set already materialised on the host by
        the caller. Supplying it keeps ``steps_host`` in sync so
        ``check_pow_solutions`` can skip its per-step device sync. A caller
        that cannot supply it permanently invalidates the mirror rather than
        letting it drift — a stale mirror would miss proof-window boundaries
        and silently drop shares.
        """
        if len(rows) == 0:
            return
        r = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        self.steps[r] += 1
        if not self.steps_host_fastpath:
            return
        if rows_host is None:
            if self.steps_host_ok:
                self.steps_host_ok = False
            return
        steps_host = self.steps_host
        for row in rows_host:
            steps_host[int(row)] += 1

    def get_window(self, row):
        pos = self.steps[row] % self.window_size
        idx = (torch.arange(self.window_size, device=self.device) + pos) % self.window_size
        return {
            "tokens":              self.chosen_tokens[idx, row],          # int64 (unchanged)
            "probs":               self.chosen_probs[idx, row],
            "topk_logits":         self.topk_logits[idx, row],
            "topk_indices":        self.topk_indices[idx, row],
            "attention_mask":      self.attention_mask[idx, row],
            "sampling_u":          self.sampling_u[idx, row],
            "softmax_normalizers": self.softmax_normalizers[idx, row],
            "logsumexp_stats":     self.logsumexp_stats[idx, row],
        }
            
# Helper for PoW-specific operations
class PowHasher:
    def __init__(self, device="cuda"):
        self.device = device

        # Initialize parameters
        self.h_b = torch.zeros(32, dtype=torch.uint8, device=device)
        self.v = torch.zeros(32, dtype=torch.uint8, device=device)
        self.target = torch.zeros(32, dtype=torch.uint8, device=device)
        self.target[-1] = 0xFF  # Default easy target
        self.share_target = None  # Optional model-adjusted share target
        self.tick = 0
        self.ipfs_cid = None 
        self.header_prefix = None  # Will store the 76-byte header prefix
        self.request_id = None
        self.difficulty = None
        self.prior_hb = ""
        self.prior_vdf = ""
        self.prior_target = ""
        self.prior_share_target = ""
        self.prior_header = ""
        self.prior_precision = ""

    # @pow_profiler
    def update_from_payload(self, payload):
        """Update parameters from a PoW payload."""
        if self.prior_hb != payload["block_hash"]:
            self.h_b = hex_to_bytes_tensor(payload["block_hash"], device=self.device)
            self.prior_hb = payload["block_hash"]

        if self.prior_vdf != payload["vdf"]:
            self.v = hex_to_bytes_tensor(payload["vdf"], device=self.device)
            self.prior_vdf = payload["vdf"]

        self.tick = int(payload["tick"])
        self.request_id = int(payload["request_id"])
        target_hex = payload["target"]
        
        if self.prior_target != target_hex:
            # Ensure target is padded to 32 bytes
            if len(target_hex) < 64:
                target_hex = "0" * (64 - len(target_hex)) + target_hex
            elif len(target_hex) > 64:
                target_hex = target_hex[-64:]
            self.target = hex_to_bytes_tensor(target_hex, device=self.device)
            self.prior_target = target_hex

        # Optional singleton share target. This keeps older vLLM forks from
        # needing a sampler dict-literal edit: update_from_payload already
        # receives the full pow payload, and check_share_solution falls back
        # to this value when a per-sequence pow_snapshot lacks share_target.
        share_target_hex = payload.get("share_target") or ""
        if share_target_hex:
            if len(share_target_hex) < 64:
                share_target_hex = "0" * (64 - len(share_target_hex)) + share_target_hex
            elif len(share_target_hex) > 64:
                share_target_hex = share_target_hex[-64:]
        if share_target_hex != self.prior_share_target:
            if not share_target_hex:
                self.share_target = None
            else:
                self.share_target = hex_to_bytes_tensor(share_target_hex, device=self.device)
            self.prior_share_target = share_target_hex

        self.difficulty = payload["difficulty"]

        # Decode header prefix if present
        if self.prior_header != payload["header_prefix"]:
            self.header_prefix = hex_to_bytes_tensor(payload["header_prefix"], device=self.device)
            self.prior_header = payload["header_prefix"]

        # Decode header prefix if present
        if "ipfs_cid" in payload:
            self.ipfs_cid = payload["ipfs_cid"]

    # @pow_profiler
    def batch_sample_tokens(self, contexts, steps, cdfs, compute_precision, ring_buffers=None, rows_tensor=None, rows_host=None, pow_snapshot=None):
        """Sample tokens for multiple sequences in a batch.

        Args:
            contexts: Token context windows (B, W) int64
            steps: Step indices for each sequence (B,) int32
            cdfs: Cumulative distribution functions for sampling (B, V)
            compute_precision: Precision string (e.g., 'fp16')
            ring_buffers: RingBuffers instance with per-row pow params (preferred)
            rows_tensor: (B,) tensor of row indices to read pow params from ring_buffers
            rows_host: Optional host list/tuple of the same row indices. When
                provided, per-row message widths are read from RingBuffers'
                host mirrors instead of unique()/item() on CUDA tensors.
            pow_snapshot: DEPRECATED - Optional dict for backwards compatibility

        When ring_buffers and rows_tensor are provided, pow params are read directly
        from GPU arrays indexed by row - no CPU-side grouping or hex decoding needed.
        This is the preferred path for mixed-request batches under priority scheduling.
        """
        B = contexts.size(0)
        ctx_bytes = _tok_le_bytes(contexts)
        j4 = _u32le(steps.view(-1, 1))

        # NEW PATH: Read pow params from GPU arrays indexed by row
        if ring_buffers is not None and rows_tensor is not None:
            # Gather ticks for all rows - (B,)
            ticks = ring_buffers.pow_tick[rows_tensor]  # (B,) int64

            # Gather header data - need to handle variable header lengths
            header_lens = ring_buffers.pow_header_len[rows_tensor]  # (B,) int32
            headers = ring_buffers.pow_header[rows_tensor]  # (B, 76) uint8
            vdfs = ring_buffers.pow_vdf[rows_tensor]  # (B, 1024) uint8
            vdf_lens = ring_buffers.pow_vdf_len[rows_tensor]  # (B,) int32

            # Convert ticks to bytes - each row gets its own T8
            # T8 needs to be (B, 4) for per-row tick values
            T8_batch = _u32le(ticks.to(torch.uint32))  # (B, 4)

            host_lengths = None
            host_admission = None
            if rows_host is not None:
                rows_host = [int(r) for r in rows_host]
                if len(rows_host) != B:
                    raise ValueError(
                        f"rows_host length {len(rows_host)} does not match batch {B}")
                host_lengths = [
                    (ring_buffers.pow_header_len_host[r],
                     ring_buffers.pow_vdf_len_host[r])
                    for r in rows_host
                ]
                # V3 admission flags (TIP-0003): a row with a
                # stored nonce gets 32 extra preimage bytes, so admission
                # participates in the same-message-width gating below.
                host_admission = [
                    ring_buffers.pow_admission_valid_host[r] for r in rows_host
                ]

            if compute_precision != self.prior_precision:
                self.precision_bytes = _str_bytes(compute_precision, batch_size=1, device=self.device)
                self.prior_precision = compute_precision

            if host_lengths is not None:
                same_lengths = (
                    all(lengths == host_lengths[0] for lengths in host_lengths)
                    and all(a == host_admission[0] for a in host_admission)
                )
            else:
                # Fallback for old callers: this path has the original
                # per-step GPU->CPU sync and should not be used by V1.
                unique_hlens = header_lens.unique()
                unique_vlens = vdf_lens.unique()
                unique_adm = ring_buffers.pow_admission_valid[rows_tensor].unique()
                same_lengths = (len(unique_hlens) == 1 and len(unique_vlens) == 1
                                and len(unique_adm) == 1)

            if same_lengths:
                # All same lengths - fast batched path
                if host_lengths is not None:
                    hlen, vlen = host_lengths[0]
                    with_admission = host_admission[0]
                else:
                    hlen = unique_hlens[0].item()
                    vlen = unique_vlens[0].item()
                    with_admission = bool(unique_adm[0].item())
                header_data = headers[:, :hlen]  # (B, hlen)
                vdf_data = vdfs[:, :vlen]  # (B, vlen)
                pb = self.precision_bytes.expand(B, -1)

                parts = [
                    header_data,           # (B, hlen)
                    vdf_data,              # (B, vlen)
                    T8_batch,              # (B, 4)
                    j4,                    # (B, 4)
                    ctx_bytes,             # (B, L*8)
                    pb,                    # (B, precision_len)
                ]
                if with_admission:
                    # V3: nonce appended after precision on every step (§7)
                    parts.append(ring_buffers.pow_admission_nonce[rows_tensor])
                msg = torch.cat(parts, dim=1)

                digests = sha256_many(msg)
                us = _digest_to_u(digests)
                token_ids = torch.searchsorted(cdfs, us.unsqueeze(-1)).squeeze(-1)
                # searchsorted returns V (== cdfs.shape[-1]) — one past the
                # last valid token id — whenever u lands above the final CDF
                # entry. That is reachable: _digest_to_u does its arithmetic
                # in float32, so digests near 0xFFFFFFFF round to exactly 1.0,
                # and the CDF's own float32 tail can sum to slightly under
                # 1.0. The result feeds straight into a gather in
                # topk_topp_sampler, so an unclamped V is an out-of-range
                # index. See _clamp_token_ids.
                _clamp_token_ids(token_ids, cdfs.shape[-1])
                token_ids = token_ids.to(torch.int64)
                return token_ids, us, digests
            else:
                # Mixed header/VDF lengths - process per-row
                all_tokens = []
                all_us = []
                all_digests = []

                for i in range(B):
                    if host_lengths is not None:
                        hlen, vlen = host_lengths[i]
                        adm_i = host_admission[i]
                    else:
                        hlen = header_lens[i].item()
                        vlen = vdf_lens[i].item()
                        adm_i = bool(ring_buffers.pow_admission_valid[rows_tensor[i]].item())
                    header_i = headers[i, :hlen].unsqueeze(0)  # (1, hlen)
                    vdf_i = vdfs[i, :vlen].unsqueeze(0)  # (1, vlen)
                    T8_i = T8_batch[i].unsqueeze(0)  # (1, 4)
                    j4_i = j4[i].unsqueeze(0)  # (1, 4)
                    ctx_i = ctx_bytes[i].unsqueeze(0)  # (1, L*8)
                    pb_i = self.precision_bytes  # (1, len)

                    parts_i = [header_i, vdf_i, T8_i, j4_i, ctx_i, pb_i]
                    if adm_i:
                        parts_i.append(
                            ring_buffers.pow_admission_nonce[rows_tensor[i]].unsqueeze(0))
                    msg_i = torch.cat(parts_i, dim=1)
                    digest_i = sha256_many(msg_i)
                    u_i = _digest_to_u(digest_i)
                    tok_i = torch.searchsorted(cdfs[i:i+1], u_i.unsqueeze(-1)).squeeze(-1)
                    _clamp_token_ids(tok_i, cdfs.shape[-1])

                    all_tokens.append(tok_i)
                    all_us.append(u_i)
                    all_digests.append(digest_i)

                token_ids = torch.cat(all_tokens, dim=0).to(torch.int64)
                us = torch.cat(all_us, dim=0)
                digests = torch.cat(all_digests, dim=0)
                return token_ids, us, digests

        # LEGACY PATH: Use pow_snapshot dict or shared state
        if pow_snapshot is not None:
            tick = pow_snapshot["tick"]

            # Prefer precomputed tensors when present to avoid repeated hex decoding
            header_data = pow_snapshot.get("header_tensor")
            vdf_data = pow_snapshot.get("vdf_tensor")
            block_hash_tensor = pow_snapshot.get("block_hash_tensor")

            if header_data is None:
                header_hex = pow_snapshot.get("header_prefix")
                if header_hex:
                    header_data = hex_to_bytes_tensor(header_hex, device=self.device)

            if vdf_data is None:
                vdf_hex = pow_snapshot["vdf"]
                vdf_data = hex_to_bytes_tensor(vdf_hex, device=self.device)

            if block_hash_tensor is None:
                block_hash_hex = pow_snapshot["block_hash"]
                block_hash_tensor = hex_to_bytes_tensor(block_hash_hex, device=self.device)

            # Fall back to block_hash if no header_prefix
            if header_data is None or header_data.numel() == 0:
                header_data = block_hash_tensor
        else:
            tick = self.tick
            header_data = self.header_prefix if self.header_prefix is not None else self.h_b
            vdf_data = self.v

        # Convert tick to bytes (single tick for all rows in legacy path)
        T8 = _u32le(torch.tensor([tick], dtype=torch.uint32, device=self.device))

        # Convert precision to bytes
        if compute_precision != self.prior_precision:
            self.precision_bytes = _str_bytes(compute_precision, batch_size=1, device=self.device)
            self.prior_precision = compute_precision

        pb = self.precision_bytes.expand(ctx_bytes.size(0), -1)

        # Build message
        msg = _build_msg(header_data, vdf_data, T8, j4, ctx_bytes, pb)

        # Compute hash
        digests = sha256_many(msg)

        # Convert to uniform values
        us = _digest_to_u(digests)

        # Sample tokens
        token_ids = torch.searchsorted(cdfs, us.unsqueeze(-1)).squeeze(-1)
        _clamp_token_ids(token_ids, cdfs.shape[-1])
        token_ids = token_ids.to(torch.int64)

        return token_ids, us, digests

    def sample_token(self, context, step, cdf):
        """Sample a token using PoW hash."""
        # Convert context to bytes
        ctx_bytes = _tok_le_bytes(context)

        # Convert step to bytes
        j4 = _u32le(step.view(-1, 1))

        # Convert tick to bytes
        T8 = _u32le(torch.tensor([self.tick], dtype=torch.uint32, device=self.device))

        # Use header_prefix if available, otherwise fall back to h_b
        header_data = self.header_prefix if self.header_prefix is not None else self.h_b

        # Get precision bytes - create if not exists
        if not hasattr(self, 'precision_bytes'):
            compute_precision = getattr(self, 'compute_precision', 'fp16')
            self.precision_bytes = _str_bytes(compute_precision, batch_size=1, device=self.device)
        pb = self.precision_bytes.expand(ctx_bytes.size(0), -1)

        # Build message
        msg = _build_msg(header_data, self.v, T8, j4, ctx_bytes, pb)

        # Compute hash
        digest = sha256_many(msg)

        # Convert to uniform value
        u = _digest_to_u(digest)

        # Sample token
        token_id = torch.searchsorted(cdf, u.unsqueeze(-1)).squeeze(-1)
        _clamp_token_ids(token_id, cdf.shape[-1])

        return token_id, u, digest

    # @pow_profiler
    def check_share_solution(self, digest, pow_snapshot):
        """Slice 11.4 — companion to ``check_solution`` that gates on
        the EASIER share threshold instead of the block target.

        Returns a same-shape bool tensor: True iff the per-row
        canonical chain header hash satisfies
        ``pow_snapshot["share_target"]``. Used by
        ``common_sampler_helper`` to dual-classify each sampled
        digest:
          - meets block target → MineResult (existing path)
          - meets share but not block → MineShare (new sub-block path)
          - meets neither → discard (unless proxy_audit_enabled)

        Returns all-False when ``share_target`` is absent (no
        share-mode for this snapshot) — caller's existing
        block-only behaviour is preserved.
        """
        if pow_snapshot:
            header_hex = pow_snapshot.get("header_prefix")
            block_hash_hex = pow_snapshot.get("block_hash")
            header_data = (
                hex_to_bytes_tensor(header_hex, device=self.device)
                if header_hex else None
            )
            if header_data is None or header_data.numel() == 0:
                header_data = hex_to_bytes_tensor(block_hash_hex, device=self.device)

            share_target_hex = pow_snapshot.get("share_target") or ""
            if share_target_hex:
                # Pad/truncate to canonical 64 hex chars before tensoring.
                if len(share_target_hex) < 64:
                    share_target_hex = "0" * (64 - len(share_target_hex)) + share_target_hex
                elif len(share_target_hex) > 64:
                    share_target_hex = share_target_hex[-64:]
                share_target_data = hex_to_bytes_tensor(share_target_hex, device=self.device)
            else:
                share_target_data = self.share_target
        else:
            header_data = self.header_prefix if self.header_prefix is not None else self.h_b
            share_target_data = self.share_target

        if share_target_data is None:
            return torch.zeros(digest.size(0), dtype=torch.bool, device=self.device)

        B = digest.size(0)
        nonces = digest[:, :4]
        headers = torch.cat([
            header_data.unsqueeze(0).expand(B, -1),
            nonces,
        ], dim=1)
        first_hash = sha256_many(headers)
        header_hashes = sha256_many(first_hash)
        return check_hash_against_target(header_hashes, share_target_data)

    # @pow_profiler
    def check_solution(self, digest, pow_snapshot=None):
        """Check if digest represents a valid PoW solution.

        Args:
            digest: The hash digest to check
            pow_snapshot: Optional dict with frozen pow params (header_prefix, target, block_hash).
                         If provided, uses these instead of shared state to avoid cross-request
                         contamination under priority scheduling.
        """
        B = digest.size(0)

        # Use snapshot params if provided, otherwise fall back to shared state
        if pow_snapshot is not None:
            header_hex = pow_snapshot["header_prefix"]
            target_hex = pow_snapshot["target"]
            block_hash_hex = pow_snapshot["block_hash"]

            header_data = hex_to_bytes_tensor(header_hex, device=self.device) if header_hex else None
            target_data = hex_to_bytes_tensor(target_hex, device=self.device)

            # Fall back to block_hash if no header_prefix
            if header_data is None or header_data.numel() == 0:
                header_data = hex_to_bytes_tensor(block_hash_hex, device=self.device)
        else:
            header_data = self.header_prefix if self.header_prefix is not None else self.h_b
            target_data = self.target

        # Extract nonces (first 4 bytes of each digest)
        nonces = digest[:, :4]

        # Build complete 80-byte headers for all sequences
        headers = torch.cat([
            header_data.unsqueeze(0).expand(B, -1),
            nonces
        ], dim=1)

        # First SHA-256
        first_hash = sha256_many(headers)

        # Second SHA-256
        header_hashes = sha256_many(first_hash)

        # Check if header hashes meet target
        return check_hash_against_target(header_hashes, target_data)

# Helper for writing proofs
class ProofWriter:
    def __init__(self, output_dir="/data/pow_proofs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.submit_callback = None
        # Initialize model_identifier with default value
        self.model_identifier = None
        self.compute_precision = None
        self.model_config_diff = None
        self.sampling_params_diff = None
        self.ipfs_cid = None
        # Proof schema version: 2 = legacy, >= 3 enables the v3 carrier
        # (canonical-JSON extra_flags + admission nonce, TIP-0003).
        self.proof_version = 2

    def set_callback(self, callback):
        """Set callback for when a solution is found."""
        self.submit_callback = callback

    def set_model_identifier(self, model_identifier):
        """Set the model_identifier to be included in proofs."""
        self.model_identifier = model_identifier

    def set_ipfs_cid(self, ipfs_cid):
        """Set the model_ifps_cid to be included in proofs."""
        self.ipfs_cid = ipfs_cid

    def set_model_config_diff(self, model_config_diff):
        """Set the model_config_diff to be included in proofs."""
        self.model_config_diff = model_config_diff

    def set_sampling_params_diff(self, sampling_params_diff):
        """Set the sampling_params_diff to be included in proofs."""
        self.sampling_params_diff = sampling_params_diff

    def set_compute_precision(self, precision: str):
        """Attach compute-precision (fp16 / int8-awq / …) to all proofs."""
        self.compute_precision = precision

    def set_proof_version(self, version: int):
        """Set the proof schema version (2 = legacy, >= 3 = v3 carrier)."""
        self.proof_version = int(version)

    def write_proof(self, seq_id, step_num, window_data, digest,
                  is_solution, pow_params, seq_info, completion_id: str | None = None,
                  admission_nonce: bytes | None = None):
        """Write a proof to disk."""
        # Proof-version agreement (TIP-0003): the miner-api proxy
        # stamps its own POW_PROOF_VERSION into the pow payload (it forces the
        # v3 fixed sampler profile keyed off THAT env), while this writer's
        # proof_version comes from the sampler process's env. Drift between
        # the two means every emitted proof is verifier-rejected (wrong
        # profile or wrong carrier) — fail loudly on the first proof instead.
        # Absent stamp (old proxy image / direct callers) skips the check.
        stamped = pow_params.get("proof_version") if isinstance(pow_params, dict) else None
        if stamped is not None and int(stamped) != int(self.proof_version):
            raise ValueError(
                f"POW_PROOF_VERSION disagreement: miner-api ingress stamped "
                f"proof_version={int(stamped)} but this sampler process is "
                f"configured for {int(self.proof_version)}; align the env on "
                f"both processes (proxy forces the v3 fixed profile only when "
                f"ITS env is >= 3)")
        # Create proof dictionary
        proof = {
            "ipfs_cid": pow_params["ipfs_cid"],
            "model_config_diff": self.model_config_diff,
            "sampling_params_diff": self.sampling_params_diff,
            "tick": pow_params["tick"],
            "target": pow_params["target"],
            "block_hash": pow_params["block_hash"],
            "vdf": pow_params["vdf"],
            "sequence_id": seq_id,
            "steps": step_num,
            "chosen_tokens": window_data["tokens"].cpu().tolist(),
            "chosen_probs": window_data["probs"].cpu().tolist(),
            "topk_logits": window_data["topk_logits"].cpu().tolist(),
            "topk_indices": window_data["topk_indices"].cpu().tolist(),
            "attention_mask": window_data["attention_mask"].cpu().tolist(),
            "sampling_u": window_data["sampling_u"].cpu().tolist(),
            "softmax_normalizers": window_data["softmax_normalizers"].cpu().tolist(),
            "logsumexp_stats": window_data["logsumexp_stats"].cpu().tolist(),
            "hash": digest[0].cpu().numpy().tobytes().hex(),
            "is_solution": bool(is_solution),
            "timestamp": time.time()
        }

        # Add model_identifier if available
        if self.model_identifier:
            proof["model_identifier"] = self.model_identifier
        if self.compute_precision:
            proof["compute_precision"] = self.compute_precision

        # Add header_prefix if available
        if "header_prefix" in pow_params:
            proof["header_prefix"] = pow_params["header_prefix"]

        # Add sequence info
        proof.update(seq_info)

        # Embed completion_id into model_config_diff (serialized as extra_flags)
        try:
            # Ensure model_config_diff is a dict we can extend
            base_diff = {}
            if isinstance(self.model_config_diff, dict):
                base_diff = dict(self.model_config_diff)
            elif isinstance(self.model_config_diff, str) and self.model_config_diff.strip():
                try:
                    import json as _json
                    base_diff = _json.loads(self.model_config_diff)
                except Exception:
                    base_diff = {"_diff": self.model_config_diff}
            # merge completion_id if provided
            if completion_id:
                base_diff["completion_id"] = completion_id
            # also allow passing pre-baked diff via pow_params
            if "model_config_diff" in pow_params and pow_params["model_config_diff"]:
                # best-effort merge: if pow_params contains JSON string, merge keys
                _mcd = pow_params["model_config_diff"]
                if isinstance(_mcd, dict):
                    base_diff.update(_mcd)
                elif isinstance(_mcd, str):
                    try:
                        import json as _json
                        base_diff.update(_json.loads(_mcd))
                    except Exception:
                        base_diff["_mcd"] = _mcd
            self.model_config_diff = base_diff
        except Exception:
            # Fallback: simple string with completion id
            if completion_id:
                self.model_config_diff = {"completion_id": completion_id}

        # V3 carrier (TIP-0003). proof["model_config_diff"] was
        # captured BEFORE the completion-id mutation above (by reference), so
        # for v3 rebuild it from the post-mutation value and merge the
        # admission nonce through the shared helper — the nonce and the
        # completion_id must both land on THIS proof. serialize_proof emits
        # the resulting string verbatim (canonical JSON) for version >= 3.
        proof["version"] = self.proof_version
        if self.proof_version >= pow_v3.V3_PROOF_VERSION:
            if admission_nonce is not None:
                proof["model_config_diff"] = pow_v3.merge_extra_flags_v3(
                    self.model_config_diff, bytes(admission_nonce).hex())
            else:
                _mcd = self.model_config_diff
                proof["model_config_diff"] = (
                    _mcd if isinstance(_mcd, str)
                    else pow_v3.canonical_json(_mcd or {}))
        elif admission_nonce is not None:
            raise ValueError("admission_nonce requires proof_version >= 3")

        # Generate filename
        filename = f"pow_proof_{seq_id}_{step_num}_{uuid.uuid4().hex[:8]}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Write to disk (optional)
        if os.environ.get("POW_SAVE_TO_DISK", "0") in ("1", "true", "True"):
            try:
                with open(filepath, 'w') as f:
                    json.dump(proof, f, indent=2)
            except Exception:
                pass

        # Serialize and write to file
        data = serialize_proof(proof)
        filename = f"pow_proof_{seq_id}_{step_num}_{uuid.uuid4().hex[:8]}.bin"
        filepath = os.path.join(self.output_dir, filename)        
        if os.environ.get("POW_SAVE_TO_DISK", "0") in ("1", "true", "True"):
            try: 
                with open(filepath, "wb") as f:
                    f.write(data)
            except Exception:
                print("UNABLE TO SAVE FB")
                pass
        return data, proof

def _to_bytes(x):
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return b''
        try:
            return bytes.fromhex(s)
        except ValueError:
            return base64.b64decode(s)
    raise TypeError(f"Expected bytes or str, got {type(x)}")

# @pow_profiler
def serialize_proof(obj: dict) -> bytes:
    b = flatbuffers.Builder(1024)

    proof_version = int(obj.get('version', 2))

    mid_off = b.CreateString(obj['model_identifier'])
    cp_off  = b.CreateString(obj['compute_precision'])
    ipfs_off  = b.CreateString(obj['ipfs_cid'])
    if proof_version >= pow_v3.V3_PROOF_VERSION:
        # v3 producer convention (TIP-0003): extra_flags is
        # canonical JSON, never pformat. A str is emitted verbatim (it is the
        # already-merged canonical JSON from the v3 merge helper).
        mcd = obj['model_config_diff']
        extra_str = mcd if isinstance(mcd, str) else pow_v3.canonical_json(mcd or {})
        extra_off = b.CreateString(extra_str)
    else:
        # v2 path untouched: legacy pformat blob, consensus-opaque.
        extra_off = b.CreateString(to_python_string(obj['model_config_diff']))
    
    # — raw bytes —
    tgt_off  = b.CreateByteVector(_to_bytes(obj['target']))
    vdf_off  = b.CreateByteVector(_to_bytes(obj['vdf']))
    block_hash_off = b.CreateByteVector(_to_bytes(obj['block_hash']))
    hash_off = b.CreateByteVector(_to_bytes(obj['hash']))
    hdr_off = b.CreateByteVector(_to_bytes(obj['header_prefix']))

    temp = float(obj['temperature'])
    p    = float(obj['top_p'])
    k    = int(obj['top_k']) & 0xFFFFFFFF
    rp   = float(obj['repetition_penalty'])        
        
    # — typed 1D vectors —
    def make_vec_uint32(data):
        b.StartVector(4, len(data), 4)
        for v in reversed(data):
            # force into C uint32 range
            val = int(v) & 0xFFFFFFFF
            b.PrependUint32(val)
        return b.EndVector()

    #  8-bit unsigned ints
    def make_vec_uint8(data):
        # element size=1, alignment=1
        b.StartVector(1, len(data), 1)
        for v in reversed(data):
            val = int(v) & 0xFF
            b.PrependUint8(val)
        return b.EndVector(len(data))

    # For a bool vector (PadMask):
    def make_vec_bool(data):
        # element size=1 byte, alignment=1
        b.StartVector(1, len(data), 1)
        for v in reversed(data):
            # Cast to Python bool
            b.PrependBool(bool(v))
        return b.EndVector(len(data))
        
    def make_vec_float32(data):
        b.StartVector(4, len(data), 4)
        for v in reversed(data):
            # round to IEEE754 float32 exactly
            f32 = np.float32(v).item()
            b.PrependFloat32(f32)
        return b.EndVector()

    ctoks = make_vec_uint32(obj['chosen_tokens'])
    pp    = make_vec_float32(obj['chosen_probs'])
    su    = make_vec_float32(obj['sampling_u'])
    sn    = make_vec_float32(obj['softmax_normalizers'])
    pt    = make_vec_uint32(obj['prompt_tokens'])
    pm    = make_vec_bool(obj['pad_mask'])

    # — 2D via wrappers with same precision guarantees —
    def _wrap_float32(row):
        FloatArray.FloatArrayStartValuesVector(b, len(row))
        for v in reversed(row):
            f32 = np.float32(v).item()
            b.PrependFloat32(f32)
        vec = b.EndVector()
        FloatArray.FloatArrayStart(b)
        FloatArray.FloatArrayAddValues(b, vec)
        return FloatArray.FloatArrayEnd(b)

    def _wrap_uint32(row):
        UIntArray.UIntArrayStartValuesVector(b, len(row))
        for v in reversed(row):
            u32 = int(v) & 0xFFFFFFFF
            b.PrependUint32(u32)
        vec = b.EndVector()
        UIntArray.UIntArrayStart(b)
        UIntArray.UIntArrayAddValues(b, vec)
        return UIntArray.UIntArrayEnd(b)

    logits_offs = [_wrap_float32(r) for r in obj['topk_logits']]
    Proof.StartTopkLogitsVector(b, len(logits_offs))
    for off in reversed(logits_offs): 
        b.PrependUOffsetTRelative(off)
    topk_logits_off = b.EndVector()

    idx_offs = [_wrap_uint32(r) for r in obj['topk_indices']]
    Proof.StartTopkIndicesVector(b, len(idx_offs))
    for off in reversed(idx_offs): 
        b.PrependUOffsetTRelative(off)
    topk_indices_off = b.EndVector()

    lse_offs = [_wrap_float32(r) for r in obj['logsumexp_stats']]
    Proof.StartLogsumexpStatsVector(b, len(lse_offs))
    for off in reversed(lse_offs): 
        b.PrependUOffsetTRelative(off)
    lse_off = b.EndVector()
    
    # — root table with explicit uint64 masking —
    Proof.Start(b)
    Proof.AddVersion(b,      proof_version )
    Proof.AddTick(b,      int(obj['tick'])      & 0xFFFFFFFFFFFFFFFF)
    Proof.AddTimestamp(b, int(obj['timestamp']) & 0xFFFFFFFFFFFFFFFF)
    Proof.AddIsSolution(b, 1 if obj['is_solution'] else 0)
    Proof.AddModelIdentifier(b, mid_off)
    Proof.AddComputePrecision(b, cp_off)
    Proof.AddIpfsCid(b, ipfs_off)
    Proof.AddExtraFlags(b, extra_off)
    Proof.AddTemperature(b, temp)
    Proof.AddTopP(b, p)
    Proof.AddTopK(b, k)
    Proof.AddRepetitionPenalty(b, rp)    

    Proof.AddTarget(b,         tgt_off)
    Proof.AddVdf(b,            vdf_off)
    Proof.AddHash(b,           hash_off)
    Proof.AddBlockHash(b,           block_hash_off)
    Proof.AddHeaderPrefix(b, hdr_off)

    Proof.AddChosenTokens(b,      ctoks)
    Proof.AddChosenProbs(b,       pp)
    Proof.AddSamplingU(b,         su)
    Proof.AddSoftmaxNormalizers(b, sn)
    Proof.AddLogsumexpStats(b,    lse_off)
    Proof.AddPromptTokens(b,      pt)
    Proof.AddPadMask(b,           pm)
    Proof.AddTopkLogits(b,        topk_logits_off)
    Proof.AddTopkIndices(b,       topk_indices_off)

    root = Proof.End(b)
    b.Finish(root, file_identifier=b"PROF")
    return bytes(b.Output())

# ======================================================================
# GPU SHA-256 (Triton) — byte-exact, fixed-length batched path
# ----------------------------------------------------------------------
# Removes the GPU->CPU->hashlib->GPU round trip for CUDA inputs. In every
# production call path the rows of a single sha256_many() call are the same
# length (rectangular `_build_msg`, and the 80-/32-byte double-SHA in
# `check_solution`); the per-row ragged case is already split into
# one-row calls upstream. So the kernel hashes a fixed `L` bytes per row —
# no ragged masking, one launch, no device round trip.
#
# Availability is gated on Triton exactly like vLLM gates its own kernels:
# the runtime image may force-disable Triton (HAS_TRITON=False), in which
# case we keep the byte-exact CPU hashlib fallback below. The CPU path is
# the reference oracle; the Triton path MUST produce identical bytes — see
# tests/test_sha256_gpu_equivalence.py.
# ======================================================================
try:
    # Authoritative flag inside the vLLM runtime (some images set it False).
    from vllm.triton_utils import HAS_TRITON as _VLLM_HAS_TRITON  # type: ignore
except Exception:
    _VLLM_HAS_TRITON = None

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORTABLE = True
except Exception:
    _TRITON_IMPORTABLE = False

# Explicit kill-switch independent of vLLM's flag (set to "1" to force CPU).
_POW_SHA256_DISABLE_TRITON = os.getenv("POW_SHA256_DISABLE_TRITON", "0") == "1"
_sha256_triton_warned = False

# SHA-256 round constants (host copy used only to build the device tensor).
_SHA256_K_PY = (
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
)
_sha256_k_cache: dict = {}  # device -> int64 (64,) tensor of round constants

# Rows (independent messages) processed per Triton program. Power of two.
_SHA256_ROWS = 64


def _sha256_use_triton() -> bool:
    """Use the Triton kernel only when Triton is importable, not force-disabled
    by the vLLM runtime, and not killed via the POW_SHA256_DISABLE_TRITON env."""
    if _POW_SHA256_DISABLE_TRITON or not _TRITON_IMPORTABLE:
        return False
    return _VLLM_HAS_TRITON is not False


def _sha256_k_dev(device) -> "torch.Tensor":
    t = _sha256_k_cache.get(device)
    if t is None:
        t = torch.tensor(_SHA256_K_PY, dtype=torch.int64, device=device)
        _sha256_k_cache[device] = t
    return t


if _TRITON_IMPORTABLE:

    @triton.jit
    def _rotr32(x, n: tl.constexpr):
        # x holds a 32-bit value in an int64; rotate right by n (0<n<32).
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    @triton.jit
    def _blkword(msg_ptr, row_base, bb, p: tl.constexpr, Lc, last8, bitlen,
                 rmask, ROWS: tl.constexpr):
        # Build schedule word `p` (0..15) of the current block from its 4 bytes,
        # applying SHA-256 padding (0x80) and the 64-bit big-endian bit length
        # where this position falls at/after the message end. `p` is
        # compile-time; bb/Lc/last8/bitlen are runtime scalars.
        word = tl.zeros((ROWS,), tl.int64)
        for k in range(4):
            pos = bb + (p * 4 + k)
            in_msg = pos < Lc
            is_pad = pos == Lc
            in_len = pos >= last8
            data = tl.load(msg_ptr + row_base + pos,
                           mask=rmask & in_msg, other=0).to(tl.int64)
            len_idx = pos - last8
            shift = tl.maximum(tl.minimum((7 - len_idx) * 8, 56), 0)
            len_byte = (bitlen >> shift) & 0xFF
            bval = data + 0x80 * is_pad.to(tl.int64) + len_byte * in_len.to(tl.int64)
            word = (word << 8) | (bval & 0xFF)
        return word & 0xFFFFFFFF

    @triton.jit
    def _store_word(out_ptr, out_row, widx: tl.constexpr, hv, rmask):
        # Serialize a 32-bit state word big-endian into out[*, widx*4 .. +4].
        tl.store(out_ptr + out_row + (widx * 4 + 0), ((hv >> 24) & 0xFF).to(tl.uint8), mask=rmask)
        tl.store(out_ptr + out_row + (widx * 4 + 1), ((hv >> 16) & 0xFF).to(tl.uint8), mask=rmask)
        tl.store(out_ptr + out_row + (widx * 4 + 2), ((hv >> 8) & 0xFF).to(tl.uint8), mask=rmask)
        tl.store(out_ptr + out_row + (widx * 4 + 3), (hv & 0xFF).to(tl.uint8), mask=rmask)

    @triton.jit
    def _sha256_fixed_kernel(msg_ptr, out_ptr, k_ptr, B, L, n_blocks,
                             ROWS: tl.constexpr):
        """One program hashes ROWS independent messages, each exactly L bytes.

        msg: (B, L) uint8 contiguous; out: (B, 32) uint8; k_ptr: (64,) int64.
        The 16-word message schedule lives in 16 named registers (m0..m15)
        rotated one slot per round, so at round t: m0=W[t-16], m1=W[t-15],
        m9=W[t-7], m14=W[t-2]. Triton 3.3.0's frontend rejects Python
        list/tuple subscripting and comprehensions inside @triton.jit, so no
        container is ever indexed.
        """
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)          # (ROWS,) int32
        rmask = rows < B
        row_base = rows.to(tl.int64) * L                # byte offset of each row

        Lc = L + 0  # scalar int; .to() broke on Triton>=3.6 (plain-int loop/arg scalars)
        bitlen = Lc * 8
        last8 = n_blocks * 64 - 8          # start of the 8 length bytes (int scalar)

        h0 = tl.full((ROWS,), 0x6a09e667, tl.int64)
        h1 = tl.full((ROWS,), 0xbb67ae85, tl.int64)
        h2 = tl.full((ROWS,), 0x3c6ef372, tl.int64)
        h3 = tl.full((ROWS,), 0xa54ff53a, tl.int64)
        h4 = tl.full((ROWS,), 0x510e527f, tl.int64)
        h5 = tl.full((ROWS,), 0x9b05688c, tl.int64)
        h6 = tl.full((ROWS,), 0x1f83d9ab, tl.int64)
        h7 = tl.full((ROWS,), 0x5be0cd19, tl.int64)

        for bidx in range(0, n_blocks):
            bb = bidx * 64  # range() var is a plain int on Triton>=3.6
            m0 = _blkword(msg_ptr, row_base, bb, 0, Lc, last8, bitlen, rmask, ROWS)
            m1 = _blkword(msg_ptr, row_base, bb, 1, Lc, last8, bitlen, rmask, ROWS)
            m2 = _blkword(msg_ptr, row_base, bb, 2, Lc, last8, bitlen, rmask, ROWS)
            m3 = _blkword(msg_ptr, row_base, bb, 3, Lc, last8, bitlen, rmask, ROWS)
            m4 = _blkword(msg_ptr, row_base, bb, 4, Lc, last8, bitlen, rmask, ROWS)
            m5 = _blkword(msg_ptr, row_base, bb, 5, Lc, last8, bitlen, rmask, ROWS)
            m6 = _blkword(msg_ptr, row_base, bb, 6, Lc, last8, bitlen, rmask, ROWS)
            m7 = _blkword(msg_ptr, row_base, bb, 7, Lc, last8, bitlen, rmask, ROWS)
            m8 = _blkword(msg_ptr, row_base, bb, 8, Lc, last8, bitlen, rmask, ROWS)
            m9 = _blkword(msg_ptr, row_base, bb, 9, Lc, last8, bitlen, rmask, ROWS)
            m10 = _blkword(msg_ptr, row_base, bb, 10, Lc, last8, bitlen, rmask, ROWS)
            m11 = _blkword(msg_ptr, row_base, bb, 11, Lc, last8, bitlen, rmask, ROWS)
            m12 = _blkword(msg_ptr, row_base, bb, 12, Lc, last8, bitlen, rmask, ROWS)
            m13 = _blkword(msg_ptr, row_base, bb, 13, Lc, last8, bitlen, rmask, ROWS)
            m14 = _blkword(msg_ptr, row_base, bb, 14, Lc, last8, bitlen, rmask, ROWS)
            m15 = _blkword(msg_ptr, row_base, bb, 15, Lc, last8, bitlen, rmask, ROWS)

            a, b_, c, d, e, f, g, hh = h0, h1, h2, h3, h4, h5, h6, h7
            for t in range(64):
                if t >= 16:
                    # m0 (=W[t-16]) is updated in place to W[t]
                    s0 = (_rotr32(m1, 7) ^ _rotr32(m1, 18) ^ (m1 >> 3)) & 0xFFFFFFFF
                    s1 = (_rotr32(m14, 17) ^ _rotr32(m14, 19) ^ (m14 >> 10)) & 0xFFFFFFFF
                    m0 = (m0 + s0 + m9 + s1) & 0xFFFFFFFF
                wt = m0
                kt = tl.load(k_ptr + t)
                S1 = (_rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)) & 0xFFFFFFFF
                ch = (e & f) ^ ((e ^ 0xFFFFFFFF) & g)
                temp1 = (hh + S1 + ch + kt + wt) & 0xFFFFFFFF
                S0 = (_rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)) & 0xFFFFFFFF
                maj = (a & b_) ^ (a & c) ^ (b_ & c)
                temp2 = (S0 + maj) & 0xFFFFFFFF
                hh = g
                g = f
                f = e
                e = (d + temp1) & 0xFFFFFFFF
                d = c
                c = b_
                b_ = a
                a = (temp1 + temp2) & 0xFFFFFFFF
                # rotate the 16-word schedule window left by one slot
                m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15 = \
                    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m0

            h0 = (h0 + a) & 0xFFFFFFFF
            h1 = (h1 + b_) & 0xFFFFFFFF
            h2 = (h2 + c) & 0xFFFFFFFF
            h3 = (h3 + d) & 0xFFFFFFFF
            h4 = (h4 + e) & 0xFFFFFFFF
            h5 = (h5 + f) & 0xFFFFFFFF
            h6 = (h6 + g) & 0xFFFFFFFF
            h7 = (h7 + hh) & 0xFFFFFFFF

        out_row = rows.to(tl.int64) * 32
        _store_word(out_ptr, out_row, 0, h0, rmask)
        _store_word(out_ptr, out_row, 1, h1, rmask)
        _store_word(out_ptr, out_row, 2, h2, rmask)
        _store_word(out_ptr, out_row, 3, h3, rmask)
        _store_word(out_ptr, out_row, 4, h4, rmask)
        _store_word(out_ptr, out_row, 5, h5, rmask)
        _store_word(out_ptr, out_row, 6, h6, rmask)
        _store_word(out_ptr, out_row, 7, h7, rmask)


def _sha256_many_triton(msg: torch.ByteTensor) -> torch.ByteTensor:
    """Fixed-length batched SHA-256 on the GPU, byte-identical to hashlib.

    Caller guarantees msg.is_cuda, B>0, L>0 and a uniform row length.
    """
    msg_c = msg.contiguous()
    B, L = msg_c.shape
    out = torch.empty((B, 32), dtype=torch.uint8, device=msg_c.device)
    n_blocks = (L + 9 + 63) // 64                      # >= 1; 1 byte 0x80 + 8 len
    k_dev = _sha256_k_dev(msg_c.device)
    grid = (triton.cdiv(B, _SHA256_ROWS),)
    _sha256_fixed_kernel[grid](msg_c, out, k_dev, B, L, n_blocks,
                               ROWS=_SHA256_ROWS)
    return out


def _sha256_many_cpu(msg: torch.ByteTensor) -> torch.ByteTensor:
    """Byte-exact reference path: ship the batch to CPU once, hash with
    hashlib, copy one (B,32) back. Also the fallback when Triton is off."""
    device = msg.device
    B, L = msg.shape

    # 1) single hop to CPU (contiguous, no-gradient)
    host = msg.detach().contiguous().to('cpu')
    arr  = host.numpy()  # uint8[B, L], zero-copy view

    # 2) output on CPU (pinned if source was CUDA) + numpy view
    out_cpu = torch.empty((B, 32), dtype=torch.uint8, device='cpu',
                          pin_memory=msg.is_cuda)
    out_np = out_cpu.numpy()

    # 3) per-row hash w/o creating Python bytes every time
    for i in range(B):
        d = hashlib.sha256(memoryview(arr[i])).digest()
        out_np[i, :] = np.frombuffer(d, dtype=np.uint8)

    # 4) single hop back to original device
    return out_cpu.to(device, non_blocking=True)


# @pow_profiler
def sha256_many(msg: torch.ByteTensor) -> torch.ByteTensor:
    """Batch SHA-256: (B, L) uint8 -> (B, 32) uint8 big-endian digests.

    Hashes the full L bytes of each row (all rows share L in every production
    call path). On CUDA with Triton available the work stays on the GPU; the
    CPU hashlib path is the byte-exact fallback/reference."""
    assert msg.dtype == torch.uint8 and msg.ndim == 2
    device = msg.device
    B, L = msg.shape

    # Degenerate shapes: keep them on the simple, proven path.
    if B == 0:
        return torch.empty((0, 32), dtype=torch.uint8, device=device)

    if msg.is_cuda and L > 0 and _sha256_use_triton():
        try:
            return _sha256_many_triton(msg)
        except Exception as exc:  # never break sampling on a kernel issue
            global _sha256_triton_warned
            if not _sha256_triton_warned:
                _sha256_triton_warned = True
                print(f"[pow_utils] Triton SHA-256 unavailable, falling back "
                      f"to CPU hashlib: {exc!r}")

    return _sha256_many_cpu(msg)

# @pow_profiler
def check_hash_against_target(digest: torch.ByteTensor,
                              target: torch.ByteTensor) -> torch.BoolTensor:
    """
    Interpret digest & target as Core does (little-endian integers).
    digest: (B,32) raw SHA256^2 output.
    target: (32,) big-endian target bytes.
    """
    # Convert target to little-endian layout
    t_le = target.flip(0)            # now least-significant at index 0

    B = digest.size(0)
    decided = torch.zeros(B, dtype=torch.bool, device=digest.device)
    result  = torch.zeros(B, dtype=torch.bool, device=digest.device)

    # Iterate from most-significant byte (index 31) down to 0
    for i in range(31, -1, -1):
        lt = (digest[:, i] < t_le[i]) & (~decided)
        gt = (digest[:, i] > t_le[i]) & (~decided)
        result |= lt
        decided |= lt | gt
        if decided.all():
            break
    result |= ~decided
    return result

# @pow_profiler
def _tok_le_bytes(tok_i64: torch.Tensor) -> torch.ByteTensor:
    """
    View an (B, L) int64 tensor as little-endian bytes.
    Output : (B, L*8) uint8   (8 bytes per token)
    """
    # Make sure memory is contiguous before the byte-level view
    t = tok_i64.contiguous()
    # Re-interpret the underlying buffer as uint8
    return t.view(torch.uint8).view(t.size(0), -1)

# @pow_profiler
def _u32le(x: torch.Tensor) -> torch.ByteTensor:
    """
    Convert a 32-bit unsigned/int tensor of shape (B,) or (B,1)
    to little-endian byte view (B, 4) uint8.
    """
    x32 = x.reshape(-1).to(torch.uint32).contiguous()     # (B,)
    x8  = x32.view(torch.uint8)                           # reinterpret as bytes
    return x8.view(-1, 4)                                 # (B,4)

# @pow_profiler
def _str_bytes(s: str,
               batch_size: int,
               device: torch.device = torch.device('cpu'),
               encoding: str = 'utf-8') -> torch.ByteTensor:
    """
    Convert a Python string `s` into a (batch_size, len(s)) uint8 ByteTensor.

    Args:
      s: the input string (e.g. 'fp16', 'bf16', etc.)
      batch_size: how many times to replicate this byte sequence (your B)
      device: where the tensor should live
      encoding: text encoding (usually 'utf-8' or 'ascii')

    Returns:
      A ByteTensor of shape (batch_size, len(s)) with dtype=torch.uint8.
    """
    b = s.encode(encoding)               # bytes object
    arr = torch.tensor(list(b),
                       dtype=torch.uint8,
                       device=device)   # shape (len(s),)
    # replicate along batch dimension
    return arr.unsqueeze(0).expand(batch_size, -1)  # (B, len(s))

# @pow_profiler
def _build_msg(header_prefix, v, T8, j4, ctx_bytes, precision, nonce=None):
    """Build the message for SHA-256 hashing.

    Args:
        header_prefix: (76,) ByteTensor of block header prefix (or (32,) for legacy block hash)
        v: (32,) ByteTensor of VDF
        T8: (8,) ByteTensor of tick
        j4: (B, 4) ByteTensor of step counter
        ctx_bytes: (B, L) ByteTensor of context tokens
        precision: (8,) ByteTensor of precision
        nonce: optional (32,) or (B, 32) ByteTensor — the v3 admission nonce,
            appended after precision for version >= 3 proofs mined with
            admission (TIP-0003). None preserves the v2 preimage
            byte-for-byte.

    Returns:
        (B, 76/32+32+8+4+L) ByteTensor of message
    """
    B = ctx_bytes.size(0)
    parts = [
        header_prefix.view(1, -1).expand(B, -1),
        v.view(1, -1).expand(B, -1),
        T8.view(1, -1).expand(B, -1),
        j4,
        ctx_bytes,
        precision,
    ]
    if nonce is not None:
        parts.append(nonce.view(-1, nonce.size(-1)).expand(B, -1)
                     if nonce.dim() > 1 else nonce.view(1, -1).expand(B, -1))
    return torch.cat(parts, dim=1)

def _digest_to_u(digest: torch.ByteTensor) -> torch.Tensor:
    """
    Map first 4 bytes of each hash to a float in [0,1).
    Properly handling unsigned values.
    """
    # Process each byte separately and combine with the right scaling
    b0 = digest[:, 0].to(torch.float32)
    b1 = digest[:, 1].to(torch.float32)
    b2 = digest[:, 2].to(torch.float32)
    b3 = digest[:, 3].to(torch.float32)

    # Combine with appropriate scaling for each byte position (little-endian)
    #
    # NB: this is float32 arithmetic. The docstring says [0,1) but that is not
    # what the code guarantees: float32 has a 24-bit significand, so the top
    # of the u32 range (roughly digests >= 0xFFFFFF80) rounds up to exactly
    # 4294967296.0 and the quotient is exactly 1.0. Callers must therefore
    # treat u == 1.0 as reachable and clamp the searchsorted result — see
    # _clamp_token_ids. The rounding is deterministic given the digest, so it
    # is consensus-safe as long as every implementation clamps identically;
    # it is NOT safe to feed the raw index to a gather.
    result = (b0 + b1 * 256 + b2 * 65536 + b3 * 16777216) / 4294967296.0
    return result


def _clamp_token_ids(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Clamp ``torch.searchsorted`` output to a valid token index, in place.

    ``torch.searchsorted(cdf, u)`` returns a value in ``[0, V]`` — V meaning
    "insert past the end". V is one past the last valid token id, so passing
    it to the ``gather`` in
    ``vllm/v1/sample/ops/topk_topp_sampler.py`` is an out-of-range index
    (silent garbage on CUDA, IndexError on CPU).

    Two independent routes reach V:

    * ``_digest_to_u`` computes in float32 and can return exactly 1.0 for the
      largest digests, while a normalised CDF's last entry is only
      *approximately* 1.0 and is frequently a hair below it.
    * A degenerate CDF (all-zero row, e.g. a fully masked vocabulary) puts
      every u past the end.

    Clamping to ``V - 1`` picks the last token in the CDF, which is what an
    exact-arithmetic implementation would have selected for u == 1.0, so the
    fix does not change the sampled token in any case that was already valid.
    """
    if vocab_size <= 0:
        return token_ids
    return token_ids.clamp_max_(vocab_size - 1)

# @pow_profiler
def hex_to_bytes_tensor(hex_str: str, device="cpu") -> torch.ByteTensor:
    if not hex_str:
        return torch.empty(0, dtype=torch.uint8, device=device)
    # 1) Use binascii to decode hex to raw bytes
    raw = binascii.unhexlify(hex_str)
    # 2) Create a CPU uint8 tensor *without* building an intermediate Python list
    #    torch.frombuffer exists as of PyTorch 1.10
    t = torch.frombuffer(raw, dtype=torch.uint8)
    # 3) Only pay the copy cost if you actually need it on GPU
    if device != "cpu":
        # non_blocking=True lets it overlap with ops if you pin memory
        t = t.to(device, non_blocking=True)
    return t

def nbits_to_target(nbits: int) -> torch.ByteTensor:
    """Convert Bitcoin nBits compact format to 256-bit target tensor."""
    # Extract exponent and mantissa
    exponent = nbits >> 24
    mantissa = nbits & 0x00ffffff

    # Calculate target value
    if exponent <= 3:
        target_int = mantissa >> (8 * (3 - exponent))
    else:
        target_int = mantissa << (8 * (exponent - 3))

    # Convert to 32-byte tensor (big-endian)
    target_bytes = target_int.to_bytes(32, byteorder='big', signed=False)
    return torch.tensor(list(target_bytes), dtype=torch.uint8)

def _has_pow(metadata) -> bool:
    if metadata.extra_args is None:
        return False
    return any( (i in metadata.extra_args and metadata.extra_args[i].get("pow"))
                for i in range(len(metadata.output_token_ids)) )

def to_python_string(input_data):
    """
    Converts a Python dict or JSON string into a plain Python literal string.
    
    :param input_data: A dict/list or a JSON-formatted string.
    :return: A string containing the Python literal representation.
    """
    # If it's a string, try parsing as JSON
    if isinstance(input_data, str):
        try:
            obj = json.loads(input_data)
        except json.JSONDecodeError:
            # Not valid JSON; return the original string
            return input_data
    else:
        obj = input_data
    
    # Use pprint to get a nicely formatted Python literal string
    return pformat(obj)
