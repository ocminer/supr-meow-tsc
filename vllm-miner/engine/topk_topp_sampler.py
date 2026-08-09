# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os
import time
import types

import torch
import torch.nn as nn
from packaging import version

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

# PoW imports — injected at Docker build time from shared-utils/pow-utils/
# Guarded so stock vLLM still loads when shared-utils are absent.
_POW_AVAILABLE = False
try:
    from vllm.sampling.pow_utils import (
        POW_WINDOW_SIZE,
        Logger as PowLogger,
        PowHasher,
        ProofWriter,
        RingBuffers,
        RowManager,
        apply_topk_topp_mask,
    )
    from vllm.sampling.zmq_pow_writer import MiningResponseSubmitter

    if os.environ.get('POW_USE_ASYNC_PROCESSING', '1') in (
            '1', 'true', 'True'):
        try:
            from vllm.sampling.common_sampler_helper_async_proper import (
                CommonSamplerHelper,
            )
        except ImportError:
            from vllm.sampling.common_sampler_helper import (
                CommonSamplerHelper,
            )
    else:
        from vllm.sampling.common_sampler_helper import CommonSamplerHelper

    _POW_AVAILABLE = True
except ImportError:
    POW_WINDOW_SIZE = 256  # default fallback

logger = init_logger(__name__)


class TopKTopPSampler(nn.Module):
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode
        # flashinfer optimization does not apply if intermediate
        # logprobs/logits after top_k/top_p need to be returned
        if (
            logprobs_mode not in ("processed_logits", "processed_logprobs")
            and current_platform.is_cuda()
        ):
            if envs.VLLM_USE_FLASHINFER_SAMPLER:
                from vllm.v1.attention.backends.flashinfer import FlashInferBackend

                capability = current_platform.get_device_capability()
                assert capability is not None
                if not FlashInferBackend.supports_compute_capability(capability):
                    capability_str = capability.as_version_str()
                    raise RuntimeError(
                        "FlashInfer does not support compute capability "
                        f"{capability_str}, unset VLLM_USE_FLASHINFER_SAMPLER=1."
                    )
                # Users must opt in explicitly via VLLM_USE_FLASHINFER_SAMPLER=1.
                logger.info_once(
                    "Using FlashInfer for top-p & top-k sampling.",
                    scope="global",
                )
                self.forward = self.forward_cuda
            else:
                logger.debug_once(
                    "FlashInfer top-p/top-k sampling is available but disabled "
                    "by default. Set VLLM_USE_FLASHINFER_SAMPLER=1 to opt in "
                    "after verifying accuracy for your workloads."
                )
                self.forward = self.forward_native

        elif current_platform.is_cpu():
            arch = current_platform.get_cpu_architecture()
            # Fall back to native implementation for POWERPC and RISCV.
            # On PowerPC argmax produces incorrect output with torch.compile.
            # PR: https://github.com/vllm-project/vllm/pull/26987
            if arch in (CpuArchEnum.RISCV, CpuArchEnum.POWERPC):
                self.forward = self.forward_native
            else:
                self.forward = self.forward_cpu
        elif (
            logprobs_mode not in ("processed_logits", "processed_logprobs")
            and rocm_aiter_ops.is_enabled()
        ):
            try:
                import aiter.ops.sampling  # noqa: F401

                self.aiter_ops = torch.ops.aiter
                logger.info_once(
                    "Using aiter sampler on ROCm (lazy import, sampling-only)."
                )
                self.forward = self.forward_hip
            except ImportError:
                logger.warning_once(
                    "aiter.ops.sampling is not available on ROCm. "
                    "Falling back to forward_native implementation."
                )
                self.forward = self.forward_native
        else:
            self.forward = self.forward_native

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """More optimized implementation for top-k and top-p sampling."""
        # We prefer `random_sample` over `flashinfer_sample` when sorting is
        # not needed. This is because `random_sample` does not require
        # CPU-GPU synchronization while `flashinfer_sample` does.
        if (k is None and p is None) or generators:
            if generators:
                logger.debug_once(
                    "FlashInfer 0.2.3+ does not support "
                    "per-request generators. Falling back to "
                    "PyTorch-native implementation."
                )
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in ("processed_logits", "processed_logprobs"), (
            "FlashInfer does not support returning logits/logprobs"
        )
        # flashinfer sampling functions expect contiguous logits.
        # In flex_attn/triton_attn fp32 inference, logits can be non-contiguous
        # because of slicing operation in logits_processor.
        return flashinfer_sample(logits.contiguous(), k, p, generators), None

    def forward_cpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling for CPU.

        The logits tensor may be updated in-place.
        """
        logits = apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        if len(generators) != logits.shape[0]:
            return compiled_random_sample(logits), logits_to_return

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

        return probs.div_(q).argmax(dim=-1).view(-1), logits_to_return

    def forward_hip(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # FIXME: Fix aiter_sampler's accuracy issue and remove this flag
        DISABLE_AITER_SAMPLER = True
        """Optimized ROCm/aiter path (same structure as forward_cuda)."""
        if (k is None and p is None) or generators:
            if generators:
                logger.warning_once(
                    "aiter sampler does not support per-request generators; "
                    "falling back to PyTorch-native."
                )
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in (
            "processed_logits",
            "processed_logprobs",
        ), "aiter sampler does not support returning logits/logprobs."
        if DISABLE_AITER_SAMPLER:
            return self.forward_native(logits, generators, k, p)
        return self.aiter_sample(logits, k, p, generators), None

    def aiter_sample(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        """Sample from logits using aiter ops."""
        use_top_k = k is not None
        use_top_p = p is not None
        # Joint k+p path
        if use_top_p and use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_k_top_p_sampling_from_probs(
                probs,
                None,
                *_to_tensor_scalar_tuple(k),
                *_to_tensor_scalar_tuple(p),
                deterministic=True,
            )
            return next_token_ids.view(-1)
        # Top-p only path
        elif use_top_p:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_p_sampling_from_probs(
                probs, None, *_to_tensor_scalar_tuple(p), deterministic=True
            )
            return next_token_ids.view(-1)
        # Top-k only path
        elif use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            renorm_probs = self.aiter_ops.top_k_renorm_probs(
                probs, *_to_tensor_scalar_tuple(k)
            )
            return torch.multinomial(renorm_probs, num_samples=1).view(-1)
        raise RuntimeError("aiter_sample was called with no active top-k or top-p.")


# Note: this is a workaround for
# https://github.com/pytorch/pytorch/pull/151218
@torch.compile(dynamic=True)
def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)


def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.
    Note however that it involves a GPU->CPU sync which can be detrimental for
    async scheduling performance.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.

    NOTE: This function includes CPU-GPU synchronization, while `random_sample`
    does not. Call this function at the end of the forward pass to minimize
    the synchronization overhead.
    """
    import flashinfer

    if version.parse(flashinfer.__version__) < version.parse("0.2.3"):
        raise ImportError(
            "FlashInfer version >= 0.2.3 required for top-k and top-p sampling. "
        )

    assert not (k is None and p is None)
    if k is None:
        # Top-p only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True
        )
    elif p is None:
        # Top-k only.
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True
        )
    else:
        # Both top-k and top-p.
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True
        )

    return next_token_ids.view(-1)


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


# ---------------------------------------------------------------------------
# PoW-aware sampler (v1-compatible)
# ---------------------------------------------------------------------------

class PowTopKTopPSampler(TopKTopPSampler):
    """PoW-aware sampler that integrates proof-of-work into token sampling."""

    def __init__(self,
                 max_concurrency: int = 1024,
                 eos_token_id: int = 2,
                 window_size: int = POW_WINDOW_SIZE,
                 device: torch.device | None = None):
        super().__init__()
        if not _POW_AVAILABLE:
            raise ImportError(
                "PoW shared-utils not installed. Install pow_utils, "
                "common_sampler_helper, and zmq_pow_writer into "
                "vllm/sampling/ or set POW_PROCESSOR_MODE=disabled.")
        self._common = CommonSamplerHelper(self)

        # Hijack vLLM's call-site so the engine calls _pow_forward
        self.forward = types.MethodType(
            PowTopKTopPSampler._pow_forward, self)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.max_concurrency = max_concurrency
        self.eos_token_id = eos_token_id
        self.window_size = window_size

        self.logger = PowLogger()
        self.row_manager = RowManager(max_concurrency)
        self.ring_buffers = RingBuffers(window_size, max_concurrency,
                                        device=self.device)
        self.pow_hasher = PowHasher()
        self.proof_writer = ProofWriter()

        processor_mode = os.environ.get('POW_PROCESSOR_MODE', 'python')
        if processor_mode != 'cpp':
            self.submitter = MiningResponseSubmitter()
            self.logger.log("Initialized Python MiningResponseSubmitter",
                            "INIT")
        else:
            self.submitter = None
            self.logger.log(
                "Skipping Python submitter (using C++ processor)", "INIT")

        self.seq_caches: dict[int, dict[str, torch.Tensor]] = {}
        self.seq_params: dict[int, dict[str, float]] = {}
        self.page_size = 16
        self.prev_max_seq_len = 0
        self._last_cleanup = time.time()

        self._pre_temp_logits = None
        self._log_Z = None
        self._sampling_tensors = None
        self._req_id_to_sid: dict[str, int] = {}
        self._next_sid: int = 0

    @staticmethod
    def _pow_apply_top_k_top_p(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply canonical PoW top-k/top-p masking.

        PoW proofs use a fixed number of FlatBuffer slots, so tokens tied at
        the k-th boundary are excluded canonically. This intentionally differs
        from upstream vLLM's ordinary inference sampler.
        """
        # Body lives in pow_utils.apply_topk_topp_mask (single source for the sampler
        # and the equivalence test). The top_p == 1.0 path uses a topk(50)-threshold
        # mask (no full-vocab sort/scatter); p < 1.0 keeps the legacy sort.
        return apply_topk_topp_mask(logits, k, p)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def _pow_forward(
        self,
        logits: torch.Tensor,
        metadata,
        pre_temp_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """
        PoW sampling forward pass.

        Returns (sampled_token_ids, None) to match v0.16 TopKTopPSampler
        contract — the second element would be processed logprobs but PoW
        requests do not support logprobs (guard is in Sampler.forward).
        """
        device = logits.device
        B = logits.shape[0]
        proof_logits = pre_temp_logits if pre_temp_logits is not None else logits
        proof_logits = proof_logits.detach().to(torch.float32)

        # -INF LOGIT FIX (2026-08-09). vLLM masks a few IN-VOCAB token positions to
        # -inf (reserved/disallowed tokens; measured: 2 of 151936 for Qwen3-8B —
        # NOT trailing padding, the tensor is already true-vocab-wide). Those -inf
        # entries poison the proof's PLAIN-MEAN stats: logsumexp_stats[4]=
        # mean(sorted[2000:]) and [5]=mean(all) become -inf (logsumexp[0] is
        # immune, which is why only 4/5 were -inf). Our verifier replays raw HF
        # logits (finite everywhere) and expects finite stats; -inf -> infinite
        # Mahalanobis -> p=0 every step -> RED (R2-R5). llama.cpp emits raw finite
        # logits and is unaffected. Fix: replace -inf with the per-row FINITE
        # minimum before any PoW computation so every field (snapshot bucket
        # means, logsumexp_full, top-k/top-p, sampling, probes) is finite. Effect
        # on the means is negligible (a couple already-lowest tokens out of
        # 151936), keeping the feature vector inside the verifier envelope. No-op
        # when all finite.
        _inf_mask = torch.isinf(proof_logits)
        if bool(_inf_mask.any()):
            _row_min = torch.nan_to_num(
                proof_logits.masked_fill(_inf_mask, float("nan")),
                nan=float("inf")).min(dim=-1, keepdim=True).values
            proof_logits = torch.where(_inf_mask, _row_min.expand_as(proof_logits), proof_logits)
            _lmask = torch.isinf(logits)
            if bool(_lmask.any()):
                _lmin = torch.nan_to_num(
                    logits.to(torch.float32).masked_fill(_lmask, float("nan")),
                    nan=float("inf")).min(dim=-1, keepdim=True).values
                logits = torch.where(_lmask, _lmin.expand_as(logits).to(logits.dtype), logits)
            if not getattr(self, "_inf_fix_logged", False):
                self._inf_fix_logged = True
                try:
                    self.logger.log(
                        f"[inf-fix] replaced {int(_inf_mask.sum().item())} -inf "
                        f"logit entries with per-row finite min", "INIT")
                except Exception:
                    pass

        # POW_PROFILE=1: per-section wall-time (cuda-synced), averaged every 128
        # steps. Diagnostic only — the syncs add overhead, so never leave on.
        _prof = os.environ.get("POW_PROFILE") == "1"
        if _prof:
            import time as _t
            if not hasattr(self, "_prof_acc"):
                self._prof_acc = {}
                self._prof_n = 0
            torch.cuda.synchronize()
            self._prof_t = _t.perf_counter()

            def _pp(tag):
                torch.cuda.synchronize()
                now = _t.perf_counter()
                self._prof_acc[tag] = self._prof_acc.get(tag, 0.0) + (now - self._prof_t) * 1000.0
                self._prof_t = now
        else:
            def _pp(tag):
                pass

        batch_req_ids = self._stable_req_ids(metadata, B)
        seq_ids: list[int] = []
        for rid in batch_req_ids:
            sid = self._req_id_to_sid.get(rid)
            if sid is None:
                sid = self._next_sid
                self._req_id_to_sid[rid] = sid
                self._next_sid += 1
            seq_ids.append(sid)

        for batch_row, rid in enumerate(batch_req_ids):
            sid = seq_ids[batch_row]

            if sid not in self.seq_params:
                rep_pen = metadata.repetition_penalties[batch_row]
                completion_id = None
                request_item_id_arg = None
                if metadata.extra_args and batch_row in metadata.extra_args:
                    args = metadata.extra_args[batch_row]
                    completion_id = args.get('completion_id')
                    request_item_id_arg = args.get('request_item_id')

                if not completion_id:
                    completion_id = rid

                pow_snapshot = None
                if metadata.extra_args and batch_row in metadata.extra_args:
                    pow_data = metadata.extra_args[batch_row].get("pow")
                    if pow_data:
                        pow_snapshot = {
                            "tick": int(pow_data["tick"]),
                            "request_id": int(pow_data["request_id"]),
                            "difficulty": pow_data["difficulty"],
                            "block_hash": pow_data["block_hash"],
                            "vdf": pow_data["vdf"],
                            "target": pow_data["target"],
                            "header_prefix": pow_data["header_prefix"],
                            "ipfs_cid": pow_data.get("ipfs_cid"),
                            # Slice 11.4 — model-adjusted share target
                            # the broker derived (base_share_target *
                            # ModelDifficultyNormalizer / model.difficulty).
                            # Optional: absent when the broker isn't
                            # in share-mode for this lease, in which
                            # case the sampler's check_share_solution
                            # returns all-False and behaviour is
                            # identical to the pre-slice-11 path.
                            "share_target": pow_data.get("share_target"),
                            # Per-request audit flag: emit a proof for every
                            # window of this sequence regardless of solution/
                            # share thresholds, routed to the completion-audit
                            # cache (proof_purpose=audit), never mining. Set by
                            # the proxy's audit injection for non-mining models
                            # (e.g. the 27B inference model on a dual-backend
                            # worker). common_sampler_helper reads it via .get,
                            # so absence is simply "no audit emission".
                            "audit_emit": bool(pow_data.get("audit_emit")),
                        }

                self.seq_params[sid] = {
                    "temperature": float(metadata.temperature[batch_row]),
                    "top_p": (float(metadata.top_p[batch_row])
                              if metadata.top_p is not None else 1.0),
                    "top_k": (int(metadata.top_k[batch_row])
                              if metadata.top_k is not None
                              else logits.shape[1]),
                    "repetition_penalty": (float(rep_pen)
                                           if rep_pen else 1.0),
                    "completion_id": (
                        (request_item_id_arg or completion_id) or rid),
                    "request_item_id": (request_item_id_arg or rid),
                    "base_completion_id": completion_id,
                    "pow_snapshot": pow_snapshot,
                }

        # 1. Update PoW params from first real request
        if metadata.extra_args:
            for i in range(B):
                args_i = metadata.extra_args.get(i, None)
                if args_i and "pow" in args_i:
                    self.pow_hasher.update_from_payload(args_i["pow"])
                    break

        _pp("ids")
        # 2. Row allocation.
        # prompt_map is consumed ONLY by _ensure_rows, and only for seq_ids that
        # don't yet have a row (ensure_rows: `if sid in seqid_to_row: continue`).
        # _safe_prompt_tokens does a per-row `.tolist()` device sync, so building
        # the whole-batch map every step costs ~B GPU->CPU syncs/step for values
        # that are discarded on all but a sequence's FIRST step of its lifetime
        # (prompts are window-invariant). POW_FAST_SETUP=1 computes prompts only
        # for genuinely new seq_ids. self.row_manager.seqid_to_row is the exact
        # dict ensure_rows checks, so this is behavior-identical (proof-neutral).
        if os.environ.get("POW_FAST_SETUP", "0") == "1":
            _known = self.row_manager.seqid_to_row
            prompt_map = {sid: self._safe_prompt_tokens(metadata, batch_row)
                          for batch_row, sid in enumerate(seq_ids)
                          if sid not in _known}
        else:
            prompt_map = {sid: self._safe_prompt_tokens(metadata, batch_row)
                          for batch_row, sid in enumerate(seq_ids)}
        _pp("promptmap")
        self._ensure_rows(seq_ids, prompt_map)
        rows = [self.row_manager.get_row(s) for s in seq_ids]

        _pp("rows")
        # 3. Top-k / top-p (temperature already applied by Sampler.sample)
        logits_f32 = logits.detach().clone()
        logsumexp_full = torch.logsumexp(logits_f32, dim=-1)

        if metadata.top_p is not None or metadata.top_k is not None:
            logits_f32 = self._pow_apply_top_k_top_p(
                logits_f32, metadata.top_k, metadata.top_p)

        _pp("mask")
        # 4. Snapshot pre-temperature logits for verifier replay.
        vocab_size = proof_logits.size(-1)
        batch_size = proof_logits.size(0)
        mean_tensor = torch.zeros(
            (batch_size, 6), dtype=torch.float32, device=device)
        mean_tensor[:, 0] = logsumexp_full
        # POW_FAST_STATS: replace the full-vocab STABLE sort (~3.8 ms/step on a
        # 5090, the dominant per-step PoW cost) with topk(2000)+tail-mean
        # (~0.5 ms, ~8x). Microbench-verified vs the full sort: top-50 values &
        # indices byte-identical, bucket means within ~1.5e-8. Opt-in until the
        # pool full-tier audit confirms the optimized build.
        if os.environ.get("POW_FAST_STATS", "0") == "1" and vocab_size > 2000:
            _tv, _ti = torch.topk(proof_logits, 2000, dim=-1, sorted=True)
            topk_vals = _tv[:, :50]
            topk_idx = _ti[:, :50]
            mean_tensor[:, 1] = _tv[:, :50].mean(-1)
            mean_tensor[:, 2] = _tv[:, 50:500].mean(-1)
            mean_tensor[:, 3] = _tv[:, 500:2000].mean(-1)
            mean_tensor[:, 4] = (proof_logits.sum(-1) - _tv.sum(-1)) / (vocab_size - 2000)
            mean_tensor[:, 5] = proof_logits.mean(-1)
        else:
            sorted_logits, sorted_indices = torch.sort(
                proof_logits, dim=-1, descending=True, stable=True)
            topk_vals = sorted_logits[:, :50]
            topk_idx = sorted_indices[:, :50]
            if vocab_size >= 50:
                mean_tensor[:, 1] = torch.mean(sorted_logits[:, :50], dim=-1)
            if vocab_size >= 500:
                mean_tensor[:, 2] = torch.mean(sorted_logits[:, 50:500], dim=-1)
            if vocab_size >= 2000:
                mean_tensor[:, 3] = torch.mean(sorted_logits[:, 500:2000], dim=-1)
            if vocab_size > 2000:
                mean_tensor[:, 4] = torch.mean(sorted_logits[:, 2000:], dim=-1)
            mean_tensor[:, 5] = torch.mean(sorted_logits, dim=-1)

        probe_step = max(vocab_size // 20, 1)
        probe_indices_list = torch.arange(
            0, vocab_size, probe_step, device=device)[:20]

        probe_logits = proof_logits.gather(
            1, probe_indices_list.unsqueeze(0).expand(batch_size, -1))

        extended_logits = torch.cat([topk_vals, probe_logits], dim=1)
        extended_indices = torch.cat([
            topk_idx,
            probe_indices_list.unsqueeze(0).expand(
                batch_size, -1).to(torch.int32),
        ], dim=1)

        _pp("stats")
        # 5. Probs & CDF
        probs = torch.softmax(logits_f32, -1, dtype=torch.float32)
        cdfs = torch.cumsum(probs, -1)

        _pp("cdf")
        # 6. PoW sampling
        rows_tensor = torch.as_tensor(
            rows, device=device, dtype=torch.long)
        steps = (self.ring_buffers.steps[rows_tensor].clone()
                 % self.window_size)
        contexts = self._get_context_windows(seq_ids).to(device)

        token_ids, u_vals, _digests = self.pow_hasher.batch_sample_tokens(
            contexts, steps, cdfs,
            self.proof_writer.compute_precision,
            ring_buffers=self.ring_buffers,
            rows_tensor=rows_tensor,
        )

        _pp("sample")
        # 7. Ring-buffer writes
        pos = self.ring_buffers.get_positions(rows_tensor)
        probs_sel = cdfs.gather(1, token_ids[:, None]).squeeze(1)

        page_flip = self._update_caches(seq_ids, token_ids)
        self.ring_buffers._write_buffers(
            pos, rows_tensor,
            extended_logits, extended_indices,
            token_ids, probs_sel,
            u_vals, page_flip,
            torch.logsumexp(logits_f32, -1),
            mean_tensor)

        self.ring_buffers.increment_steps(rows_tensor)

        _pp("ringwrite")
        # 8. EOS / PoW checks / cleanup
        self._check_eos(seq_ids, token_ids)
        self._check_pow_solutions(seq_ids)
        self._cleanup_stale_sequences()

        _pp("checks")
        if _prof:
            self._prof_n += 1
            if self._prof_n % 128 == 0:
                tot = sum(self._prof_acc.values())
                line = " ".join(f"{k}={v/self._prof_n:.2f}"
                                for k, v in self._prof_acc.items())
                print(f"[powprof] B={B} avg ms/step: {line} TOTAL={tot/self._prof_n:.2f}",
                      flush=True)

        return token_ids.to(torch.int64), None

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stable_req_ids(meta, batch_size: int) -> list[str]:
        if meta.extra_args:
            return [meta.extra_args[i]["req_id"]
                    if i in meta.extra_args else f"dummy-{i}"
                    for i in range(batch_size)]
        return [f"dummy-{i}" for i in range(batch_size)]

    def _safe_prompt_tokens(self, meta, i: int) -> list[int]:
        if meta.prompt_token_ids is None:
            return []
        t = meta.prompt_token_ids[i][
            meta.prompt_token_ids[i] != meta.extra_args[i]['pad_id']]
        return [] if t is None or t.numel() == 0 else t.tolist()

    # Delegate to CommonSamplerHelper
    def log_prompt_data(self, sampling_metadata):
        return self._common.log_prompt_data(sampling_metadata)

    def _detect_real_inference(self, sampling_metadata):
        return self._common.detect_real_inference(sampling_metadata)

    def _process_solution(self, seq_id, row):
        return self._common.process_solution(seq_id, row)

    def _reset_sampler_state(self):
        return self._common.reset_sampler_state()

    def _init_sequence_cache(self, seq_id, prompt_tokens):
        return self._common.init_sequence_cache(seq_id, prompt_tokens)

    def _get_context_windows(self, seq_ids):
        return self._common.get_context_windows(seq_ids)

    def _ensure_rows(self, seq_ids, prompt_mapping):
        return self._common.ensure_rows(seq_ids, prompt_mapping)

    def _update_caches(self, seq_ids, tokens):
        return self._common.update_caches(seq_ids, tokens)

    def _free_sequence(self, seq_id):
        return self._common.free_sequence(seq_id)

    def _check_eos(self, seq_ids, tokens):
        return self._common.check_eos(seq_ids, tokens)

    def _cleanup_stale_sequences(self, max_age=300, interval=60):
        return self._common.cleanup_stale_sequences(max_age, interval)

    def _check_pow_solutions(self, seq_ids):
        return self._common.check_pow_solutions(seq_ids)

    def _ensure_sorted_topk(self, topk_logits, topk_indices):
        return self._common.ensure_sorted_topk(topk_logits, topk_indices)

    def _process_pow_params(self, sampling_metadata):
        return self._common.process_pow_params(sampling_metadata)

    def _alloc_row(self, seq_id: int) -> int | None:
        if seq_id in self.row_manager.seqid_to_row:
            return self.row_manager.seqid_to_row[seq_id]
        if self.row_manager.free_rows:
            row = min(self.row_manager.free_rows)
            self.row_manager.seqid_to_row[seq_id] = row
            self.row_manager.free_rows.remove(row)
            return row
        oldest_sid, oldest_row = self.row_manager.get_oldest_sequence(
            self.ring_buffers.steps)
        if oldest_sid is not None:
            self._free_sequence(oldest_sid)
            return self._alloc_row(seq_id)
        return None
