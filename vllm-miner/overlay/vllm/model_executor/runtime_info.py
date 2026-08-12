# vllm/model_executor/runtime_info.py

from dataclasses import dataclass, fields
from huggingface_hub import HfApi
import logging
import pathlib
import os
import re
import torch
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------- helpers -------------------------------------------------

_DTYPE_STR = {
    torch.float16:         "fp16",
    torch.bfloat16:        "bf16",
    torch.float32:         "fp32",
    torch.float64:         "fp64",
    torch.int8:            "int8",
    # Collapse all FP8 variants under a single canonical "fp8" string. The
    # verifier loads model_id-specific weights, so E4M3 vs E5M2 is implicit
    # in the model checkpoint rather than the precision tag.
    getattr(torch, "float8_e4m3", None):    "fp8",
    getattr(torch, "float8_e4m3fn", None):  "fp8",
    getattr(torch, "float8_e4m3fnuz", None): "fp8",
    getattr(torch, "float8_e5m2", None):    "fp8",
    getattr(torch, "float8_e5m2fnuz", None): "fp8",
}

def _dtype_name(dtype) -> str:
    # dtype can be a torch.dtype or a string (when it came from CLI)
    if isinstance(dtype, str):
        d = dtype.lower()
        # Normalize all common FP8 spellings to canonical "fp8".
        if d in ("float8", "f8", "f8e4m3", "f8e5m2", "fp8"):
            return "fp8"
        return d
    return _DTYPE_STR.get(dtype, str(dtype))


# Verifier vocabulary — must mirror proof_verifier.py:993 and :1357 plus the
# C++ quick verifier (which hashes the string as bytes). Adding a value here
# without updating the verifier is a contract break.
_VERIFIER_PRECISION_VOCAB = {"fp16", "bf16", "fp32", "int8", "fp8"}


def _get_quant_config(model_cfg):
    """Resolve the quantization config across vLLM versions.

    - older: `model_cfg.quantization_config` (top-level)
    - v0.10/v0.16/v0.19: `model_cfg.model_arch_config.quantization_config`
    """
    qcfg = getattr(model_cfg, "quantization_config", None)
    if qcfg is not None:
        return qcfg
    marc = getattr(model_cfg, "model_arch_config", None)
    if marc is not None:
        return getattr(marc, "quantization_config", None)
    return None


def _quant_flavour(model_cfg) -> str | None:
    """Best-effort quantization flavour string for diagnostics.

    Returned as part of model_config_diff (NOT compute_precision) so the
    verifier replay path is unaffected.
    """
    qcfg = _get_quant_config(model_cfg)
    if qcfg is not None:
        if isinstance(qcfg, dict):
            algo = str(qcfg.get("quant_method", "awq")).lower()
            bits = qcfg.get("bits", 8)
        else:
            algo = str(getattr(qcfg, "quant_method", "awq")).lower()
            bits = getattr(qcfg, "bits", 8)
        return f"int{bits}-{algo}"
    quant_str = getattr(model_cfg, "quantization", None)
    if quant_str:
        return str(quant_str).lower()
    return None


def _detect_precision(model_cfg) -> str:
    """Return the EFFECTIVE replay dtype the verifier understands.

    The verifier's vocabulary is {fp16, bf16, fp32, int8, fp8} — see
    proof_verifier.py:993 and :1357 (Python full verifier) and the
    quick_verifier (which hashes the string as bytes). Anything outside
    that set will not replay correctly: the Python verifier silently
    coerces unknown strings to fp32 and the dtype-mismatch check at
    :1357 then rejects.

    Pure quantization flavour (awq, gptq, etc.) goes into
    model_config_diff for diagnostics — separate from compute_precision
    which selects the replay dtype.

    Raises if the resolved dtype is not in the verifier vocabulary,
    rather than silently mislabel.
    """
    dtype = getattr(model_cfg, "dtype", None)
    if dtype is None or (isinstance(dtype, str) and dtype.lower() == "auto"):
        raise RuntimeError(
            "RuntimeInfo: model_cfg.dtype is unresolved ('auto' or None) — "
            "ModelConfig __post_init__ may not have completed. Cannot emit "
            "a canonical compute_precision."
        )

    name = _dtype_name(dtype)
    if name in _VERIFIER_PRECISION_VOCAB:
        return name

    raise RuntimeError(
        f"RuntimeInfo: dtype '{name}' is outside the verifier vocabulary "
        f"{sorted(_VERIFIER_PRECISION_VOCAB)}. Either set the model's "
        "compute dtype to one of {fp16, bf16, fp32, int8, fp8}, or extend "
        "the verifier vocabulary before producing proofs at this dtype."
    )


def _resolve_sha(model_id: str, revision: str | None) -> str:
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    try:
        return HfApi().model_info(model_id, revision=revision).sha
    except Exception:
        pass
    try:
        cache = pathlib.Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
        snap = next((cache / "hub").glob(
            f"models--{model_id.replace('/','--')}*/snapshots/*"))
        return snap.name
    except Exception:
        return "unknown"


def _safe_model_config_diff(model_cfg) -> Dict[str, Any]:
    """Compute a diff of model_cfg vs a default-constructed ModelConfig.

    This is risky on v0.19+: ModelConfig has `model="Qwen/Qwen3-0.6B"` as the
    field default, and __post_init__ network-fetches the HF image-processor
    config and model-arch config for that default model, which can fail or
    hang in offline / minimal images. If the default construction fails we
    return an empty diff and log a warning rather than crashing the entire
    RuntimeInfo path (which would otherwise leave compute_precision unset
    and trip the PowHasher precision-bytes encode at runtime).
    """
    try:
        default_cfg = type(model_cfg)()
    except Exception as exc:
        logger.warning(
            "RuntimeInfo: default ModelConfig construction failed (%s); "
            "model_config_diff will be empty for this proof.", exc,
        )
        return {}

    diff: Dict[str, Any] = {}
    try:
        for f in fields(model_cfg):
            name = f.name
            try:
                val = getattr(model_cfg, name)
                default_val = getattr(default_cfg, name)
            except Exception:
                continue
            if val != default_val:
                diff[name] = val
    except Exception as exc:
        logger.warning(
            "RuntimeInfo: model_config_diff iteration failed (%s); "
            "returning partial/empty diff.", exc,
        )
    return diff


# ---------- dataclass ----------------------------------------------

@dataclass(frozen=True)
class RuntimeInfo:
    max_concurrency: int
    page_size: int | None

    model_id: str
    model_sha: str
    compute_precision: str

    model_config_diff: Dict[str, Any]
    sampling_params_diff: Dict[str, Any]

    @classmethod
    def from_configs(cls, scheduler_cfg, model_cfg, block_size: int | None):
        # sampling_params_diff is best-effort across vLLM versions
        try:
            sampling_diff = model_cfg.get_diff_sampling_param()
        except Exception as exc:
            logger.warning(
                "RuntimeInfo: get_diff_sampling_param() unavailable (%s); "
                "sampling_params_diff will be empty.", exc,
            )
            sampling_diff = {}

        # compute_precision is verifier-vocabulary only; raises on unknown.
        precision = _detect_precision(model_cfg)

        # Quantization flavour (if any) goes into the diff for diagnostics —
        # NOT into compute_precision, which the verifier hashes and uses to
        # select replay dtype. Keeping this separate lets PoW work for FP8 /
        # AWQ / GPTQ models today and lets a future verifier upgrade pick
        # the flavour up from the diff without changing the hash contract.
        config_diff = _safe_model_config_diff(model_cfg)
        flavour = _quant_flavour(model_cfg)
        if flavour:
            config_diff.setdefault("quantization_flavour", flavour)

        # model_id MUST be the HF repo id ("Qwen/Qwen3-8B"), NOT the local
        # snapshot path. Loading offline (HF_HUB_OFFLINE=1, model pre-seeded
        # under /models/hub on the egress-locked cGPU worker) makes vLLM
        # redirect model_cfg.model to the on-disk snapshot path
        # (ModelConfig.__post_init__: maybe_model_redirect). That path then
        # gets stamped into every PoW proof's model_identifier, and the
        # miner-proxy ProofCollector — matching it against the chain-
        # registered "<name>@<commit>" — DROPS ~89% of proofs as a stale-
        # model mismatch, silently killing mining. served_model_name is
        # captured BEFORE the redirect (same __post_init__), preserving the
        # repo id; fall back to model only if unset.
        _served = getattr(model_cfg, "served_model_name", None)
        if isinstance(_served, (list, tuple)):
            _served = _served[0] if _served else None
        _model_id = _served or model_cfg.model

        return cls(
            max_concurrency      = scheduler_cfg.max_num_seqs,
            page_size            = block_size,
            model_id             = _model_id,
            model_sha            = _resolve_sha(model_cfg.model, getattr(model_cfg, "revision", None)),
            compute_precision    = precision,
            model_config_diff    = config_diff,
            sampling_params_diff = sampling_diff,
        )
