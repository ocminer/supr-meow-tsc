"""V3 prompt-binding / admission helpers (TIP-0003).

Pure-Python, torch-free reference implementation of every v3 primitive so the
same module serves miners, the Python verifier, test-vector generation and the
C++ equivalence tests. The torch fast paths (batched sampling in pow_utils.py)
must produce byte-identical messages to `build_step_message` here.

Scope (TIP-0003):
  §3  extra_flags carrier   — merge_extra_flags_v3 / extract_admission_nonce
  §4  conservative B_cred   — mass_q63_for_step / credit_units_for_step /
                              b_cred_units_from_bounds  (R=1024 table)
  §5  tier rule             — tier_for_b_cred_units
  §6  Argon2id admission    — admission_message / argon2id_digest /
                              admission_expected_tries / admission_target /
                              admission_valid
  §7  v3 step hashing       — build_step_message(admission_nonce=...) /
                              step_u_from_message

Consensus-determinism notes (agreed deviations from the plan's prose, same
semantics, exactly computable in every language):
  * B_cred is accumulated in integer credit units via a checked-in R=1024 Q63
    threshold table. Runtime tiering uses no log2/libm path; endpoint rounding
    and threshold rounding are conservative, so credited B never exceeds true B.
  * expected_tries is derived with integer-only arithmetic from integer chain
    constants (ELIG_ALPHA as a rational, reference times in microseconds) and
    admission_target = (2**256 - 1) // expected_tries, representable in
    uint256 for expected_tries == 1 (the plan's floor(2**256/tries) is not).
"""

import json
import math
import struct
from hashlib import sha256

# --------------------------------------------------------------------------- #
# Constants (chain constants marked CALIBRATION are placeholders pending the
# §12 open decisions; they must come from consensus chain params at activation)
# --------------------------------------------------------------------------- #

V3_PROOF_VERSION = 3

POW_WINDOW_SIZE = 256                    # mirrors pow_utils.POW_WINDOW_SIZE

ADMISSION_NONCE_BYTES = 32
ADMISSION_NONCE_HEX_LEN = 64             # exactly 64 lowercase hex chars

# B_cred is scored in integer CREDIT UNITS (§4): R units == 1 bit. The credit
# is a deterministic table lookup (bcred_table_r1024), NOT runtime -log2, so it
# is byte-identical in every language and free of float non-associativity.
# These scalars are consensus constants (hardcoded); the ~256 KiB threshold
# array is loaded LAZILY (see _thresholds_q63) so the many miner/worker images
# that vendor pow_v3.py only for the carrier/sampler helpers never need the
# table file — only tier-scoring sites (verifier, functional tests) do.
BCRED_R = 1024                                # units per bit
BCRED_N_MAX = 32 * BCRED_R                    # 32 * R  (per-step credit cap)
BCRED_Q_ONE = 1 << 63                         # Q63 unit == mass 1.0
# SHA-256 of the checked-in little-endian table (bcred_table_r1024.bin); the
# lazy loader asserts the imported module reports the same identity.
BCRED_TABLE_SHA256 = \
    "05929aee8475b4c2a4ceafd054e34f11521101028f93a2d5583e15f0285ff8ea"

# Tier thresholds in credit units. Chain params carry these as BITS
# (V3BFloorBits / V3BFreeBits); the tier comparison multiplies by R.
B_FLOOR_BITS = 45                        # CALIBRATION (initial floor)
B_FREE_BITS = 70                         # CALIBRATION (initial high tier)
B_FLOOR_UNITS = B_FLOOR_BITS * BCRED_R
B_FREE_UNITS = B_FREE_BITS * BCRED_R

# Per-step credit cap == the table's max index (32 bits worth of units). The
# 2*ATOL widening floors mass_q well above this in practice (~12 bits), so the
# cap is purely defensive.
B_STEP_MAX_UNITS = BCRED_N_MAX

# Interval-mass tolerance — must equal the verifier's ATOL
# (services/verification-api/src/config/constants.py). ATOL_Q63_CEIL is the
# EXACT integer ceil(ATOL * 2^63); the mass widening adds 2*ATOL_Q63_CEIL. It
# is a checked-in constant (identical in pow_v3.h) so no float feeds the Q63
# arithmetic: ceil(0.0001_f64 * 2^63) == 922337203685478.
ATOL = 0.0001
ATOL_Q63_CEIL = 922337203685478

# Consensus parser bounds for the v3 extra_flags carrier (§3) — identical in
# Python and C++; violations mean "no nonce claimed", never a parse crash.
EXTRA_FLAGS_MAX_BYTES = 4096             # CALIBRATION
EXTRA_FLAGS_MAX_DEPTH = 8                # CALIBRATION

# ELIG_ALPHA = 0.04 as an exact rational (§1).
ELIG_ALPHA_NUM = 4
ELIG_ALPHA_DEN = 100

# ARGON_PROFILE (§1): Argon2id variant, memory, iterations, lanes, output len.
ARGON2_TIME_COST = 1                     # CALIBRATION
ARGON2_MEMORY_KIB = 8192                 # CALIBRATION (8 MiB)
ARGON2_LANES = 1                         # CALIBRATION
ARGON2_HASH_LEN = 32
# Fixed public salt: pure domain-separation constant (the puzzle's entropy is
# in the message; Argon2 requires a salt >= 8 bytes). 16 bytes, never changes.
ARGON2_SALT = b"TC_V3_ADMISSION!"

# Reference timings in integer microseconds (§1) — integer so the target
# derivation is exact in every language.
ARGON_REF_US = 4_000                     # CALIBRATION: Argon2id(8MiB,t=1,l=1) eval,
                                         # measured 4010us on EPYC 7R13 (2026-07-07)
DECODE_US_AT_NORMALIZER = 6_000_000      # CALIBRATION: one 256-window standalone
                                         # decode at difficulty == ModelDifficultyNormalizer;
                                         # measured B=1 5.97s, Qwen3-8B bf16 on L40S
                                         # → expected_tries = 60 at normalizer

MODEL_DIFFICULTY_NORMALIZER = 1_000_000  # consensus/params.h default

UINT256_MAX = (1 << 256) - 1

# v3.0 sampler profile (§1, §2) — CONSENSUS-FIXED, not miner- or model-chosen.
# Enforcement is exact equality against the proof's sampler fields; any
# missing or divergent field makes a v3 proof invalid. Per-model profiles in
# model `extra` are ignored/rejected for v3.0 (`extra` is not in the model
# hash, so a sampler policy there would be unbound).
SAMPLER_PROFILE_V3 = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

# Tier decisions (§5)
TIER_INVALID = "invalid"                 # B_cred < B_FLOOR
TIER_ADMISSION = "admission_required"    # B_FLOOR <= B_cred < B_FREE
TIER_FREE = "free"                       # B_cred >= B_FREE


# --------------------------------------------------------------------------- #
# §3 — extra_flags carrier
# --------------------------------------------------------------------------- #

def canonical_json(obj) -> str:
    """Miner-side canonical JSON (a producer convention, NOT a consensus rule)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def is_valid_admission_nonce_hex(value) -> bool:
    """Exactly 64 lowercase hex chars (consensus shape rule)."""
    if not isinstance(value, str) or len(value) != ADMISSION_NONCE_HEX_LEN:
        return False
    return all(c in "0123456789abcdef" for c in value)


def merge_extra_flags_v3(extra_flags, admission_nonce_hex: str) -> str:
    """Merge the admission nonce under the top-level "v3" key without dropping
    existing content (completion_id, proof_purpose, replay dtype, ...).

    `extra_flags` may be a dict, a JSON string, a non-JSON string (legacy
    pformat blob or empty) or None. Non-JSON string content is preserved under
    the "_diff" key exactly like ProofWriter.write_proof's completion-id merge.
    Output is canonical JSON (producer convention).
    """
    if not is_valid_admission_nonce_hex(admission_nonce_hex):
        raise ValueError(
            "admission_nonce_hex must be exactly 64 lowercase hex chars")

    base = {}
    if isinstance(extra_flags, dict):
        base = dict(extra_flags)
    elif isinstance(extra_flags, str) and extra_flags.strip():
        try:
            parsed = json.loads(extra_flags)
            base = parsed if isinstance(parsed, dict) else {"_diff": extra_flags}
        except json.JSONDecodeError:
            base = {"_diff": extra_flags}
    elif extra_flags is not None and not isinstance(extra_flags, str):
        raise TypeError(f"extra_flags must be dict/str/None, got {type(extra_flags)}")

    v3 = base.get("v3")
    v3 = dict(v3) if isinstance(v3, dict) else {}
    v3["admission_nonce"] = admission_nonce_hex
    base["v3"] = v3
    return canonical_json(base)


class _DuplicateKeyError(ValueError):
    pass


def _pairs_reject_duplicates(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise _DuplicateKeyError(k)
        d[k] = v
    return d


def _depth_of(obj, _depth=1):
    if _depth > EXTRA_FLAGS_MAX_DEPTH:
        return _depth
    if isinstance(obj, dict):
        return max([_depth] + [_depth_of(v, _depth + 1) for v in obj.values()])
    if isinstance(obj, list):
        return max([_depth] + [_depth_of(v, _depth + 1) for v in obj])
    return _depth


def extract_admission_nonce(extra_flags) -> "bytes | None":
    """Consensus extraction + shape rule (§3 parser bounds). Returns the 32
    raw nonce bytes, or None when no admission is claimed. NEVER raises.

    A nonce is claimed only by the exact shape
    `{"v3":{"admission_nonce":"<64 lowercase hex>"}, ...}`. ANY violation —
    empty/oversized input (> EXTRA_FLAGS_MAX_BYTES), invalid UTF-8,
    unparseable JSON, duplicate object keys, nesting deeper than
    EXTRA_FLAGS_MAX_DEPTH, non-object top level, "v3" not an object, key
    absent, or a nonce value that is not a string of exactly 64 lowercase hex
    chars — means NO nonce claimed: the free tier accepts, below B_FREE the
    tier rule rejects. Consensus does NOT require canonical JSON.
    """
    if isinstance(extra_flags, (bytes, bytearray)):
        try:
            extra_flags = bytes(extra_flags).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
    if not isinstance(extra_flags, str) or not extra_flags.strip():
        return None
    if len(extra_flags.encode("utf-8", errors="surrogatepass")) > EXTRA_FLAGS_MAX_BYTES:
        return None
    try:
        parsed = json.loads(extra_flags,
                            object_pairs_hook=_pairs_reject_duplicates)
    except (ValueError, RecursionError):
        # includes JSONDecodeError and _DuplicateKeyError
        return None
    if not isinstance(parsed, dict) or _depth_of(parsed) > EXTRA_FLAGS_MAX_DEPTH:
        return None
    v3 = parsed.get("v3")
    if not isinstance(v3, dict):
        return None
    value = v3.get("admission_nonce")
    if not is_valid_admission_nonce_hex(value):
        return None
    return bytes.fromhex(value)


# --------------------------------------------------------------------------- #
# §7 — v3 step message (pure-bytes reference of pow_utils._build_msg)
# --------------------------------------------------------------------------- #

def _u16le(n: int) -> bytes:
    return struct.pack("<H", n)


def _u32le(n: int) -> bytes:
    return struct.pack("<I", n & 0xFFFFFFFF)


def build_step_message(header_prefix: bytes,
                       vdf: bytes,
                       tick: int,
                       step: int,
                       context_tokens,
                       precision: str,
                       admission_nonce: "bytes | None" = None,
                       window_size: int = POW_WINDOW_SIZE) -> bytes:
    """Byte-exact reference of the sampler preimage
        header_prefix | vdf | u32le(tick) | u32le(step) | ctx_window | precision
    with the v3 admission nonce appended when present (§7).

    ctx_window: the LAST `window_size` tokens of context_tokens, left-padded
    with zeros to exactly `window_size` entries, 8 bytes little-endian per
    token — identical to pow_utils._tok_le_bytes on the rolling window and to
    QuickVerifier::ComputeUValue (quick_verifier.cpp:406).
    """
    if admission_nonce is not None and len(admission_nonce) != ADMISSION_NONCE_BYTES:
        raise ValueError("admission_nonce must be exactly 32 bytes")

    window = [0] * window_size
    ctx = list(context_tokens)[-window_size:]
    if ctx:
        window[window_size - len(ctx):] = ctx

    parts = [
        bytes(header_prefix),
        bytes(vdf),
        _u32le(int(tick)),
        _u32le(int(step)),
        b"".join(struct.pack("<q", int(t)) for t in window),
        precision.encode("utf-8"),
    ]
    if admission_nonce is not None:
        parts.append(bytes(admission_nonce))
    return b"".join(parts)


def step_u_from_message(message: bytes):
    """SHA-256 the step message; return (u in [0,1), digest bytes).

    u = little-endian uint32 of the first 4 digest bytes / 2^32 — identical to
    pow_utils._digest_to_u and QuickVerifier::DigestToU.
    """
    digest = sha256(message).digest()
    u = int.from_bytes(digest[:4], "little") / 4294967296.0
    return u, digest


# --------------------------------------------------------------------------- #
# §6 — Argon2id admission puzzle
# --------------------------------------------------------------------------- #

# Domain tag for the full-prefix commitment bound into admission (§6).
PROMPT_CTX_TAG = b"TC_V3_PROMPT_CTX"
PROMPT_COMMITMENT_BYTES = 32


def prompt_commitment(prompt_tokens, pad_mask=None) -> bytes:
    """SHA256(tag | u32le(n_tokens) | prompt_tokens_i64le
    | u32le(n_mask) | pad_mask_u8) — commits to the FULL model-visible prefix
    of the window (§6).

    `prompt_tokens` is the proof's existing prompt_tokens field, which for
    windows after the first already includes the previously generated tokens
    (the miner archives everything before the window). Token layout is the
    same 8-byte little-endian encoding the sampler preimage uses
    (_tok_le_bytes); pad_mask is one byte per bool in proof order (it affects
    attention masking, so it is part of the model-visible state). The
    commitment (not the raw prefix) enters the Argon2id message so the prefix
    is hashed once per window, not once per grind attempt, and the admission
    preimage stays fixed-size.

    For v3, pad_mask is canonical shape: omitted means all-false for the
    prompt; otherwise it must have exactly one bit per prompt token. The length
    prefixes make the token/mask boundary unambiguous even for adversarial
    malformed proofs.
    """
    tokens = [int(t) for t in prompt_tokens]
    if len(tokens) > 0xFFFFFFFF:
        raise ValueError("prompt_tokens too long for u32le length prefix")
    if pad_mask is None:
        mask = [False] * len(tokens)
    else:
        mask = [bool(b) for b in pad_mask]
    if len(mask) != len(tokens):
        raise ValueError("pad_mask length must equal prompt_tokens length")
    if len(mask) > 0xFFFFFFFF:
        raise ValueError("pad_mask too long for u32le length prefix")
    parts = [PROMPT_CTX_TAG,
             _u32le(len(tokens)),
             b"".join(struct.pack("<q", t) for t in tokens),
             _u32le(len(mask)),
             bytes(1 if b else 0 for b in mask)]
    return sha256(b"".join(parts)).digest()


def admission_message(window_first_step_message: bytes,
                      model_identifier: "str | bytes",
                      admission_nonce: bytes,
                      prompt_commitment_digest: bytes) -> bytes:
    """msg_w | prompt_commitment | u16le(len(model_identifier))
    | model_identifier | nonce (§6).

    `window_first_step_message` is build_step_message(...) at the window's
    FIRST step WITHOUT the nonce appended (the nonce enters here explicitly).
    `prompt_commitment_digest` (fixed 32 bytes, from prompt_commitment())
    binds the FULL model-visible prefix: msg_w's rolling 256-window alone
    would let a miner vary out-of-window prefix tokens — which the model DOES
    condition on — and amortize one admission across different decode paths.
    model_identifier is length-prefixed because it is the only variable-length
    field between two fixed-layout regions.
    """
    if len(admission_nonce) != ADMISSION_NONCE_BYTES:
        raise ValueError("admission_nonce must be exactly 32 bytes")
    if len(prompt_commitment_digest) != PROMPT_COMMITMENT_BYTES:
        raise ValueError("prompt_commitment must be exactly 32 bytes")
    mid = (model_identifier.encode("utf-8")
           if isinstance(model_identifier, str) else bytes(model_identifier))
    if len(mid) > 0xFFFF:
        raise ValueError("model_identifier too long for u16le length prefix")
    return b"".join([bytes(window_first_step_message),
                     bytes(prompt_commitment_digest),
                     _u16le(len(mid)), mid, bytes(admission_nonce)])


def argon2id_digest(message: bytes,
                    time_cost: int = ARGON2_TIME_COST,
                    memory_kib: int = ARGON2_MEMORY_KIB,
                    lanes: int = ARGON2_LANES,
                    hash_len: int = ARGON2_HASH_LEN,
                    salt: bytes = ARGON2_SALT) -> bytes:
    """Raw Argon2id digest of the admission message (§6).

    Lazy import: argon2-cffi is only required on paths that actually grind or
    verify admission (miners targeting the low tier; verifiers on v3 proofs).
    """
    from argon2.low_level import Type, hash_secret_raw
    return hash_secret_raw(secret=bytes(message), salt=salt,
                           time_cost=time_cost, memory_cost=memory_kib,
                           parallelism=lanes, hash_len=hash_len, type=Type.ID)


def admission_expected_tries(difficulty: int,
                             normalizer: int = MODEL_DIFFICULTY_NORMALIZER,
                             decode_us_at_normalizer: int = DECODE_US_AT_NORMALIZER,
                             elig_alpha_num: int = ELIG_ALPHA_NUM,
                             elig_alpha_den: int = ELIG_ALPHA_DEN,
                             argon_ref_us: int = ARGON_REF_US) -> int:
    """Integer-exact §6 derivation.

        decode_us       = decode_us_at_normalizer * normalizer / difficulty
        expected_tries  = ELIG_ALPHA * decode_us / argon_ref_us
                        = (alpha_num * decode_us_at_normalizer * normalizer)
                          // (alpha_den * argon_ref_us * difficulty)

    floored to an integer, clamped to >= 1. Registered `difficulty` is an
    INVERSE compute scalar: more FLOPs => LOWER difficulty => more tries.
    """
    difficulty = int(difficulty)
    if difficulty <= 0:
        raise ValueError("difficulty must be positive")
    numerator = elig_alpha_num * decode_us_at_normalizer * int(normalizer)
    denominator = elig_alpha_den * argon_ref_us * difficulty
    if denominator <= 0:
        raise ValueError("invalid admission constants")
    return max(1, numerator // denominator)


def admission_target(difficulty: int, **kwargs) -> int:
    """admission_target = (2^256 - 1) // expected_tries (uint256-representable
    form of the plan's floor(2^256 / expected_tries); differs by at most 1)."""
    return UINT256_MAX // admission_expected_tries(difficulty, **kwargs)


def admission_valid(digest: bytes, target: int) -> bool:
    """uint256_le(digest) < target (§6): digest read as LITTLE-endian uint256."""
    if len(digest) != ARGON2_HASH_LEN:
        raise ValueError("admission digest must be 32 bytes")
    return int.from_bytes(digest, "little") < int(target)


def build_admission_preimage(*, header_prefix: bytes, vdf: bytes, tick: int,
                             step: int, context_tokens, prefix_tokens,
                             prefix_pad_mask, precision: str, difficulty: int,
                             model_identifier: str,
                             max_tries_factor: int = 1):
    """SINGLE SOURCE for the v3 admission grind inputs (§6/§9). Returns the exact
    5-tuple the native grind consumes:

        (msg_w, model_identifier, target_le, max_tries, commitment)

    Both the miner sampler (common_sampler_helper) and the scheduler
    (AdmissionScheduler) MUST build the grind preimage through this one function
    so it is byte-identical on both sides — a nonce ground here validates when the
    sampler hashes the same msg_w at sample time. Every model/precision/difficulty
    input is EXPLICIT: no hidden process-global (`model_identifier` and `precision`
    come from the writer on vLLM and from the pow payload on llama.cpp, so the
    caller must pass them in).

    - msg_w binds this window's first-step preimage (header|vdf|tick|step|
      ctx_window|precision, NO nonce): `context_tokens` is the rolling window
      (build_step_message keeps its last `POW_WINDOW_SIZE`).
    - commitment binds the full pre-window prefix: `prefix_tokens`/`prefix_pad_mask`
      are the archive before this window (the eventual proof's prompt tokens)."""
    difficulty = int(difficulty)
    msg_w = build_step_message(bytes(header_prefix), bytes(vdf), int(tick),
                               int(step), context_tokens, precision)
    commitment = prompt_commitment(prefix_tokens, prefix_pad_mask)
    target_le = admission_target(difficulty).to_bytes(32, "little")
    max_tries = int(admission_expected_tries(difficulty) * int(max_tries_factor))
    return bytes(msg_w), str(model_identifier), target_le, max_tries, bytes(commitment)


# --------------------------------------------------------------------------- #
# §4 — numerically conservative B_cred
# --------------------------------------------------------------------------- #

_THRESHOLDS_Q63 = None       # lazily populated, non-increasing, len == N_MAX+1


def _thresholds_q63():
    """Load and cache the R=1024 credit-threshold table (§4). Imported lazily
    so importers that use only the carrier/sampler helpers never pull in the
    ~256 KiB array. Works whether pow_v3 is a package member (utils.pow_v3) or
    a bare top-level module (services flatten shared-utils/*.py)."""
    global _THRESHOLDS_Q63
    if _THRESHOLDS_Q63 is None:
        try:
            from . import bcred_table_r1024 as _t
        except ImportError:
            import bcred_table_r1024 as _t
        if _t.BCRED_R != BCRED_R or _t.BCRED_N_MAX != BCRED_N_MAX:
            raise RuntimeError("bcred_table_r1024: R/N_MAX mismatch")
        if _t.BCRED_TABLE_SHA256 != BCRED_TABLE_SHA256:
            raise RuntimeError("bcred_table_r1024: table SHA-256 mismatch")
        _THRESHOLDS_Q63 = _t.THRESHOLDS_Q63
    return _THRESHOLDS_Q63


def _f64_to_q63_floor(x: float) -> int:
    """floor(x * 2^63) for a finite double x in [0, 1], via EXACT integer
    arithmetic (no `double * 9.22e18`, which would round in the 52-bit
    mantissa and be FMA/platform dependent).

    frexp gives x = m * 2^e with m in [0.5, 1), so mant = m * 2^53 is an exact
    integer in [2^52, 2^53) and x * 2^63 = mant * 2^(e+10). A right shift
    truncates toward zero == floor for a non-negative value.
    """
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return BCRED_Q_ONE
    m, e = math.frexp(x)
    mant = int(math.ldexp(m, 53))
    shift = e + 10
    if shift >= 0:
        return mant << shift
    return mant >> (-shift)


def _f64_to_q63_ceil(x: float) -> int:
    """ceil(x * 2^63) for a finite double x in [0, 1], EXACT (see _floor)."""
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return BCRED_Q_ONE
    m, e = math.frexp(x)
    mant = int(math.ldexp(m, 53))
    shift = e + 10
    if shift >= 0:
        return mant << shift                       # exact, no remainder
    s = -shift
    return (mant + (1 << s) - 1) >> s               # round up


def mass_q63_for_step(lower: float, upper: float,
                      atol_q63_ceil: int = ATOL_Q63_CEIL) -> int:
    """Conservative interval mass of one step in Q63 fixed point (§4).

    Quantise the ENDPOINTS (not the width): the upper endpoint rounds UP
    (ceil), the lower endpoint rounds DOWN (floor), and the ATOL widening adds
    2 * ceil(ATOL * 2^63). Every rounding direction OVER-estimates the mass, so
    the resulting credit can never exceed the true credit. Rejects NaN/Inf and
    upper < lower (proof invalid). Result is clamped to [0, 2^63].
    """
    lower = float(lower)
    upper = float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"invalid entropy bounds: lower={lower} upper={upper}")
    if upper < lower:
        raise ValueError(f"invalid entropy bounds: upper {upper} < lower {lower}")
    hi_q = _f64_to_q63_ceil(upper)          # in [0, 2^63]
    lo_q = _f64_to_q63_floor(lower)         # in [0, 2^63], <= hi_q
    mass = (hi_q - lo_q) + 2 * atol_q63_ceil
    if mass > BCRED_Q_ONE:
        mass = BCRED_Q_ONE
    return mass


def credit_units_for_step(mass_q63: int) -> int:
    """Credit units for one step: the largest n in [0, N_MAX] with
    THRESHOLDS_Q63[n] >= mass_q63 (§4). Because the table is non-increasing
    and THRESHOLDS_Q63[0] == 2^63 >= mass_q63 (mass is clamped), n == 0 always
    qualifies, so the credit is >= 0 and <= N_MAX (the per-step cap).

    threshold_q63[n] = floor(2^63 * 2^(-n/R)) rounds DOWN while mass rounds UP,
    so `mass_q63 <= threshold[n]` implies the true mass is < 2^(-n/R): the
    credit never over-counts (never-over-credit invariant)."""
    if mass_q63 <= 0:
        raise ValueError("mass_q63 <= 0: invalid interval never earns credit")
    thresholds = _thresholds_q63()
    lo, hi = 0, BCRED_N_MAX
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if thresholds[mid] >= mass_q63:
            lo = mid
        else:
            hi = mid - 1
    return lo


def b_cred_units_from_bounds(lower_bounds, upper_bounds,
                             atol_q63_ceil: int = ATOL_Q63_CEIL) -> int:
    """Sum of per-step credit units over the window (§4).

    A SEPARATE integer accumulator (uint64 range: <= 256 * N_MAX == 8_388_608)
    — never the reuse gate's per-step-capped prefix. Integer accumulation is
    exact and independent of list / GPU reduction order. Both quick and full
    verification call this on their respective bounds. Raises ValueError on
    length mismatch or invalid bounds (proof invalid)."""
    lower_list = [float(x) for x in lower_bounds]
    upper_list = [float(x) for x in upper_bounds]
    if len(lower_list) != len(upper_list):
        raise ValueError("entropy bounds size mismatch")
    total = 0
    for lo, hi in zip(lower_list, upper_list):
        total += credit_units_for_step(mass_q63_for_step(lo, hi, atol_q63_ceil))
    return total


def b_cred_bits(b_cred_units: int) -> float:
    """Float view of a credit-unit B_cred, for logs/metrics only — never for
    the tier comparison (compare in integer units)."""
    return b_cred_units / BCRED_R


# --------------------------------------------------------------------------- #
# §5 — tier rule
# --------------------------------------------------------------------------- #

def tier_for_b_cred_units(b_cred_units: int,
                          b_floor_units: int = B_FLOOR_UNITS,
                          b_free_units: int = B_FREE_UNITS) -> str:
    """B_cred < B_FLOOR -> invalid; < B_FREE -> admission required; else free.

    All comparisons are in integer credit units (R units == 1 bit).

    Callers must additionally enforce: a PRESENT admission nonce is verified
    regardless of tier (present => valid), and absent + admission_required =>
    invalid (§5).

    Trust boundary (§4): B_cred is computed from the proof's SUBMITTED top-k
    evidence; its soundness is conditional on full-replay / red-block
    enforcement being active for v3 proofs. v3 MUST NOT activate without it.
    """
    if b_cred_units < b_floor_units:
        return TIER_INVALID
    if b_cred_units < b_free_units:
        return TIER_ADMISSION
    return TIER_FREE
