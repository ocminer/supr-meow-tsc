# SPDX-License-Identifier: Apache-2.0
# Uint256 arithmetic 
def set_compact(nCompact: int):
    """
    Convert a compact nBits value (uint32) into a full 256-bit target integer,
    along with negative and overflow flags, exactly as in C++ arith_uint256::SetCompact.
    """
    # Extract exponent (size) and mantissa (nWord)
    exponent = nCompact >> 24
    mantissa = nCompact & 0x007fffff
    negative = bool(nCompact & 0x00800000)
    
    # Reconstruct the full target
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        target = mantissa << (8 * (exponent - 3))
    
    # Detect overflow (mantissa != 0 and target wouldn't fit in 256 bits)
    overflow = False
    if mantissa != 0:
        # exponent > 34 means shifting by > 248 bits → overflow 256-bit range
        if exponent > 34:
            overflow = True
        else:
            # C++ also checks if mantissa is too large to fit after shift
            # i.e., nWord > 0xff << (8*(exponent-3))
            if exponent > 3 and mantissa > (0xff << (8 * (exponent - 3))):
                overflow = True

    return target, negative, overflow

def get_compact(target: int, negative: bool = False):
    """
    Convert a full 256-bit target integer into compact nBits (uint32),
    optionally preserving a negative flag, exactly as in C++ arith_uint256::GetCompact.
    """
    if isinstance(target, (bytes, bytearray)):
        target = int.from_bytes(target, byteorder="big")
            
    # Determine byte size of target
    size = (target.bit_length() + 7) // 8
    
    # Shift target to fit into 3-byte mantissa
    if size <= 3:
        mantissa = target << (8 * (3 - size))
    else:
        mantissa = target >> (8 * (size - 3))
    
    # If the sign bit (0x00800000) is set, shift mantissa down and bump exponent
    if mantissa & 0x00800000:
        mantissa >>= 8
        size += 1
    
    # Compose compact: 1-byte exponent, 3-byte mantissa
    compact = (size << 24) | (mantissa & 0x007fffff)
    if negative and mantissa != 0:
        compact |= 0x00800000
    
    return compact

def adjust_nbits_by_multiplier(nbits: int, multiplier: int, divider: int):
    # 1. Unpack correctly
    target, negative, overflow = set_compact(nbits)

    # 2. Scale target
    new_target = (target * multiplier) // divider

    # 3. (Optionally) detect if new_target overflows 256 bits
    new_overflow = new_target.bit_length() > 256

    # 4. Re-pack into "compact"
    new_nbits = get_compact(new_target, negative)

    # 5. Make sure that the loss of resolution in compact represenatation does not fuck up target accuracy downstream
    new_target, negative, overflow = set_compact(new_nbits)

    # 6. Return everything
    return {
      "nbits": new_nbits,
      "target_bytes": new_target.to_bytes(32, byteorder="big"),
      "overflow": new_overflow,
      "negative": negative
    }


def derive_adjusted_block_target(base_nbits: int, difficulty: int, normalizer: int,
                                 pow_limit_nbits: int):
    """Model-adjusted BLOCK target/bits in the CONSENSUS direction, SATURATED
    at the chain powLimit.

        adjusted_target = min(powLimit, floor(base_target * normalizer / difficulty))

    ``difficulty`` (D) is the INVERSE-compute model difficulty: a bigger /
    more expensive model has a SMALLER D and therefore a LARGER (easier)
    target. bcore consensus (ContextualCheckBlock, validation.cpp) rejects an
    nAdjBits easier than ``base * N / D`` with ``bad-adjusted-bits``; the node
    block assembler (node/miner.cpp) and the broker/worker share path
    (mining_protocol.derive_adjusted_share_target) use the identical N / D
    direction. This is the RECIPROCAL of a naive difficulty multiply
    (``base * D / N``) — the dedicated helper exists so the direction is
    explicit and cannot be re-inverted.

    ``pow_limit_nbits`` (compact) is the chain powLimit (== consensus.powLimit,
    uniform 0x207fffff across the tensor chains). It is REQUIRED: bcore
    saturates the adjusted target at powLimit (miner.cpp / validation.cpp) and
    DeriveTarget rejects any compact target above powLimit, so the worker must
    cap identically. Without it, a legitimate larger-model miner (D < N) on a
    near-powLimit chain (tensor-test/regtest, where base == powLimit) emits an
    above-powLimit target that consensus rejects, and ``base * N / D`` can also
    exceed 256 bits (raising OverflowError before a block is ever produced).

    Returns the same dict shape as adjust_nbits_by_multiplier (``nbits``,
    ``target_bytes``, ``overflow``, ``negative``).
    """
    if difficulty is None or int(difficulty) <= 0:
        raise ValueError(
            f"derive_adjusted_block_target: difficulty (D) must be > 0; got {difficulty!r}")
    if normalizer is None or int(normalizer) <= 0:
        raise ValueError(
            f"derive_adjusted_block_target: normalizer (N) must be > 0; got {normalizer!r}. "
            "Source from the chain's ModelDifficultyNormalizer.")
    if pow_limit_nbits is None or int(pow_limit_nbits) <= 0:
        raise ValueError(
            f"derive_adjusted_block_target: pow_limit_nbits must be > 0; got {pow_limit_nbits!r}. "
            "Source from the chain's powLimit (consensus.powLimit).")

    base_target, negative, _ = set_compact(int(base_nbits))
    pow_limit_target, _, _ = set_compact(int(pow_limit_nbits))

    adjusted = (base_target * int(normalizer)) // int(difficulty)
    # Saturate at powLimit BEFORE re-packing: mirrors bcore's cap and prevents
    # both an above-powLimit target (consensus reject) and a >256-bit overflow.
    if adjusted > pow_limit_target:
        adjusted = pow_limit_target

    new_nbits = get_compact(adjusted, negative)
    # Re-decode so the returned target reflects compact quantization exactly
    # (get_compact truncates the mantissa, i.e. rounds the target DOWN, so the
    # result can only stay at or below powLimit — never above).
    new_target, negative, overflow = set_compact(new_nbits)
    return {
        "nbits": new_nbits,
        "target_bytes": new_target.to_bytes(32, byteorder="big"),
        "overflow": overflow,
        "negative": negative,
    }