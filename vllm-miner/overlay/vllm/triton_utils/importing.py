# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import types
from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)

# PoW fork note: we previously force-disabled HAS_TRITON to keep the
# top-k dispatch on the deterministic pytorch sort path. That was overkill
# — PoW already uses its private _pow_apply_top_k_top_p helper, so the
# Triton dispatch in apply_top_k_top_p only affects NON-PoW requests in
# mixed batches. Meanwhile, v0.19's block_table.py:150 calls
# `_compute_slot_mapping_kernel[(num_reqs + 1,)](...)` which requires the
# real Triton launch syntax (`function[grid](args)`) — our stub
# TritonPlaceholder cannot support that and raises
# `TypeError: 'function' object is not subscriptable`. So we keep the
# upstream auto-detection.
HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)
if HAS_TRITON:
    try:
        from triton.backends import backends

        # It's generally expected that x.driver exists and has
        # an is_active method.
        # The `x.driver and` check adds a small layer of safety.
        active_drivers = [
            x.driver for x in backends.values() if x.driver and x.driver.is_active()
        ]

        # Check if we're in a distributed environment where CUDA_VISIBLE_DEVICES
        # might be temporarily empty (e.g., Ray sets it to "" during actor init)
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        is_distributed_env = (
            cuda_visible_devices is not None and len(cuda_visible_devices.strip()) == 0
        )

        # Apply lenient driver check for distributed environments
        if is_distributed_env and len(active_drivers) == 0:
            # Allow 0 drivers in distributed environments - they may become
            # active later when CUDA context is properly initialized
            logger.debug(
                "Triton found 0 active drivers in distributed environment. "
                "This is expected during initialization."
            )
        elif not is_distributed_env and len(active_drivers) != 1:
            # Strict check for non-distributed environments
            logger.info(
                "Triton is installed but %d active driver(s) found "
                "(expected 1). Disabling Triton to prevent runtime errors.",
                len(active_drivers),
            )
            HAS_TRITON = False
    except ImportError:
        # This can occur if Triton is partially installed or triton.backends
        # is missing.
        logger.warning(
            "Triton is installed, but `triton.backends` could not be imported. "
            "Disabling Triton."
        )
        HAS_TRITON = False
    except Exception as e:
        # Catch any other unexpected errors during the check.
        logger.warning(
            "An unexpected error occurred while checking Triton active drivers:"
            " %s. Disabling Triton.",
            e,
        )
        HAS_TRITON = False

if not HAS_TRITON:
    logger.info(
        "Triton not installed or not compatible; certain GPU-related"
        " functions will not be available."
    )


class TritonPlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton")
        self.__version__ = "3.4.0"
        self.jit = self._dummy_decorator("jit")
        self.autotune = self._dummy_decorator("autotune")
        self.heuristics = self._dummy_decorator("heuristics")
        self.Config = self._dummy_decorator("Config")
        self.language = TritonLanguagePlaceholder()

    def _dummy_decorator(self, name):
        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

        return decorator


class TritonLanguagePlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton.language")
        self.constexpr = None
        self.dtype = None
        self.int64 = None
        self.int32 = None
        self.tensor = None
        self.exp = None
        self.log = None
        self.log2 = None
