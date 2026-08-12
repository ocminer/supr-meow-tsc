# SPDX-License-Identifier: Apache-2.0
# common_sampler.py
import time, torch, struct, hashlib
import os
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor

# One boundary row's fully-prepared admission grind (pure inputs — no GPU/ring
# access), so the expensive Argon2 grind can run off the engine thread.
_AdmissionJob = namedtuple(
    "_AdmissionJob",
    "row seq_id difficulty msg_w model_id target_le max_tries commitment")
from vllm.sampling.pow_utils import PowState, sha256_many, _tok_le_bytes, _u32le, _str_bytes, _build_msg, _digest_to_u, POW_WINDOW_SIZE, check_hash_against_target, SequenceCache, Logger,RowManager,RingBuffers,PowHasher,ProofWriter, hex_to_bytes_tensor
from vllm.sampling.zmq_pow_writer import MiningResponseSubmitter
from vllm.sampling.uint256_arithmetics import get_compact
# V3 prompt-binding / admission helpers (TIP-0003) — deployed next
# to pow_utils.py in every image.
try:
    from vllm.sampling import pow_v3
except ImportError:
    import pow_v3

# tokens.min()>=0 is a per-step host sync (bool(tensor)); it's a validation guard,
# not a correctness requirement, so it's off by default and gated behind a flag.
_POW_DEBUG_ASSERTS = os.environ.get('POW_DEBUG_ASSERTS', '0') in ('1', 'true', 'True')

class CommonSamplerHelper:
    def __init__(self, owner, proxy_audit_enabled=None):
        # owner is the sampler instance (so we can reach window_size, device, logger, etc.)
        self.s = owner
        # Allow override of proxy audit setting, otherwise check environment
        if proxy_audit_enabled is not None:
            self.proxy_audit_enabled = proxy_audit_enabled
        else:
            self.proxy_audit_enabled = os.environ.get('POW_PROXY_ENABLE', '0') in ('1', 'true', 'True')
        
        # Choose processor based on environment
        processor_mode = os.environ.get('POW_PROCESSOR_MODE', 'python')
        self.use_cpp_processor = False
        self.proof_processor = None

        if processor_mode == 'cpp':
            # cpp was EXPLICITLY requested. A silent fallback to the Python
            # proof assembler here is a correctness trap: an operator who set
            # POW_PROCESSOR_MODE=cpp would believe they shipped the C++ path
            # while actually running Python proof assembly (e.g. the 27B
            # audit path, which must match the mining fleet). Fail hard so a
            # broken/absent proof_processor.so surfaces at startup instead of
            # degrading silently. Opt out with POW_PROCESSOR_FALLBACK=1 only
            # for local dev where the .so isn't built.
            allow_fallback = os.environ.get('POW_PROCESSOR_FALLBACK', '0') in ('1', 'true', 'True')
            try:
                import proof_processor
                self.proof_processor = proof_processor.ProofProcessor()
                self.use_cpp_processor = True
                print("Using C++ ProofProcessor for proof processing")
                # Note: model metadata is set later in sampler.py after runtime_info is available
            except ImportError as e:
                if not allow_fallback:
                    raise ImportError(
                        "POW_PROCESSOR_MODE=cpp was requested but the C++ "
                        f"proof_processor extension failed to import ({e}). "
                        "Refusing to silently fall back to Python proof "
                        "assembly. Build proof_processor.so, or set "
                        "POW_PROCESSOR_FALLBACK=1 to permit the Python path "
                        "(local dev only)."
                    ) from e
                print(f"WARNING: C++ proof_processor unavailable ({e}); "
                      "POW_PROCESSOR_FALLBACK set, using Python proof assembly")
                self.use_cpp_processor = False
        else:
            self.use_cpp_processor = False

        # ---- V3 admission grind scheduling (TIP-0003) ----
        # POW_V3_ADMISSION_MODE:
        #   off (default) — never grind; emit v3 INERT (no admission nonce
        #           folded into u / the final hash). This is the inert-rollout
        #           default and MUST match the standalone scheduler default
        #           (admission_scheduler.py): a v3 proof is safe to emit before
        #           activation only when nonce-less, because bcore judges a
        #           version>=3 proof under v2/reuse rules until height >=
        #           V3ActivationHeight (validation.cpp:4099-4108). Folding a
        #           nonce early diverges u from consensus -> self-punishing
        #           block rejection. NOTE: the activation gate is height-driven
        #           in bcore, but the MINER's pow.difficulty is the positive
        #           registry value, so 'always' here would grind regardless of
        #           height — hence off is the only safe default. Operators opt
        #           in to 'always' for admission-band mining AT/AFTER activation.
        #   always — speculatively grind admission for EVERY window at its
        #           boundary (pay ~ELIG_ALPHA of decode even on windows that
        #           turn out free). Only takes effect for proof version >= 3.
        #           v3 windows landing in [B_FLOOR, B_FREE) get a nonce instead
        #           of being forfeited as nonce-less mining.
        # The tier is only known post-decode, so these are the two honest
        # strategies (§6 overhead note). Grinding runs in the native
        # admission_grind (C++/libargon2, GIL released) — never a Python
        # nonce loop (§9).
        self.admission_mode = os.environ.get(
            'POW_V3_ADMISSION_MODE', 'off').strip().lower()
        try:
            self.admission_max_tries_factor = max(1, int(os.environ.get(
                'POW_V3_GRIND_MAX_TRIES_FACTOR', '16')))
        except ValueError:
            self.admission_max_tries_factor = 16
        self._admission_grind_fn = None       # injectable for tests
        self._admission_grind_resolved = False
        self._admission_warned_no_grinder = False
        # Async (row-parking) admission state: row -> (_AdmissionJob, Future).
        # A boundary row grinds in the background while other rows keep sampling;
        # poll_admission() harvests finished nonces. Drives the non-blocking
        # begin_admission_for_rows/poll_admission API (vs the blocking
        # ensure_admission_for_rows). Empty unless the async path is used.
        self._adm_pending = {}

    def assert_v3_ready(self):
        """Startup self-test for v3 mining (fail fast, never at the first
        window boundary): the native Argon2 grinder must exist AND find a
        nonce against the easiest possible target. Raises RuntimeError —
        callers invoke this when POW_PROOF_VERSION >= 3 and must let it
        propagate; a v3 miner without Argon2 would silently forfeit every
        admission-band window."""
        grind_fn = self._resolve_admission_grind_fn()
        if grind_fn is None:
            raise RuntimeError(
                "POW_PROOF_VERSION >= 3 but the native admission_grind is "
                "unavailable (proof_processor.so missing or built without "
                "libargon2 / POW_V3_HAVE_ARGON2). Rebuild with libargon2 or "
                "mine v2.")
        # All-0xff little-endian target = max uint256: any digest except
        # all-0xff passes the strict <, so the first try wins.
        msg_w = pow_v3.build_step_message(
            bytes(76), bytes(32), 0, 0, [], "selftest")
        commitment = pow_v3.prompt_commitment([], [])
        try:
            nonce = grind_fn(bytes(msg_w), "v3-selftest", b"\xff" * 32, 4,
                             bytes(commitment))
        except Exception as exc:
            raise RuntimeError(
                f"v3 startup self-test: native admission_grind failed: {exc}"
            ) from exc
        if nonce is None or len(bytes(nonce)) != 32:
            raise RuntimeError(
                "v3 startup self-test: admission_grind found no nonce against "
                "the trivial target — Argon2 backend is broken.")
        self.s.logger.log("v3 startup self-test passed: native Argon2 "
                          "admission grinder operational", "INFO")

    def _resolve_admission_grind_fn(self):
        """Resolve the native grind loop once. Returns a callable
        (msg_w: bytes, model_identifier: str, target_le: bytes32,
        max_tries: int, prompt_commitment: bytes32) -> bytes32 | None,
        or None when no native module is importable — we then mine
        nonce-less rather than run a Python nonce loop (§9)."""
        if not self._admission_grind_resolved:
            self._admission_grind_resolved = True
            if self._admission_grind_fn is None:
                try:
                    import proof_processor
                    self._admission_grind_fn = getattr(
                        proof_processor, "admission_grind", None)
                except ImportError:
                    self._admission_grind_fn = None
        return self._admission_grind_fn

    def ensure_admission_for_rows(self, seq_ids, rows, steps, contexts,
                                  compute_precision):
        """Grind + store admission for every row whose NEXT sampled token is
        a window-first step (steps is already window-relative: boundary ==
        0). Must run BEFORE batch_sample_tokens hashes those steps — the
        nonce enters every u of the window (§6: admission is pre-decode).
        No-op unless POW_V3_ADMISSION_MODE=always and proof version >= 3,
        so the default hot path pays nothing.

        The grinds are independent per row and any nonce < target is valid, so
        they run CONCURRENTLY: the native admission_grind releases the GIL, so a
        thread pool grinds all boundary rows at once instead of serially (a batch
        of B fresh prompts otherwise costs B x single-grind before the first
        token). Stale-nonce clears and result writes stay on the engine thread;
        only the pure-C++ grind runs on workers."""
        if self.admission_mode != 'always':
            return
        if int(getattr(self.s.proof_writer, 'proof_version', 2) or 2) < pow_v3.V3_PROOF_VERSION:
            return
        boundary = (steps == 0).nonzero(as_tuple=False).flatten().tolist()
        if not boundary:
            return
        grind_fn = self._resolve_admission_grind_fn()
        if grind_fn is None:
            self._warn_no_grinder()
            return
        # Prepare (engine thread): build preimages + clear stale nonces.
        jobs = []
        for i in boundary:
            row = rows[i]
            if row is None:
                continue
            job = self._prepare_admission(
                seq_ids[i], row, contexts[i].tolist(), 0, compute_precision)
            if job is not None:
                jobs.append(job)
        if not jobs:
            return
        # Grind. One row runs inline (no pool/scheduling overhead); many rows
        # fan out across the pool (bounded to ~physical cores — 8 MiB Argon2 is
        # memory-bandwidth-bound past that).
        if len(jobs) == 1:
            results = [(jobs[0], self._grind(grind_fn, jobs[0]))]
        else:
            pool = self._admission_pool()
            futs = [(j, pool.submit(self._grind, grind_fn, j)) for j in jobs]
            results = [(j, f.result()) for j, f in futs]
        # Store (engine thread): ring-buffer writes are not worker-safe.
        for job, nonce in results:
            self._store_admission_result(job, nonce)

    # ----- true async admission (row parking) — non-blocking API ------------- #
    # Unlike ensure_admission_for_rows (which blocks the sampler until every
    # boundary nonce is ground), this lets a boundary row grind in the BACKGROUND
    # while the other rows keep sampling. The sampler drives it per tick:
    #   pending = begin_admission_for_rows(...)   # submit new boundaries, get not-ready set
    #   ... sample only rows NOT in `pending` ...
    #   begin_admission_for_rows already polled; use poll_admission() again if needed
    # A row's window-first token cannot be sampled until its nonce lands (the nonce
    # enters the step hash), so the CALLER must exclude `pending` rows from
    # batch_sample_tokens AND from commit this tick — that exclusion is the
    # engine-level integration (see design note; not wired here).

    def begin_admission_for_rows(self, seq_ids, rows, steps, contexts,
                                 compute_precision):
        """NON-BLOCKING. Submit a background grind for every boundary row
        (steps==0) that is not already pending or ready, harvest anything already
        finished, and return the set of rows whose nonce is NOT yet available
        (the caller must not sample those this tick). No-op set unless
        POW_V3_ADMISSION_MODE=always and proof version >= 3."""
        if self.admission_mode != 'always':
            return set()
        if int(getattr(self.s.proof_writer, 'proof_version', 2) or 2) < pow_v3.V3_PROOF_VERSION:
            return set()
        grind_fn = self._resolve_admission_grind_fn()
        if grind_fn is None:
            self._warn_no_grinder()
            return set()
        boundary = (steps == 0).nonzero(as_tuple=False).flatten().tolist()
        pool = None
        for i in boundary:
            row = rows[i]
            if row is None or row in self._adm_pending:
                continue
            job = self._prepare_admission(
                seq_ids[i], row, contexts[i].tolist(), 0, compute_precision)
            if job is None:
                continue                      # skip -> mines nonce-less, not parked
            if pool is None:
                pool = self._admission_pool()
            self._adm_pending[row] = (job, pool.submit(self._grind, grind_fn, job))
        self.poll_admission()                 # harvest any already-finished grinds
        return set(self._adm_pending)

    def poll_admission(self):
        """NON-BLOCKING. Store the nonce for every finished grind and drop it from
        the pending set. Returns the set of rows that just became ready."""
        if not self._adm_pending:
            return set()
        ready = set()
        for row, (job, fut) in list(self._adm_pending.items()):
            if fut.done():
                self._store_admission_result(job, fut.result())
                del self._adm_pending[row]
                ready.add(row)
        return ready

    def pending_admission_rows(self):
        """Rows whose admission grind is still in flight (must not be sampled)."""
        return set(self._adm_pending)

    @staticmethod
    def _grind(grind_fn, job):
        return grind_fn(job.msg_w, job.model_id, job.target_le,
                        job.max_tries, job.commitment)

    def _admission_pool(self):
        """Lazily-built grind pool. Sized to leave one core for the engine
        thread; override with POW_V3_GRIND_THREADS. Argon2 8 MiB scales ~linearly
        to physical cores then flattens (memory bandwidth), so more is wasteful."""
        pool = getattr(self, "_adm_pool", None)
        if pool is None:
            default = max(1, (os.cpu_count() or 2) - 1)
            raw = os.environ.get("POW_V3_GRIND_THREADS")
            n = default
            if raw:
                try:
                    n = int(raw)
                except (TypeError, ValueError):
                    self.s.logger.log(
                        f"POW_V3_GRIND_THREADS={raw!r} is not an int — using "
                        f"default {default}", "WARN")
                else:
                    if n < 1:
                        self.s.logger.log(
                            f"POW_V3_GRIND_THREADS={n} < 1 — clamping to 1", "WARN")
                        n = 1
            pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="v3grind")
            self._adm_pool = pool
        return pool

    def _warn_no_grinder(self):
        if not self._admission_warned_no_grinder:
            self._admission_warned_no_grinder = True
            self.s.logger.log(
                "v3 admission requested but native admission_grind is "
                "unavailable (proof_processor.so without argon2?) — "
                "mining nonce-less; [B_FLOOR, B_FREE) windows will be "
                "forfeited", "ERROR")

    def _prepare_admission(self, seq_id, row, context_tokens, step_idx,
                           compute_precision):
        """Build one row's grind inputs and CLEAR its stale nonce (a nonce admits
        exactly one window — msg_w changes at every boundary). Returns an
        _AdmissionJob, or None to skip (row mines nonce-less). Pure prep: no grind,
        no result write, so it is safe to run before fanning grinds out to workers."""
        self.s.ring_buffers.write_admission_nonce(row, None)

        snapshot = self.s.seq_params.get(seq_id, {}).get("pow_snapshot")
        if not snapshot:
            self.s.logger.log(
                f"v3 admission skipped for seq {seq_id}: no pow_snapshot", "WARN")
            return None
        difficulty = snapshot.get("difficulty")
        try:
            difficulty = int(difficulty)
        except (TypeError, ValueError):
            difficulty = 0
        if difficulty <= 0:
            self.s.logger.log(
                f"v3 admission skipped for seq {seq_id}: no registered "
                f"difficulty on snapshot", "WARN")
            return None

        # Build the grind preimage through the SINGLE shared source so the
        # sampler and the scheduler produce byte-identical bytes (§6). msg_w is
        # this window's first-step preimage (header falls back to block_hash
        # exactly like the sampling path); the commitment binds the full
        # pre-window prefix (the archive at the boundary IS the eventual proof's
        # prompt tokens).
        header_hex = snapshot.get("header_prefix") or snapshot.get("block_hash")
        cache = self.s.seq_caches.get(seq_id, {})
        model_identifier = (getattr(self.s.proof_writer, "model_identifier",
                                    None) or "unknown")
        msg_w, model_identifier, target_le, max_tries, commitment = \
            pow_v3.build_admission_preimage(
                header_prefix=bytes.fromhex(header_hex),
                vdf=bytes.fromhex(snapshot["vdf"]),
                tick=int(snapshot["tick"]),
                step=int(step_idx),
                context_tokens=context_tokens,
                prefix_tokens=cache.get("archive_list", []) or [],
                prefix_pad_mask=cache.get("pad_mask_list", []) or [],
                precision=compute_precision,
                difficulty=difficulty,
                model_identifier=model_identifier,
                max_tries_factor=self.admission_max_tries_factor,
            )
        return _AdmissionJob(row, seq_id, difficulty, msg_w,
                             model_identifier, target_le, max_tries, commitment)

    def _store_admission_result(self, job, nonce):
        """Engine-thread: write the ground nonce (or note exhaustion). Returns
        True iff a nonce was stored."""
        if nonce is None:
            self.s.logger.log(
                f"v3 admission grind exhausted {job.max_tries} tries for seq "
                f"{job.seq_id} (difficulty {job.difficulty}) — window mined "
                f"nonce-less", "WARN")
            return False
        self.s.ring_buffers.write_admission_nonce(job.row, bytes(nonce))
        return True

    def ensure_admission_for_window(self, seq_id, row, context_tokens,
                                    step_idx, compute_precision):
        """Single-row serial path (startup self-test + unit tests). Clears any
        stale nonce, grinds against the registered-difficulty target, and stores
        the selected 32 bytes before the window's first sampled token. Returns
        True iff an admissible nonce was stored. The batch hot path uses
        ensure_admission_for_rows (concurrent)."""
        job = self._prepare_admission(seq_id, row, context_tokens, step_idx,
                                      compute_precision)
        if job is None:
            return False
        grind_fn = self._resolve_admission_grind_fn()
        if grind_fn is None:
            self._warn_no_grinder()
            return False
        return self._store_admission_result(job, self._grind(grind_fn, job))

    def init_sequence_cache(self, seq_id, prompt_tokens):
        W = self.s.window_size
        prompt_len = len(prompt_tokens)

        if getattr(self.s, 'DEBUG_LOG', False):
            self.s.logger.log(f"Initializing cache for seq_id={seq_id} (len={prompt_len})")

        archive_list  = list(prompt_tokens)
        pad_mask_list = [False] * prompt_len

        ring = torch.zeros(W, dtype=torch.int64, device=self.s.device)
        tail = torch.as_tensor(prompt_tokens[-W:], dtype=torch.int64, device=self.s.device)
        ring[:tail.numel()] = tail
        ring_pos    = tail.numel() % W
        ring_filled = min(tail.numel(), W)

        cache = {
            "last_updated": time.time(),
            "archive_list": archive_list,
            "pad_mask_list": pad_mask_list,
            "ring": ring,
            "ring_pos": ring_pos,
            "ring_filled": ring_filled,
            # #2b defer-host accounting: gen_count counts device-committed decode
            # steps (hot path, all sampled rows); prompt_len lets the post-sync hook
            # check gen_count == len(archive_list)-prompt_len, i.e. that every device
            # step also produced a committed (non-discarded) archive token.
            "gen_count": 0,
            "prompt_len": prompt_len,
        }
        self.s.seq_caches[seq_id] = cache

    def get_context_windows(self, seq_ids):
        B, W = len(seq_ids), self.s.window_size
        out = torch.empty((B, W), dtype=torch.int64, device=self.s.device)
        for i, sid in enumerate(seq_ids):
            c  = self.s.seq_caches[sid]
            r, rp = c["ring"], c["ring_pos"]
            if rp == 0:
                out[i] = r
            else:
                out[i, :W-rp] = r[rp:]
                out[i, W-rp:] = r[:rp]
        return out

    def ensure_rows(self, seq_ids, prompt_mapping):
        for sid in seq_ids:
            if sid in self.s.row_manager.seqid_to_row:
                continue
            row = self.s.row_manager.allocate_row(sid)
            if row is None:
                old_sid, _ = self.s.row_manager.get_oldest_sequence(self.s.ring_buffers.steps)
                if old_sid is not None:
                    self.s._free_sequence(old_sid)
                    row = self.s.row_manager.allocate_row(sid)
            if row is not None:
                self.s.ring_buffers.clear_row(row)
                self.s._init_sequence_cache(sid, prompt_mapping[sid])
                # Write pow params to GPU arrays for this row (decode hex ONCE here)
                seq_params = self.s.seq_params.get(sid, {})
                pow_snapshot = seq_params.get("pow_snapshot")
                if pow_snapshot:
                    self.s.ring_buffers.write_pow_params(row, pow_snapshot)

    def update_caches(self, seq_ids, tokens):
        now = time.time()
        if _POW_DEBUG_ASSERTS:
            min_token = tokens.min()
            if min_token < 0:
                raise AssertionError(f"Negative token found: {min_token.item()}")

        toks_cpu = tokens.detach().cpu().tolist()
        max_len = 1
        PAGE    = self.s.page_size
        crossed = False

        for i, sid in enumerate(seq_ids):
            c = self.s.seq_caches.get(sid); 
            if not c: continue
            tok = toks_cpu[i]
            c["archive_list"].append(tok)
            c["pad_mask_list"].append(False)
            c["last_updated"] = now

            pos = c["ring_pos"]
            c["ring"][pos] = tokens[i]
            c["ring_pos"] = (pos + 1) % self.s.window_size
            c["ring_filled"] = min(c["ring_filled"] + 1, self.s.window_size)

            max_len = max(max_len, len(c["archive_list"]))

        prev_page    = self.s.prev_max_seq_len // PAGE
        curr_page    = max_len // PAGE
        self.s.prev_max_seq_len = max_len
        return (prev_page != curr_page and max_len > 0)

    def update_caches_device_only(self, seq_ids, tokens):
        """Hot-path subset of update_caches: GPU context-ring update + host-counter
        page-flip ONLY. No D2H copy, no archive/pad-mask/EOS/proof work — those are
        deferrable past vLLM's own sampled_token_ids.tolist() sync. Sync-free:
        c['ring'][pos] = tokens[i] is a GPU->GPU write. page_flip is computed from a
        host gen-counter, but MUST match legacy update_caches exactly because it is
        written into the proof attention_mask: legacy uses max_len = len(archive_list)
        = prompt_len + gen_count, so we reproduce prompt_len + gen_count here (NOT
        gen_count alone) or the proof's attention_mask would diverge. Used by the
        POW_ABLATE_DEFER_HOST ablation and the POW_DEFER_HOST real split."""
        now = time.time()
        PAGE = self.s.page_size
        max_len = 1
        for i, sid in enumerate(seq_ids):
            c = self.s.seq_caches.get(sid)
            if not c: continue
            pos = c["ring_pos"]
            c["ring"][pos] = tokens[i]                       # GPU->GPU, no host sync
            c["ring_pos"] = (pos + 1) % self.s.window_size
            c["ring_filled"] = min(c["ring_filled"] + 1, self.s.window_size)
            c["last_updated"] = now
            gc = c.get("gen_count", 0) + 1                   # host counter, no token value
            c["gen_count"] = gc
            # mirror legacy len(archive_list) == prompt_len + gen_count
            max_len = max(max_len, c.get("prompt_len", 0) + gc)
        prev_page = self.s.prev_max_seq_len // PAGE
        curr_page = max_len // PAGE
        self.s.prev_max_seq_len = max_len
        return (prev_page != curr_page and max_len > 0)

    def archive_and_check_host(self, sid, toks):
        """Deferred host-side bookkeeping for one sequence, run from the model
        runner AFTER vLLM's own sampled_token_ids.tolist() sync on the COMMITTED
        tokens (toks = host ints; [] rows are filtered by the caller). Does the
        work that update_caches_device_only skipped: archive/pad-mask append, EOS
        free, and the window-boundary proof — all host-side, no GPU D2H sync (the
        boundary uses the host gen_count, not the GPU steps tensor). Returns True
        if a divergence was detected (debug only)."""
        c = self.s.seq_caches.get(sid)
        if not c:
            return False
        for tok in toks:
            c["archive_list"].append(tok)
            c["pad_mask_list"].append(False)
        c["last_updated"] = time.time()

        diverged = False
        if _POW_DEBUG_ASSERTS:
            # gen_count (device, all sampled rows) must equal committed archive
            # growth (post-sync, non-discarded rows). A mismatch means a
            # partial-prefill row advanced the device ring/steps but not the
            # archive => Option-A is unsafe here and full-defer (B) is required.
            committed = len(c["archive_list"]) - c.get("prompt_len", 0)
            gc = c.get("gen_count", committed)
            if gc != committed:
                diverged = True
                self.s.logger.log(
                    f"[pow-defer] DIVERGENCE sid={sid} gen_count={gc} "
                    f"archive_committed={committed} (partial-prefill discard leaked "
                    f"into device state)", "ERROR")

        # EOS — host int compare, no sync
        if toks and toks[-1] == self.s.eos_token_id:
            self.free_sequence(sid)
            return diverged

        # window-boundary proof, triggered off the host gen_count (== device steps
        # when no discard); process_solution reads the GPU proof buffers written in
        # the hot path for this step.
        gc = c.get("gen_count", 0)
        if gc > 0 and gc % self.s.window_size == 0:
            row = self.s.row_manager.get_row(sid)
            if row is not None:
                self.s._process_solution(sid, row)
        return diverged

    def free_sequence(self, seq_id):
        row = self.s.row_manager.free_row(seq_id)
        if row is not None:
            self.s.ring_buffers.clear_row(row)
        self.s.seq_caches.pop(seq_id, None)
        self.s.seq_params.pop(seq_id, None)
        if hasattr(self.s, '_req_id_to_sid'):
            # clean reverse‐map if present
            self.s._req_id_to_sid = {rid: sid for rid, sid in self.s._req_id_to_sid.items() if sid != seq_id}

    def check_eos(self, seq_ids, tokens):
        eos_mask = tokens == self.s.eos_token_id
        if not eos_mask.any(): return
        for i, has in enumerate(eos_mask):
            if has:
                self.free_sequence(seq_ids[i])
                self.s.logger.log(f"Sequence {seq_ids[i]} ended with EOS", "INFO")

    def cleanup_stale_sequences(self, max_age=300, interval=60):
        now = time.time()
        if now - self.s._last_cleanup < interval:
            return
        self.s._last_cleanup = now
        stale = [sid for sid, c in self.s.seq_caches.items()
                 if now - c["last_updated"] > max_age]
        for sid in stale:
            self.free_sequence(sid)
            self.s.logger.log(f"Cleaned up stale seq {sid}", "INFO")

    def reset_sampler_state(self):
        self.s.logger.log("Performing complete sampler state reset", "INFO")
        self.s.seq_caches = {}
        self.s.seq_params = {}
        self.s.row_manager.seqid_to_row = {}
        self.s.row_manager.free_rows = deque(range(self.s.max_concurrency))
        # zero out all ring_buffers tensors in place
        for attr in ("topk_logits","topk_indices","chosen_probs",
                     "chosen_tokens","attention_mask","sampling_u",
                     "softmax_normalizers","steps"):
            getattr(self.s.ring_buffers, attr).zero_()
        # Also clear pow param arrays
        for attr in ("pow_tick", "pow_request_id", "pow_header", "pow_vdf", "pow_vdf_len",
                     "pow_target", "pow_share_target", "pow_block_hash", "pow_header_len",
                     "pow_valid", "pow_admission_nonce", "pow_admission_valid"):
            getattr(self.s.ring_buffers, attr).zero_()
        # host mirror of the v3 admission flags must reset with the tensors
        self.s.ring_buffers.pow_admission_valid_host = (
            [False] * self.s.ring_buffers.max_rows)
        # ...and so must the host mirror of steps, which was just zeroed on
        # device by the loop above. steps_host_ok is NOT restored here: if a
        # caller already invalidated the mirror it stays invalid for the
        # lifetime of the process.
        self.s.ring_buffers.steps_host = (
            [0] * self.s.ring_buffers.max_rows)
        self.s._pre_temp_logits = None
        self.s._log_Z = None
        self.s._sampling_tensors = None
        self.s._last_cleanup = time.time()
        self.s.logger.log("Sampler state has been completely reset", "INFO")
        return True

    def check_pow_solutions(self, seq_ids):
        rb = self.s.ring_buffers
        # `is True` rather than truthiness: only an explicit, in-sync mirror
        # may take the fast path. Anything else (missing attribute, a mock,
        # a half-initialised RingBuffers) must fall through to the device
        # tensor, which is always correct.
        if getattr(rb, "steps_host_ok", False) is True:
            # Fast path, opt-in via POW_STEPS_HOST_FASTPATH=1. The default
            # path below builds a device tensor and calls
            # nonzero().tolist() on EVERY decode step, i.e. a GPU->CPU
            # synchronisation per token. When the host mirror is being kept
            # in sync by every writer of ring_buffers.steps (see
            # RingBuffers.increment_steps) the same boundary test is pure
            # Python. steps_host_ok goes False permanently the moment any
            # writer cannot maintain the mirror, so this can only be taken
            # when the mirror is authoritative.
            steps_host = rb.steps_host
            window_size = self.s.window_size
            get_row = self.s.row_manager.get_row
            for sid in seq_ids:
                row = get_row(sid)
                if row is None:
                    continue
                steps = steps_host[row]
                if steps > 0 and steps % window_size == 0:
                    self.s._process_solution(sid, row)
            return
        rows = [self.s.row_manager.get_row(sid) for sid in seq_ids if self.s.row_manager.get_row(sid) is not None]
        if not rows: return
        rows_tensor = torch.tensor(rows, device=self.s.device, dtype=torch.long)
        steps_vals  = self.s.ring_buffers.steps[rows_tensor]
        mask        = (steps_vals % self.s.window_size == 0) & (steps_vals > 0)
        for idx in torch.nonzero(mask, as_tuple=False).squeeze(1).tolist():
            sid = seq_ids[idx]
            self.s._process_solution(sid, rows[idx])

    def ensure_sorted_topk(self, topk_logits, topk_indices):
        is_sorted = torch.all(topk_logits[:, :-1] >= topk_logits[:, 1:], dim=1)
        if not is_sorted.all():
            for pos in (~is_sorted).nonzero().squeeze(-1):
                order = torch.argsort(topk_logits[pos], descending=True)
                topk_logits[pos], topk_indices[pos] = (
                  topk_logits[pos][order],
                  topk_indices[pos][order]
                )

    def process_pow_params(self, sampling_metadata):
        if not sampling_metadata.seq_groups:
            return
        first = sampling_metadata.seq_groups[0]
        extra = getattr(first.sampling_params, "extra_args", {}) or {}
        if (p := extra.get("pow")):
            self.s.pow_hasher.update_from_payload(p)
            # self.s.logger.log(f"Updated PoW params: tick={p['tick']}", "INFO")

    def log_prompt_data(self, sampling_metadata):
        """Log prompt tokens and attention mask for debugging."""
        if len(sampling_metadata.seq_groups) == 0:
            self.s.logger.log("[DEBUG] Empty batch, nothing to log", "DEBUG")
            return

        log_file = os.path.join(
            os.path.dirname(self.s.logger.log_file_path),
            "prompt_data.log"
        )
        try:
            with open(log_file, "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] NEW BATCH\n")
                for g_idx, g in enumerate(sampling_metadata.seq_groups):
                    f.write(f"Group {g_idx}, {len(g.seq_ids)} sequences\n")
                    for seq_idx, sid in enumerate(g.seq_ids):
                        seq_data = g.seq_data[sid]
                        # prompt_token_ids tuple
                        f.write(f"Sequence {sid}:\n")
                        f.write(f"  prompt_token_ids (tuple): {seq_data.prompt_token_ids}\n")
                        # array form
                        f.write(f"  prompt_token_ids_array: {list(seq_data.prompt_token_ids_array)}\n")
                        # attention mask if present
                        att = (list(seq_data.attention_mask)
                               if hasattr(seq_data, "attention_mask") else [])
                        f.write(f"  attention_mask: {att}\n")
                        # other attrs
                        attrs = [
                            a for a in dir(seq_data)
                            if not a.startswith("_")
                            and a not in ("prompt_token_ids",
                                          "prompt_token_ids_array",
                                          "attention_mask")
                        ]
                        f.write(f"  Other attributes: {attrs}\n")
                        if hasattr(seq_data, "output_token_ids"):
                            f.write(f"  output_token_ids: {list(seq_data.output_token_ids)}\n")
                        f.write("\n")
        except Exception as e:
            self.s.logger.log(f"Error logging prompt data: {e}", "ERROR")

    def detect_real_inference(self, sampling_metadata):
        """Return True if this looks like a real inference call."""
        if sampling_metadata.seq_groups:
            first = sampling_metadata.seq_groups[0]
            extra = getattr(first.sampling_params, "extra_args", {}) or {}
            if extra.get("pow"):
                return True
            if len(sampling_metadata.seq_groups) == 1 and len(first.seq_ids) < 5:
                return True
        return False

    def process_solution(self, seq_id, row):
        """Process a potential PoW solution."""
        # Use C++ processor if available and enabled
        if self.use_cpp_processor:
            return self._process_solution_cpp(seq_id, row)
        else:
            return self._process_solution_python(seq_id, row)
    
    def _process_solution_cpp(self, seq_id, row):
        """Process solution using C++ ProofProcessor."""
        import numpy as np

        # Get step number and window data
        step_num = self.s.ring_buffers.steps[row].item()
        window_data = self.s.ring_buffers.get_window(row)

        # Compute digest
        tokens = window_data["tokens"]
        tokens_tensor = tokens.unsqueeze(0)
        tokens_bytes = _tok_le_bytes(tokens_tensor)
        step_offset = step_num % self.s.window_size

        # Use per-sequence pow_snapshot - required to avoid cross-request contamination
        seq_params = self.s.seq_params.get(seq_id, {})
        pow_snapshot = seq_params.get("pow_snapshot")

        if not pow_snapshot:
            # Cannot safely build proof without per-sequence snapshot - shared state may be contaminated
            self.s.logger.log(f"Skipping proof for seq {seq_id}: no pow_snapshot available", "WARN")
            return

        # Per-request audit flag — emit unconditionally for the
        # completion-audit cache, routed by the C++ processor to the
        # audit submit path (never solution/share). See process_proof's
        # audit_emit branch.
        audit_emit = bool(pow_snapshot.get("audit_emit"))

        # Use frozen per-sequence params
        tick = pow_snapshot["tick"]
        header_hex = pow_snapshot["header_prefix"]
        vdf_hex = pow_snapshot["vdf"]
        block_hash_hex = pow_snapshot["block_hash"]
        target_hex = pow_snapshot["target"]
        ipfs_cid = pow_snapshot.get("ipfs_cid") or ""
        request_id = pow_snapshot["request_id"]
        difficulty = pow_snapshot["difficulty"]

        header_data = hex_to_bytes_tensor(header_hex, device=self.s.device) if header_hex else None
        vdf_data = hex_to_bytes_tensor(vdf_hex, device=self.s.device)
        target_data = hex_to_bytes_tensor(target_hex, device=self.s.device)
        block_hash_data = hex_to_bytes_tensor(block_hash_hex, device=self.s.device)

        if header_data is None or header_data.numel() == 0:
            header_data = block_hash_data

        j4 = _u32le(torch.tensor([step_offset], dtype=torch.uint32, device=self.s.device))
        T8 = _u32le(torch.tensor([tick], dtype=torch.uint32, device=self.s.device))
        precision_bytes = _str_bytes(
            self.s.proof_writer.compute_precision,
            batch_size=tokens_bytes.size(0),
            device=self.s.device
        )
        # V3 (TIP-0003): the final target-critical digest is the
        # SAME construction as every step u — when this row sampled with an
        # admission nonce, the nonce must be appended here too, or the derived
        # header nonce disagrees with verifier replay.
        admission_nonce_t = (
            self.s.ring_buffers.pow_admission_nonce[row]
            if self.s.ring_buffers.pow_admission_valid_host[row] else None
        )
        msg = _build_msg(header_data, vdf_data, T8, j4, tokens_bytes, precision_bytes,
                         nonce=admission_nonce_t)
        digest = sha256_many(msg)
        # Slice 11.4 — dual-threshold classification. ``check_solution``
        # gates on the block target; ``check_share_solution`` gates on
        # the easier model-adjusted share target. A digest can meet:
        #   - both (block hits are a STRICT subset of share hits) → emit as MineResult; broker also credits the matching share
        #   - share only → emit as sub-block share (is_solution=False
        #     on the wire; worker-side classifier routes to MineShare)
        #   - neither → skip unless proxy_audit_enabled is set
        # Without a ``share_target`` on the snapshot, ``check_share_solution``
        # returns all-False and behaviour is identical to the
        # pre-slice-11 single-check path.
        is_solution = self.s.pow_hasher.check_solution(digest, pow_snapshot)
        is_share = self.s.pow_hasher.check_share_solution(digest, pow_snapshot)
        emit_any = (
            is_solution.any() or is_share.any() or audit_emit
            or self.proxy_audit_enabled
        )
        if not emit_any:
            return

        # Prepare cache data with typed arrays
        cache = self.s.seq_caches.get(seq_id, {})
        cache_data = {
            "archive_list": cache.get("archive_list", []),
            "pad_mask_list": cache.get("pad_mask_list", [])
        }

        # Prepare window data as numpy arrays
        window_data_cpp = {
            "tokens": window_data["tokens"].contiguous().cpu().numpy().astype(np.int32),
            "probs": window_data["probs"].contiguous().cpu().numpy().astype(np.float32),
            "topk_logits": window_data["topk_logits"].contiguous().cpu().numpy().astype(np.float32),
            "topk_indices": window_data["topk_indices"].contiguous().cpu().numpy().astype(np.int32),
            "attention_mask": window_data["attention_mask"].contiguous().cpu().numpy().astype(bool),
            "sampling_u": window_data["sampling_u"].contiguous().cpu().numpy().astype(np.float32),
            "softmax_normalizers": window_data["softmax_normalizers"].contiguous().cpu().numpy().astype(np.float32),
            "logsumexp_stats": window_data["logsumexp_stats"].contiguous().cpu().numpy().astype(np.float32)
        }

        # Prepare POW hasher data - use per-sequence snapshot when available
        header_bytes = header_data
        pow_hasher_data = {
            "tick": tick,
            "target": target_data.contiguous().cpu().numpy().tobytes(),
            "vdf": vdf_data.contiguous().cpu().numpy().tobytes(),
            "block_hash": block_hash_data.contiguous().cpu().numpy().tobytes(),
            "header_prefix": header_bytes.contiguous().cpu().numpy().tobytes(),
            "ipfs_cid": ipfs_cid,
            "request_id": request_id,
            "difficulty": difficulty,
            "window_size": self.s.window_size
        }
        # Forward the miner-api ingress proof-version stamp (TIP-0003
        # §9 agreement check): ProofProcessor::assemble_proof_dict compares it
        # against its own configured proof_version_ and fails loudly on env
        # drift between the proxy and this process. Absent on old proxies.
        if pow_snapshot and pow_snapshot.get("proof_version") is not None:
            pow_hasher_data["proof_version"] = int(pow_snapshot["proof_version"])
        if admission_nonce_t is not None:
            # 32 raw bytes; ProofProcessor merges them into
            # extra_flags.v3.admission_nonce via the shared helper (§3/§9) —
            # it never re-grinds and never infers from the digest.
            pow_hasher_data["admission_nonce"] = (
                admission_nonce_t.contiguous().cpu().numpy().tobytes())

        completion_id = seq_params.get("completion_id")
        
        # Get digest as numpy array
        digest_np = digest[0].contiguous().cpu().numpy()
        
        # Call C++ processor. audit_emit routes to the broker-aware
        # audit submit path inside process_proof (proof_purpose=audit,
        # never solution/share); is_solution is forced False on the wire
        # for audit proofs so they can't be misread as mining hits.
        result = self.proof_processor.process_proof(
            seq_id=seq_id,
            step_num=step_num,
            cache_data=cache_data,
            window_data=window_data_cpp,
            digest=digest_np,
            is_solution=(False if audit_emit else is_solution.any().item()),
            pow_hasher_data=pow_hasher_data,
            seq_params=seq_params,
            completion_id=completion_id,
            audit_emit=audit_emit,
            # C++ must not infer "share" from "not a solution". A share
            # is valid only if the final header hash satisfied the
            # adjusted share target, matching verify-service.
            is_share=(False if audit_emit else is_share.any().item()),
        )

        # Log if solution
        if not audit_emit and is_solution.any():
            level = "INFO" if result.get("queued", False) else "ERROR"
            self.s.logger.log(
                f"{'Solution submitted to core-node' if result.get('queued') else 'Failed to submit solution'}", 
                level
            )
            self.s.logger.log(f"Found PoW solution for sequence {seq_id}!", "INFO")
        
        return result
    
    def _process_solution_python(self, seq_id, row):
        """Process solution using Python ProofWriter (original implementation)."""
        step_num = self.s.ring_buffers.steps[row].item()
        window_data = self.s.ring_buffers.get_window(row)
        tokens = window_data["tokens"]
        tokens_tensor = tokens.unsqueeze(0)
        tokens_bytes = _tok_le_bytes(tokens_tensor)
        step_offset = step_num % self.s.window_size

        # Use per-sequence pow_snapshot - required to avoid cross-request contamination
        seq_params = self.s.seq_params.get(seq_id, {})
        pow_snapshot = seq_params.get("pow_snapshot")

        if not pow_snapshot:
            # Cannot safely build proof without per-sequence snapshot - shared state may be contaminated
            self.s.logger.log(f"Skipping proof for seq {seq_id}: no pow_snapshot available", "WARN")
            return

        # Use frozen per-sequence params
        tick = pow_snapshot["tick"]
        header_hex = pow_snapshot["header_prefix"]
        vdf_hex = pow_snapshot["vdf"]
        block_hash_hex = pow_snapshot["block_hash"]
        target_hex = pow_snapshot["target"]
        ipfs_cid = pow_snapshot.get("ipfs_cid") or ""
        request_id = pow_snapshot["request_id"]
        difficulty = pow_snapshot["difficulty"]

        header_data = hex_to_bytes_tensor(header_hex, device=self.s.device) if header_hex else None
        vdf_data = hex_to_bytes_tensor(vdf_hex, device=self.s.device)
        target_data = hex_to_bytes_tensor(target_hex, device=self.s.device)
        block_hash_data = hex_to_bytes_tensor(block_hash_hex, device=self.s.device)

        if header_data is None or header_data.numel() == 0:
            header_data = block_hash_data

        j4 = _u32le(torch.tensor([step_offset], dtype=torch.uint32, device=self.s.device))
        T8 = _u32le(torch.tensor([tick], dtype=torch.uint32, device=self.s.device))
        precision_bytes = _str_bytes(
            self.s.proof_writer.compute_precision,
            batch_size=tokens_bytes.size(0),
            device=self.s.device
        )
        # V3 (TIP-0003): final digest must include the row's
        # admission nonce when one was sampled with — same rule as the cpp
        # branch above.
        admission_nonce_t = (
            self.s.ring_buffers.pow_admission_nonce[row]
            if self.s.ring_buffers.pow_admission_valid_host[row] else None
        )
        msg = _build_msg(header_data, vdf_data, T8, j4, tokens_bytes, precision_bytes,
                         nonce=admission_nonce_t)
        digest = sha256_many(msg)
        # Slice 11.4 — dual-threshold classification, mirroring the
        # cpp branch (see _process_solution_cpp above for the full
        # rationale). Sub-block proofs that meet share but not block
        # go out with is_solution=False; the worker-side classifier
        # routes them to MineShare.
        is_solution = self.s.pow_hasher.check_solution(digest, pow_snapshot)
        is_share = self.s.pow_hasher.check_share_solution(digest, pow_snapshot)
        # audit_emit: per-request unconditional emission for the
        # completion-audit cache — no fake-easy targets; thresholds
        # stay untouched and the proof routes to the audit channel only.
        audit_emit = bool(pow_snapshot.get("audit_emit"))
        emit_any = (
            is_solution.any() or is_share.any() or audit_emit
            or self.proxy_audit_enabled
        )
        if not emit_any:
            return

        # assemble seq_info & pow_params
        cache = self.s.seq_caches.get(seq_id, {})
        archive = cache.get("archive_list", [])
        padmask = cache.get("pad_mask_list", [])
        if len(archive) > self.s.window_size:
            prompt_tokens = archive[:-self.s.window_size]
            prompt_pad = padmask[:-self.s.window_size]
        else:
            prompt_tokens, prompt_pad = [], []

        seq_info = {
            "prompt_tokens": prompt_tokens,
            "pad_mask": prompt_pad,
            **seq_params
        }
        pow_params = {
            "tick": tick,
            "target": target_data.cpu().numpy().tobytes().hex(),
            "vdf": vdf_data.cpu().numpy().tobytes().hex(),
            "block_hash": block_hash_data.cpu().numpy().tobytes().hex(),
            "header_prefix": (
                header_data.cpu().numpy().tobytes().hex()
                if header_data is not None else ""
            ),
            "ipfs_cid": ipfs_cid
        }
        # Forward the miner-api ingress proof-version stamp (TIP-0003
        # §9): ProofWriter.write_proof compares it against its configured
        # proof_version and fails loudly on env drift. Absent on old proxies.
        if pow_snapshot and pow_snapshot.get("proof_version") is not None:
            pow_params["proof_version"] = int(pow_snapshot["proof_version"])

        # Attach completion_id from stable req-id mapping
        completion_id = None
        try:
            completion_id = self.s.seq_params.get(seq_id, {}).get("completion_id")
        except Exception:
            completion_id = None

        pow_blob, pow_dic = self.s.proof_writer.write_proof(
            seq_id, step_num, window_data, digest,
            is_solution.any().item(), pow_params, seq_info, completion_id=completion_id,
            admission_nonce=(
                bytes(admission_nonce_t.cpu().numpy().tobytes())
                if admission_nonce_t is not None else None),
        )
        pow_blob_hash = hashlib.sha256(pow_blob).digest()
        nonce = struct.unpack('<I', digest[0, :4].cpu().numpy().tobytes())[0]
        adjusted_bits = get_compact(target_data.cpu().numpy().tobytes())

        # Audit-only sequences: the proof exists for the completion-audit
        # cache, never for mining — even a digest that happens to meet the
        # block target must not close a broker lease for an unregistered
        # model. Submit to the audit channel and stop here.
        if audit_emit:
            if hasattr(self.s.submitter, 'submit_proof_for_audit'):
                self.s.submitter.submit_proof_for_audit(
                    req_id=request_id,
                    proof_dict=pow_dic
                )
            return

        # Submit proof for audit if proxy is enabled (for ALL proofs, not just solutions)
        if self.proxy_audit_enabled and hasattr(self.s.submitter, 'submit_proof_for_audit'):
            self.s.submitter.submit_proof_for_audit(
                req_id=request_id,
                proof_dict=pow_dic
            )

        # Always submit to core-node if it's a valid solution. Share-only
        # hits go to the broker-mode share path when the submitter supports
        # it; in local_miner mode submit_share intentionally no-ops so Core
        # Node never sees sub-block proofs.
        if is_solution.any():
            success = self.s.submitter.submit_solution(
                req_id=request_id,
                nonce=nonce,
                adjusted_bits=adjusted_bits,
                pow_blob_hash=pow_blob_hash,
                difficulty=difficulty,
                proof_dict=pow_dic
            )
            level = "INFO" if success else "ERROR"
            self.s.logger.log(
                f"{'Solution submitted to core-node' if success else 'Failed to submit solution to core-node'}", level
            )
            self.s.logger.log(f"Found PoW solution for sequence {seq_id}!", "INFO")
        elif is_share.any() and hasattr(self.s.submitter, 'submit_share'):
            success = self.s.submitter.submit_share(
                req_id=request_id,
                nonce=nonce,
                adjusted_bits=adjusted_bits,
                pow_blob_hash=pow_blob_hash,
                difficulty=difficulty,
                proof_dict=pow_dic
            )
            level = "DEBUG" if success else "ERROR"
            self.s.logger.log(
                f"{'Share submitted to broker' if success else 'Failed to submit share to broker'}",
                level
            )
