# SPDX-License-Identifier: Apache-2.0
"""
ZMQ writer for sending mining response proofs with integrated Proof serialization
"""
import zmq
import logging
import threading
import queue
import time
import flatbuffers
import os
import numpy as np
import struct
from typing import Optional, Dict, Any

# Production-safe imports expect the package `proof` to be on PYTHONPATH
from proof import MiningResponse, Proof, FloatArray, UIntArray

# V3 prompt-binding helpers (TIP-0003) — deployed next to this
# file everywhere it is copied.
try:
    import pow_v3
except ImportError:
    from . import pow_v3

logger = logging.getLogger(__name__)

# Mutually-exclusive PoW egress modes (the PoW writer egress envvar
# contract). The mode chooses both the primary
# destination and whether dual-publish to a secondary "proxy" socket is
# allowed; the two modes are not meant to coexist in one deployment.
_EGRESS_MODE_LOCAL_MINER = "local_miner"
_EGRESS_MODE_BROKER = "broker"
_VALID_EGRESS_MODES = frozenset({_EGRESS_MODE_LOCAL_MINER, _EGRESS_MODE_BROKER})
_DEFAULT_EGRESS_MODE = _EGRESS_MODE_LOCAL_MINER
# Truthy values accepted for the POW_PROXY_ENABLE flag. Kept identical
# to the legacy parsing so a config that was truthy yesterday stays
# truthy today.
_TRUTHY_VALUES = ('1', 'true', 'True')


class PowEgressConfigError(RuntimeError):
    """Raised at writer construction when env config is internally
    inconsistent — e.g. broker mode with a Core Node primary destination
    or dual-publish still enabled. Operators MUST fix the env, not catch
    this exception. A loud, fail-fast startup beats a silent solution
    leak to Core Node from a worker that thinks it's in broker mode."""


class MiningResponseWriter:
    """Sends mining response proofs via ZMQ"""

    def __init__(self, max_queue_size: int = 100):
        """
        Initialize the mining response writer

        Args:
            max_queue_size: Maximum number of responses to queue

        Env-driven config (read once at construction):
            POW_EGRESS_MODE: ``local_miner`` (default, preserves the
                legacy Core-Node-primary + optional proxy dual-publish
                topology) or ``broker`` (primary destination is the
                miner-proxy ProofCollector at 127.0.0.1:7002 by default;
                no dual-publish path; POW_PROXY_ENABLE must be falsy
                and ZMQ_PUSH_HOST must not look like a Core Node).
            ZMQ_PUSH_HOST / ZMQ_PUSH_PORT: primary destination override.
                Defaults differ per mode (see above).
            POW_PROXY_ENABLE / POW_PROXY_PUSH_HOST / POW_PROXY_PUSH_PORT:
                dual-publish secondary socket. Honoured only in
                ``local_miner`` mode. ``broker`` mode refuses to start
                if POW_PROXY_ENABLE is truthy.
            POW_SAVE_TO_DISK: write each FB to /data/pow_proofs/.

        Raises:
            ValueError: POW_EGRESS_MODE is not one of the supported values.
            PowEgressConfigError: broker mode is internally inconsistent
                with the other env vars (proxy enabled, or primary points
                at a hostname that looks like Core Node).
        """
        # ---------------------------------------------------------- mode
        egress_mode_raw = os.environ.get('POW_EGRESS_MODE', _DEFAULT_EGRESS_MODE)
        if egress_mode_raw not in _VALID_EGRESS_MODES:
            raise ValueError(
                f"POW_EGRESS_MODE={egress_mode_raw!r} is not supported; "
                f"must be one of {sorted(_VALID_EGRESS_MODES)}"
            )
        self._egress_mode = egress_mode_raw

        # ---------------------------------------------------- destinations
        # local_miner: primary defaults to localhost:7000 (Core Node).
        # broker: primary defaults to 127.0.0.1:7002 (miner-proxy
        # ProofCollector). Operator env overrides win in both modes,
        # subject to the broker-mode safety checks below.
        if self._egress_mode == _EGRESS_MODE_BROKER:
            default_host = '127.0.0.1'
            default_port = 7002
        else:
            default_host = 'localhost'
            default_port = 7000
        self.push_host = os.environ.get('ZMQ_PUSH_HOST', default_host)
        self.push_port = int(os.environ.get('ZMQ_PUSH_PORT', default_port))

        # ------------------------------------------------- proxy / dual-publish
        proxy_enable_raw = os.environ.get('POW_PROXY_ENABLE', '0')
        proxy_enable_truthy = proxy_enable_raw in _TRUTHY_VALUES
        if self._egress_mode == _EGRESS_MODE_BROKER:
            # Broker-mode safety net 1: POW_PROXY_ENABLE must be falsy.
            # The point of broker mode is that the worker has a SINGLE
            # destination (the ProofCollector); a dual-publish path
            # would let solutions reach Core Node without the broker's
            # lease being closed.
            if proxy_enable_truthy:
                raise PowEgressConfigError(
                    f"POW_EGRESS_MODE=broker is incompatible with "
                    f"POW_PROXY_ENABLE={proxy_enable_raw!r}; broker mode "
                    f"has no dual-publish path. Set POW_PROXY_ENABLE=false "
                    f"or switch to POW_EGRESS_MODE=local_miner."
                )
            # Broker-mode safety net 2: refuse if the primary host looks
            # like a Core Node service. The common misconfiguration is to
            # flip POW_EGRESS_MODE=broker but leave ZMQ_PUSH_HOST pointing
            # at the previous Core Node DNS name. We do not want to
            # silently forward solutions to Core Node from a worker that
            # advertises broker mode.
            if 'core-node' in self.push_host.lower():
                raise PowEgressConfigError(
                    f"POW_EGRESS_MODE=broker but ZMQ_PUSH_HOST={self.push_host!r} "
                    f"looks like a Core Node destination. The primary destination "
                    f"in broker mode must be the miner-proxy ProofCollector. "
                    f"Set ZMQ_PUSH_HOST to a non-Core-Node hostname (default 127.0.0.1)."
                )
            self._proxy_enable = False
            self._proxy_host = ''
            self._proxy_port = 0
        else:
            self._proxy_enable = proxy_enable_truthy
            self._proxy_host = os.environ.get('POW_PROXY_PUSH_HOST', 'localhost')
            self._proxy_port = int(os.environ.get('POW_PROXY_PUSH_PORT', 7002))

        self.max_queue_size = max_queue_size

        # Queue for responses to send
        self.response_queue = queue.Queue(maxsize=max_queue_size)

        # Threading components
        self.running = False
        self.thread = None

        # ZMQ components
        self._zmq_context = None
        self._socket = None
        self._proxy_socket = None

        # Disk save toggle
        self._save_to_disk = os.environ.get('POW_SAVE_TO_DISK', '0') in _TRUTHY_VALUES

        # Stats
        self.messages_sent = 0
        self.messages_failed = 0

        logger.info(
            "MiningResponseWriter configured: egress_mode=%s primary=%s:%d proxy_enable=%s",
            self._egress_mode, self.push_host, self.push_port, self._proxy_enable,
        )
        
    def start(self):
        """Start the writer thread"""
        if self.running:
            logger.warning("Mining response writer already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Mining response writer started on port {self.push_port}")
    
    def stop(self):
        """Stop the writer thread gracefully"""
        logger.info("Stopping mining response writer...")
        self.running = False
        
        # Add sentinel to unblock queue
        try:
            self.response_queue.put(None, block=False)
        except queue.Full:
            pass
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.error("Writer thread did not stop cleanly")
        
        logger.info("Mining response writer stopped")
    
    def submit_response(self, 
                       req_id: int,
                       nonce: int, 
                       adjusted_bits: int,
                       pow_blob_hash: bytes,
                       difficulty: int,
                       proof_dict: dict,
                       proxy_only: bool = False) -> bool:
        """
        Submit a mining response to be sent
        
        Args:
            req_id: Request ID from the mining job
            nonce: Found nonce value
            adjusted_bits: Adjusted difficulty bits
            pow_blob_hash: Hash of the PoW blob
            difficulty: Mining difficulty
            proof_dict: Dictionary containing proof data (not serialized bytes!)
            proxy_only: If True, only send to proxy (not core-node)
            
        Returns:
            True if queued successfully, False if queue is full
        """
        try:
            response_data = {
                'req_id': req_id,
                'nonce': nonce,
                'adjusted_bits': adjusted_bits,
                'pow_blob_hash': pow_blob_hash,
                'difficulty': difficulty,
                'proof_dict': proof_dict,  # Pass the dictionary, not bytes
                'proxy_only': proxy_only  # Flag for proxy-only submission
            }
            
            self.response_queue.put(response_data, block=False)
            logger.debug(f"Queued mining response for req_id={req_id}, nonce={nonce}")
            return True
            
        except queue.Full:
            logger.warning("Response queue full, dropping mining response")
            self.messages_failed += 1
            return False

    def is_broker_mode(self) -> bool:
        """Return True when the writer's single egress is miner-proxy."""
        return self._egress_mode == _EGRESS_MODE_BROKER
    
    def _run(self):
        """Main writer loop - runs in background thread"""
        logger.info("Mining response writer thread started")
        
        try:
            # Create ZMQ context and socket
            self._zmq_context = zmq.Context()
            self._socket = self._zmq_context.socket(zmq.PUSH)
            self._socket.connect(f"tcp://{self.push_host}:{self.push_port}")
            
            # Set socket options
            self._socket.setsockopt(zmq.SNDHWM, 1000)  # High water mark
            self._socket.setsockopt(zmq.LINGER, 1000)  # Linger on close (ms)
            
            logger.info(f"ZMQ connected to tcp://{self.push_host}:{self.push_port}")

            # Optional proxy socket
            if self._proxy_enable:
                try:
                    self._proxy_socket = self._zmq_context.socket(zmq.PUSH)
                    self._proxy_socket.connect(f"tcp://{self._proxy_host}:{self._proxy_port}")
                    self._proxy_socket.setsockopt(zmq.SNDHWM, 1000)
                    self._proxy_socket.setsockopt(zmq.LINGER, 1000)
                    logger.info(f"ZMQ proxy connected to tcp://{self._proxy_host}:{self._proxy_port}")
                except Exception as e:
                    logger.error(f"Failed to setup proxy ZMQ socket: {e}")
                    self._proxy_socket = None
            
            while self.running:
                try:
                    # Get response from queue (with timeout)
                    response_data = self.response_queue.get(timeout=1.0)
                    
                    # Check for sentinel
                    if response_data is None:
                        break
                    
                    # Serialize and send
                    fb_data = self._serialize_response(response_data)

                    # --- Optional: SAVE THE BINARY TO DISK ---
                    if self._save_to_disk:
                        try:
                            self.save_dir = "/data/pow_proofs/"
                            os.makedirs(self.save_dir, exist_ok=True)
                            file_name = f"{response_data['req_id']}_{response_data['nonce']}.bin"
                            file_path = os.path.join(self.save_dir, file_name)
                            with open(file_path, 'wb') as f:
                                f.write(fb_data)
                        except Exception:
                            logger.exception("Failed to save proof blob to disk")

                    # Routing depends on egress mode:
                    #   broker      → single destination (primary IS the
                    #                 ProofCollector); proxy_only is a
                    #                 no-op effect because there is no
                    #                 separate audit channel. Every
                    #                 proof goes to the primary; the
                    #                 broker classifies it downstream.
                    #   local_miner → preserve legacy split: solutions
                    #                 go to primary (Core Node) and
                    #                 optionally also to proxy; audit
                    #                 (proxy_only=True) goes ONLY to
                    #                 proxy.
                    proxy_only = response_data.get('proxy_only', False)

                    if self._egress_mode == _EGRESS_MODE_BROKER:
                        self._socket.send(fb_data)
                    elif proxy_only:
                        # Send ONLY to proxy for audit (non-solutions)
                        if self._proxy_socket is not None:
                            self._proxy_socket.send(fb_data)
                            logger.debug(f"Sent proof to proxy only for audit: req_id={response_data['req_id']}")
                    else:
                        # Send to core-node (always for solutions)
                        self._socket.send(fb_data)

                        # Also send to proxy if configured (dual-publish for solutions)
                        if self._proxy_socket is not None:
                            try:
                                self._proxy_socket.send(fb_data, flags=zmq.DONTWAIT)
                            except Exception as e:
                                logger.error(f"Proxy ZMQ send error: {e}")
                    
                    self.messages_sent += 1
                    logger.info(f"Sent mining response: req_id={response_data['req_id']}, "
                               f"nonce={response_data['nonce']}")
                    
                except queue.Empty:
                    continue
                    
                except zmq.error.ZMQError as e:
                    logger.error(f"ZMQ send error: {e}")
                    self.messages_failed += 1
                    
                except Exception as e:
                    logger.exception(f"Error sending mining response: {e}")
                    self.messages_failed += 1
                    
        except Exception as e:
            logger.exception(f"Fatal writer error: {e}")
            
        finally:
            # Clean up resources
            if self._socket:
                self._socket.close()
            if self._proxy_socket:
                self._proxy_socket.close()
            if self._zmq_context:
                self._zmq_context.term()
                
        logger.info("Mining response writer thread stopped")
    
    def _serialize_response(self, response_data: Dict[str, Any]) -> bytes:
        """
        Serialize response data to FlatBuffer format with integrated Proof
        
        Args:
            response_data: Dictionary containing response fields and proof_dict
            
        Returns:
            Serialized FlatBuffer bytes
        """
        builder = flatbuffers.Builder(1024)
        
        # Get the proof dictionary
        obj = response_data['proof_dict']
        
        # Extract completion_id if available
        completion_id_str = ""
        if 'completion_id' in obj:
            completion_id_str = str(obj.get('completion_id', ''))
        elif 'model_config_diff' in obj:
            # Try to extract from model_config_diff for backward compatibility
            try:
                import json
                mcd = obj['model_config_diff']
                if isinstance(mcd, dict):
                    completion_id_str = mcd.get('completion_id', '')
                elif isinstance(mcd, str):
                    mcd_dict = json.loads(mcd)
                    completion_id_str = mcd_dict.get('completion_id', '')
            except:
                pass
        
        # Helper functions
        def _to_bytes(hex_str):
            """Convert hex string to bytes"""
            return bytes.fromhex(hex_str)
        
        def to_python_string(val):
            """Convert value to string"""
            if isinstance(val, str):
                return val
            elif isinstance(val, dict):
                import json
                return json.dumps(val)
            return str(val)
        
        # === SERIALIZE PROOF TABLE FIRST ===
        
        # Create strings for Proof
        proof_version = int(obj.get('version', 2))
        mid_off = builder.CreateString(obj.get('model_identifier', ''))
        cp_off = builder.CreateString(obj.get('compute_precision', ''))
        ipfs_off = builder.CreateString(obj.get('ipfs_cid', ''))
        if proof_version >= pow_v3.V3_PROOF_VERSION:
            # v3 carrier (TIP-0003): canonical JSON; merge a
            # sampler-selected admission nonce (side key, bytes or 64-hex)
            # through the shared helper just before serialization.
            mcd = obj.get('model_config_diff', {})
            nonce = obj.get('admission_nonce')
            if nonce is not None:
                nonce_hex = (nonce.hex() if isinstance(nonce, (bytes, bytearray))
                             else str(nonce))
                extra_str = pow_v3.merge_extra_flags_v3(mcd, nonce_hex)
            else:
                extra_str = mcd if isinstance(mcd, str) else pow_v3.canonical_json(mcd or {})
            extra_off = builder.CreateString(extra_str)
        else:
            extra_off = builder.CreateString(to_python_string(obj.get('model_config_diff', {})))
        
        # Create byte vectors for Proof
        tgt_off = builder.CreateByteVector(_to_bytes(obj['target']))
        vdf_off = builder.CreateByteVector(_to_bytes(obj['vdf']))
        block_hash_off = builder.CreateByteVector(_to_bytes(obj['block_hash']))
        hash_off = builder.CreateByteVector(_to_bytes(obj['hash']))
        hdr_off = builder.CreateByteVector(_to_bytes(obj.get('header_prefix', '')))
        
        # Extract scalar values
        temp = float(obj.get('temperature', 1.0))
        p = float(obj.get('top_p', 1.0))
        k = int(obj.get('top_k', 50)) & 0xFFFFFFFF
        rp = float(obj.get('repetition_penalty', 1.0))
        
        # Create 1D vectors
        def make_vec_uint32(data):
            builder.StartVector(4, len(data), 4)
            for v in reversed(data):
                val = int(v) & 0xFFFFFFFF
                builder.PrependUint32(val)
            return builder.EndVector()
        
        def make_vec_uint8(data):
            builder.StartVector(1, len(data), 1)
            for v in reversed(data):
                val = int(v) & 0xFF
                builder.PrependUint8(val)
            return builder.EndVector()
        
        def make_vec_bool(data):
            builder.StartVector(1, len(data), 1)
            for v in reversed(data):
                builder.PrependBool(bool(v))
            return builder.EndVector()
        
        def make_vec_float32(data):
            builder.StartVector(4, len(data), 4)
            for v in reversed(data):
                f32 = np.float32(v).item()
                builder.PrependFloat32(f32)
            return builder.EndVector()
        
        # Create 1D vectors for Proof
        ctoks = make_vec_uint32(obj['chosen_tokens'])
        pp = make_vec_float32(obj['chosen_probs'])
        su = make_vec_float32(obj['sampling_u'])
        sn = make_vec_float32(obj['softmax_normalizers'])
        pt = make_vec_uint32(obj['prompt_tokens'])
        pm = make_vec_bool(obj['pad_mask'])
        
        # Create 2D arrays
        def _wrap_float32(row):
            FloatArray.StartValuesVector(builder, len(row))
            for v in reversed(row):
                f32 = np.float32(v).item()
                builder.PrependFloat32(f32)
            vec = builder.EndVector()
            FloatArray.Start(builder)
            FloatArray.AddValues(builder, vec)
            return FloatArray.End(builder)
        
        def _wrap_uint32(row):
            UIntArray.StartValuesVector(builder, len(row))
            for v in reversed(row):
                u32 = int(v) & 0xFFFFFFFF
                builder.PrependUint32(u32)
            vec = builder.EndVector()
            UIntArray.Start(builder)
            UIntArray.AddValues(builder, vec)
            return UIntArray.End(builder)
        
        # Create 2D arrays for Proof
        logits_offs = [_wrap_float32(r) for r in obj['topk_logits']]
        Proof.StartTopkLogitsVector(builder, len(logits_offs))
        for off in reversed(logits_offs):
            builder.PrependUOffsetTRelative(off)
        topk_logits_off = builder.EndVector()
        
        idx_offs = [_wrap_uint32(r) for r in obj['topk_indices']]
        Proof.StartTopkIndicesVector(builder, len(idx_offs))
        for off in reversed(idx_offs):
            builder.PrependUOffsetTRelative(off)
        topk_indices_off = builder.EndVector()
        
        lse_offs = [_wrap_float32(r) for r in obj['logsumexp_stats']]
        Proof.StartLogsumexpStatsVector(builder, len(lse_offs))
        for off in reversed(lse_offs):
            builder.PrependUOffsetTRelative(off)
        lse_off = builder.EndVector()
        
        # Build the Proof table
        Proof.Start(builder)
        Proof.AddVersion(builder, proof_version)
        Proof.AddTick(builder, int(obj['tick']) & 0xFFFFFFFFFFFFFFFF)
        Proof.AddTimestamp(builder, int(obj['timestamp']) & 0xFFFFFFFFFFFFFFFF)
        Proof.AddIsSolution(builder, 1 if obj['is_solution'] else 0)
        Proof.AddModelIdentifier(builder, mid_off)
        Proof.AddComputePrecision(builder, cp_off)
        Proof.AddIpfsCid(builder, ipfs_off)
        Proof.AddExtraFlags(builder, extra_off)
        Proof.AddTemperature(builder, temp)
        Proof.AddTopP(builder, p)
        Proof.AddTopK(builder, k)
        Proof.AddRepetitionPenalty(builder, rp)
        
        Proof.AddTarget(builder, tgt_off)
        Proof.AddVdf(builder, vdf_off)
        Proof.AddHash(builder, hash_off)
        Proof.AddBlockHash(builder, block_hash_off)
        Proof.AddHeaderPrefix(builder, hdr_off)
        
        Proof.AddChosenTokens(builder, ctoks)
        Proof.AddChosenProbs(builder, pp)
        Proof.AddSamplingU(builder, su)
        Proof.AddSoftmaxNormalizers(builder, sn)
        Proof.AddLogsumexpStats(builder, lse_off)
        Proof.AddPromptTokens(builder, pt)
        Proof.AddPadMask(builder, pm)
        Proof.AddTopkLogits(builder, topk_logits_off)
        Proof.AddTopkIndices(builder, topk_indices_off)
        
        proof_offset = Proof.End(builder)
        
        # === NOW CREATE MININGRESPONSE WITH PROOF REFERENCE ===
        
        # Create pow_blob_hash vector
        pow_blob_hash_offset = builder.CreateByteVector(response_data['pow_blob_hash'])
        
        # Create completion_id string
        completion_id_offset = builder.CreateString(completion_id_str)
        if completion_id_str:
            logger.info(f"[DEBUG zmq_pow_writer] Sending proof with completion_id: {completion_id_str}")
        
        # Create MiningResponse
        MiningResponse.Start(builder)
        MiningResponse.AddReqId(builder, response_data['req_id'])
        MiningResponse.AddNonce(builder, response_data['nonce'])
        MiningResponse.AddAdjustedBits(builder, response_data['adjusted_bits'])
        MiningResponse.AddPowBlobHash(builder, pow_blob_hash_offset)
        MiningResponse.AddDifficulty(builder, response_data['difficulty'])
        MiningResponse.AddPowBlob(builder, proof_offset)  # This is now a proper table reference!
        MiningResponse.AddCompletionId(builder, completion_id_offset)
        response_offset = MiningResponse.End(builder)
        
        # Finish the buffer
        builder.Finish(response_offset)
        
        return bytes(builder.Output())
    
    def get_status(self) -> Dict[str, Any]:
        """Get writer status and statistics"""
        return {
            "running": self.running,
            "port": self.push_port,
            "queue_size": self.response_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "has_socket": self._socket is not None
        }

class MiningResponseSubmitter:
    """
    High-level interface for submitting mining responses
    Manages the writer lifecycle and provides a simple API
    """
    
    def __init__(self):
        self.writer = MiningResponseWriter()
        self.writer.start()
        
    def submit_proof_for_audit(self,
                              req_id: int,
                              proof_dict: dict) -> bool:
        """
        Submit a proof for audit (non-solution).

        Routing by egress mode:
          - ``broker``: the single destination IS the miner-proxy
            ProofCollector, so audit proofs go via the primary socket.
            What keeps them out of the mining path is the explicit
            ``proof_purpose=audit`` marker stamped below (carried in
            ``model_config_diff`` → ``Proof.extra_flags``, no schema
            change) — the collector branches on it BEFORE its mining
            filters and never feeds them to MINE_RESULT/MINE_SHARE,
            so a misclassified audit proof can't close a broker lease.
          - ``local_miner``: legacy split — audit proofs only flow when
            ``POW_PROXY_ENABLE=true`` provides the proxy channel, and
            then ONLY to the proxy (never Core Node). No-op otherwise,
            matching the C++ submitter
            (``pow_zmq_writer.cpp:submit_proof_for_audit``); the C++
            path keeps mining-only semantics — audit sequences are
            delegated to this python writer by CommonSamplerHelper.

        Args:
            req_id: Request ID from mining job
            proof_dict: Proof data as dictionary

        Returns:
            True if submitted successfully, or True if intentionally
            no-op'd (local_miner without proxy channel).
        """
        import json as _json
        # Stamp the explicit purpose marker so the ProofCollector can
        # classify without heuristics. model_config_diff is what
        # _serialize_response writes into Proof.extra_flags.
        mcd = proof_dict.get('model_config_diff')
        if isinstance(mcd, str) and mcd.strip():
            try:
                mcd = _json.loads(mcd)
            except Exception:
                mcd = {"_diff": mcd}
        if not isinstance(mcd, dict):
            mcd = {}
        mcd['proof_purpose'] = 'audit'
        proof_dict['model_config_diff'] = mcd

        if self.writer._egress_mode == _EGRESS_MODE_BROKER:
            return self.writer.submit_response(
                req_id=req_id,
                nonce=0,  # Dummy value for non-solutions
                adjusted_bits=0,  # Dummy value for non-solutions
                pow_blob_hash=b'',  # Dummy value for non-solutions
                difficulty=0,  # Dummy value for non-solutions
                proof_dict=proof_dict,
                proxy_only=False  # broker mode: primary socket IS the collector
            )

        # local_miner: no-op when there is no audit channel live.
        if not self.writer._proxy_enable:
            return True

        # Create a minimal response for proxy-only submission
        # We don't need nonce/difficulty/etc for non-solutions
        return self.writer.submit_response(
            req_id=req_id,
            nonce=0,  # Dummy value for non-solutions
            adjusted_bits=0,  # Dummy value for non-solutions
            pow_blob_hash=b'',  # Dummy value for non-solutions
            difficulty=0,  # Dummy value for non-solutions
            proof_dict=proof_dict,
            proxy_only=True  # Critical: only send to proxy, not core-node
        )
    
    def submit_solution(self,
                       req_id: int,
                       nonce: int,
                       adjusted_bits: int,
                       pow_blob_hash: bytes,
                       difficulty: int,
                       proof_dict: dict) -> bool:
        """
        Submit a mining solution to core-node (and proxy if enabled)
        
        Args:
            req_id: Request ID from mining job
            nonce: Found nonce
            adjusted_bits: Adjusted difficulty
            pow_blob_hash: Hash of proof blob
            difficulty: Mining difficulty
            proof_dict: Proof data as dictionary (NOT serialized bytes!)
            
        Returns:
            True if submitted successfully
        """
        return self.writer.submit_response(
            req_id=req_id,
            nonce=nonce,
            adjusted_bits=adjusted_bits,
            pow_blob_hash=pow_blob_hash,
            difficulty=difficulty,
            proof_dict=proof_dict  # Pass dictionary, not bytes
        )

    def submit_share(self,
                     req_id: int,
                     nonce: int,
                     adjusted_bits: int,
                     pow_blob_hash: bytes,
                     difficulty: int,
                     proof_dict: dict) -> bool:
        """Submit a sub-block mining share in broker mode.

        Shares have a consumer only when the writer's primary destination is
        the broker-side miner-proxy ProofCollector. In local_miner mode the
        primary destination is a Core Node, which must not receive sub-block
        proofs, so this intentionally no-ops.
        """
        if not self.writer.is_broker_mode():
            return True
        return self.writer.submit_response(
            req_id=req_id,
            nonce=nonce,
            adjusted_bits=adjusted_bits,
            pow_blob_hash=pow_blob_hash,
            difficulty=difficulty,
            proof_dict=proof_dict,
            proxy_only=False,
        )

    def is_broker_mode(self) -> bool:
        return self.writer.is_broker_mode()
    
    def shutdown(self):
        """Shutdown the submitter"""
        self.writer.stop()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get submission statistics"""
        return self.writer.get_status()
