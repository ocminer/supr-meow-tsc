#!/usr/bin/env python3
"""vLLM <-> stratum bridge for TensorCash PoW mining.

Speaks the same stratum protocol as supr-meow-tsc (src/stratum.cpp) and does the
same proof->submit as src/poi.cpp, but the compute backend is the local vLLM
PoW worker instead of the embedded llama engine.

Flow:
  pool --stratum--> jobs/targets/model
       bridge injects the job's PoW params into vLLM /v1/completions requests
       (vllm_xargs["pow"]) -> vLLM PoW sampler emits MiningResponse over ZMQ
       bridge PULLs those, computes the header hash, and mining.submit()s them.

The pool ISSUES the VDF (field 9) — verified against suprnova — so no local
chiavdf. prompt_seed (field 8) is absent on suprnova, so prompts are self-salted.

LOCAL/TEST use: point --pool at the fake pool. Do NOT point at production until
the pool full-tier audit is arranged.
"""
import argparse, socket, json, threading, time, hashlib, base64, os, sys, itertools, queue, random
import zmq
sys.path.insert(0, os.path.expanduser('~/src-meow/vendor/tensorcash/shared-utils/fb-schemas'))
from proof.MiningResponse import MiningResponse
import urllib.request

MODEL_ID_DEFAULT = "Qwen/Qwen3-8B@9c925d64d72725edaf899c6cb9c377fd0709d9c5"
# MAINNET prompt style 1 (open-ended). A factual "explain X" prompt makes the 8B
# too confident -> low transcript entropy -> below the chain's B_cred floor
# (>=70 bits / ~1.18 bits/step) -> shares rejected. This is the exact prompt
# supr-meow-tsc/src/main.cpp uses (POW_PROMPT_STYLE=1, default since 2026-08-06).
PROMPT_TMPL = ("Invent a strange short tale that has never been told. "
               "Do not explain it. Begin abruptly. [{salt}]")


def sha256d(b):
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


class Bridge:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.job = None          # dict: pow payload + job_id
        self.job_by_req = {}     # req_id -> job_id
        self.difficulty = 1000000
        self.share_target = None
        self.model_id = MODEL_ID_DEFAULT
        self.nonce = itertools.count(1)
        self.sock = None
        self.wfile = None
        self.next_id = itertools.count(3)
        self.submitted = 0
        self.accepted = 0
        self.rejected = 0
        self.stale = 0
        self.below_floor = 0     # proofs dropped by the B_cred gate
        self._id_checked = False # dumped+logged the stamped model id once
        self.dbg_resp = 0
        self.proofs = 0
        self._t0 = time.time()
        self.stop = False
        self.salt_base = f"{int(time.time())}-{os.getpid()}"  # unique per run

    # ---- stratum ----------------------------------------------------------
    def connect(self):
        ip = socket.getaddrinfo(self.args.host, self.args.port, socket.AF_INET,
                                socket.SOCK_STREAM)[0][4]
        self.sock = socket.create_connection(ip, timeout=30)
        # Drop the connect timeout — otherwise the reader's blocking recv raises
        # TimeoutError after 30s of no pool traffic (idle gaps between jobs are
        # normal), the reader thread dies, pongs stop, and the pool drops us.
        self.sock.settimeout(None)
        # v2 (drain-stall fix, 2026-08-09): a half-open pool socket used to
        # block the drain thread forever inside send() -> no submits -> the
        # pool idle-dropped us every ~2.5min. SO_SNDTIMEO bounds only writes
        # (recv stays blocking for the reader): a stuck write now raises
        # OSError -> send() drops the line -> supervisor reconnects.
        import struct as _struct
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO,
                             _struct.pack("ll", 30, 0))
        # keepalive: detect a dead peer during idle gaps (75s idle, 4x15s probes)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 75)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 15)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4)
        # Separate read/write handles — a single rwb makefile shared across the
        # reader thread and send() drops/garbles the submit responses.
        self.rfile = self.sock.makefile("rb")
        self.wfile = self.sock.makefile("wb")
        self.send({"id": 1, "method": "mining.subscribe",
                   "params": ["vllm-stratum-bridge/0.1", None,
                              {"protocol": "tsc/1.0", "prompt_seed": True, "pool_vdf": True}]})
        self.send({"id": 2, "method": "mining.authorize", "params": [self.args.user, self.args.password]})

    def send(self, o):
        # Best-effort: on a dropped socket, drop this line (the supervisor
        # reconnects); never crash the caller. Guard writes with the send lock.
        with self.lock:
            wf = self.wfile
            if wf is None:
                return
            try:
                wf.write((json.dumps(o) + "\n").encode()); wf.flush()
            except OSError:
                self.wfile = None   # signal the supervisor to reconnect

    def reader(self):
        # One pass over the current connection; returns on EOF/error so the
        # supervisor can reconnect.
        try:
            for line in self.rfile:
                if self.stop:
                    return
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                self.on_message(m)
        except OSError:
            return

    def supervisor(self):
        # Keep a live pool connection: (re)connect, run the reader until it ends
        # or the socket breaks, then reconnect. Jobs/dispatch/drain persist.
        while not self.stop:
            try:
                self.connect()
                print("[conn] connected", flush=True)
                self.reader()          # blocks until EOF / error
            except OSError as e:
                print(f"[conn] connect failed: {e}", flush=True)
            self.wfile = None
            try:
                self.sock.close()
            except Exception:
                pass
            if self.stop:
                break
            # v2: jittered backoff — when a pool restart cycles every miner at
            # once, fixed 3s delays herd the reconnects; jitter desynchronizes.
            _d = random.uniform(2.0, 5.0)
            print(f"[conn] reconnecting in {_d:.1f}s...", flush=True)
            time.sleep(_d)

    def on_message(self, m):
        method = m.get("method")
        if method == "mining.notify":
            self.on_notify(m["params"])
        elif method == "mining.set_target" and m.get("params"):
            self.share_target = m["params"][0]
        elif method == "mining.set_model" and m.get("params"):
            o = m["params"][0]
            self.model_id = f'{o.get("name")}@{o.get("commit")}'
            self.difficulty = int(o.get("difficulty", self.difficulty))
        elif method == "mining.ping":
            self.send({"id": m.get("id"), "method": "mining.pong", "params": []})
        elif m.get("id", 0) >= 3 and ("result" in m or "error" in m):
            # Submit response. Per stratum.cpp: accepted iff no error (result
            # field is NOT checked); error code 21 = stale (normal at block edge).
            if self.dbg_resp < 3:
                print(f"[resp] {json.dumps(m)[:120]}", flush=True); self.dbg_resp += 1
            err = m.get("error")
            if err is None:
                self.accepted += 1
                print("share accepted", flush=True)
            else:
                code = (err[0] if isinstance(err, list) and err else
                        err.get("code", 20) if isinstance(err, dict) else 20)
                if code == 21:
                    self.stale += 1
                else:
                    self.rejected += 1
                    print(f"[submit] REJECTED id={m['id']} code={code}: {err}", flush=True)
        elif m.get("id") == 1:
            print(f"[stratum] subscribed: {str(m.get('result'))[:80]}", flush=True)
        elif m.get("id") == 2:
            print(f"[stratum] authorized: {m.get('result')}", flush=True)

    def on_notify(self, p):
        if len(p) < 7:
            return
        header_prefix = p[1]
        block_target = p[2]
        request_id = int(p[7]) if len(p) > 7 and isinstance(p[7], (int, float)) else 0
        pool_vdf = p[9] if len(p) > 9 and isinstance(p[9], str) and p[9] else self.args.placeholder_vdf
        pool_vdf_tick = int(p[10]) if len(p) > 10 and isinstance(p[10], (int, float)) else self.args.placeholder_tick
        job_id = p[0]
        pow_payload = {
            "block_hash": header_prefix[8:72],
            "vdf": pool_vdf,
            "tick": pool_vdf_tick,
            "request_id": request_id,
            "target": block_target,
            "share_target": self.share_target or ("0" * 2 + "f" * 62),
            "difficulty": self.difficulty,
            "header_prefix": header_prefix,
            "model_identifier": self.model_id,
            "compute_precision": "bf16",
            "proof_version": 3,
        }
        clean = bool(p[6]) if len(p) > 6 else False
        with self.lock:
            if clean:
                self.job_by_req.clear()
            self.job = {"job_id": job_id, "pow": pow_payload, "vdf_tick": pool_vdf_tick}
            self.job_by_req[request_id] = job_id
            if len(self.job_by_req) > 32:
                for k in list(self.job_by_req)[:16]:
                    self.job_by_req.pop(k, None)
        print(f"[job] {job_id} req_id={request_id} height={p[4] if len(p)>4 else '?'} "
              f"vdf={'pool' if (len(p)>9 and p[9]) else 'placeholder'} tick={pool_vdf_tick}", flush=True)

    # ---- vLLM dispatch ----------------------------------------------------
    def dispatch_loop(self, worker_id):
        pi = worker_id
        while not self.stop:
            with self.lock:
                job = self.job
            if not job:
                time.sleep(0.5); continue
            salt = f"{self.salt_base}-{worker_id}-{pi}"   # unique per window
            body = {
                "model": "Qwen/Qwen3-8B",
                "prompt": PROMPT_TMPL.format(salt=salt),
                "max_tokens": 256, "min_tokens": 256, "ignore_eos": True,
                "temperature": 1.0, "top_k": 50, "top_p": 1.0,
                "vllm_xargs": {"pow": job["pow"]},
            }
            pi += self.args.concurrency
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.args.vllm_port}/v1/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.args.vllm_key}"})
                urllib.request.urlopen(req, timeout=600).read()
            except Exception as e:
                print(f"[dispatch] err: {e}", flush=True); time.sleep(1)

    # ---- proof drain + submit --------------------------------------------
    def drain_loop(self):
        ctx = zmq.Context.instance()
        pull = ctx.socket(zmq.PULL)
        pull.bind(f"tcp://127.0.0.1:{self.args.zmq_port}")
        pull.setsockopt(zmq.RCVTIMEO, 1000)
        print(f"[drain] PULL bound on {self.args.zmq_port}", flush=True)
        while not self.stop:
            try:
                raw = pull.recv()
            except zmq.Again:
                continue
            self.proofs += 1
            try:
                self.submit_proof(raw)
            except Exception as e:
                print(f"[drain] submit err: {e}", flush=True)

    @staticmethod
    def _b_cred(pb):
        """B_cred in bits = sum(-log2 chosen_prob) over the window, mirroring the
        chain's credit accounting (pow_v3). Returns None if chosen_probs absent."""
        import math
        n = pb.ChosenProbsLength()
        if not n:
            return None
        total = 0.0
        for i in range(n):
            p = pb.ChosenProbs(i)
            if p <= 0.0:
                p = 1e-12
            elif p > 1.0:
                p = 1.0
            total += -math.log2(p)
        return total

    def _check_model_id(self, pb, raw):
        """Byte-check the stamped model_identifier once: log it and dump the raw
        submitted proof so `strings <dump> | grep 'Qwen/Qwen3-8B@'` can confirm
        the real commit is embedded (not @unknown). Don't trust config, check bytes."""
        mid = pb.ModelIdentifier()
        if isinstance(mid, (bytes, bytearray)):
            mid = mid.decode("utf-8", "replace")
        ok = bool(mid) and mid.endswith("@9c925d64d72725edaf899c6cb9c377fd0709d9c5")
        print(f"[modelid] stamped identifier = {mid!r}  "
              f"{'OK' if ok else 'BAD (expected @9c925d64...)'}", flush=True)
        try:
            with open(self.args.dump_proof, "wb") as fh:
                fh.write(raw)
            print(f"[modelid] wrote submitted proof -> {self.args.dump_proof}  "
                  f"(byte-check: strings {self.args.dump_proof} | "
                  f"grep 'Qwen/Qwen3-8B@')", flush=True)
        except Exception as e:
            print(f"[modelid] dump failed: {e}", flush=True)
        self._id_checked = True

    def submit_proof(self, raw):
        mr = MiningResponse.GetRootAs(bytearray(raw), 0)
        pb = mr.PowBlob()
        if pb is None or pb.HeaderPrefixLength() != 76 or pb.HashLength() != 32:
            return
        if not self._id_checked:
            self._check_model_id(pb, raw)
        # B_cred pre-submit gate: drop windows below the useful floor (default
        # 70 = B_FREE; admission off) so we never submit chain-unwinnable work.
        b_cred = self._b_cred(pb)
        if b_cred is not None and b_cred < self.args.b_floor:
            self.below_floor += 1
            if self.below_floor <= 5 or self.below_floor % 25 == 0:
                print(f"[bcred] DROP b_cred={b_cred:.1f} < {self.args.b_floor} "
                      f"(dropped={self.below_floor})", flush=True)
            return
        req_id = mr.ReqId()
        hp = bytes(pb.HeaderPrefixAsNumpy().tolist())
        digest = bytes(pb.HashAsNumpy().tolist())
        hdr80 = hp + digest[:4]                 # nonce = digest[:4]  (poi.cpp)
        achieved = sha256d(hdr80)[::-1].hex()    # display order
        with self.lock:
            job_id = self.job_by_req.get(req_id) or (self.job["job_id"] if self.job else None)
            vdf_tick = self.job["vdf_tick"] if self.job else 315000
        if not job_id:
            return
        nonce = next(self.nonce)
        proof_b64 = base64.b64encode(raw).decode()
        self.send({"id": next(self.next_id), "method": "mining.submit",
                   "params": [self.args.user, job_id, nonce, proof_b64, achieved, vdf_tick]})
        self.submitted += 1


    def stats_loop(self):
        while not self.stop:
            time.sleep(15)
            print(f"[stats] proofs={self.proofs} submitted={self.submitted} "
                  f"accepted={self.accepted} stale={self.stale} rejected={self.rejected} "
                  f"below_floor={self.below_floor}", flush=True)
            _el=max(time.time()-self._t0,1e-9)
            # v2: label fixed — this is the accepted-share rate (share difficulty
            # gates emission), NOT the generation windows/s. Report both honestly:
            # proofs = every window pulled off the engine (emission-gated), so
            # proofs/el is the nearest observable to end-to-end proof rate.
            print(f"prof-e2e accepted={self.accepted} ({self.accepted/_el:.2f}/s) "
                  f"proofs={self.proofs} ({self.proofs/_el:.2f}/s) in {_el:.0f}s", flush=True)

    def run(self):
        threading.Thread(target=self.supervisor, daemon=True).start()
        # give the first connection a moment before dispatch starts
        time.sleep(3)
        threading.Thread(target=self.drain_loop, daemon=True).start()
        threading.Thread(target=self.stats_loop, daemon=True).start()
        for w in range(self.args.concurrency):
            threading.Thread(target=self.dispatch_loop, args=(w,), daemon=True).start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3399)
    ap.add_argument("--user", default="tc1qdebugonly.vllmbridge")
    ap.add_argument("--password", default="x")
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--vllm-key", default="dbg")
    ap.add_argument("--zmq-port", type=int, default=7911)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--placeholder-vdf", default="00" * 100)
    ap.add_argument("--placeholder-tick", type=int, default=315000)
    # Pre-submit B_cred gate. Chain (pow_v3.h): <45 bits = Invalid, [45,70) =
    # AdmissionRequired (needs a nonce; our path runs POW_V3_ADMISSION_MODE=off
    # so those windows are forfeited/nonce-less), >=70 = Free. With admission
    # off the effective useful floor is B_FREE=70 -> drop anything below it so
    # the re-audit judges only shippable proofs.
    ap.add_argument("--b-floor", type=float, default=70.0)
    ap.add_argument("--dump-proof",
                    default="/data/pow_proofs/submitted_sample.bin")
    args = ap.parse_args()
    print(f"[bridge] pool={args.host}:{args.port} vllm=127.0.0.1:{args.vllm_port} "
          f"zmq_pull={args.zmq_port} conc={args.concurrency}", flush=True)
    Bridge(args).run()


if __name__ == "__main__":
    main()
