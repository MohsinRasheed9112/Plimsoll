"""Adapter for the reference engine: process control and one RPC per operation.

A call whose connection breaks, because the server was killed, raises
EngineUnavailable; the runner then records the operation's outcome as unknown.
An error reply raises EngineError; the reference contract documents no error as
having no effect, so the runner also records it as unknown.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Client

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EngineUnavailable(Exception):
    pass


class EngineError(Exception):
    pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class RefEngine:
    def __init__(self, data_dir, mode, threads=2, M=16, ef_construction=100, cap=100_000,
                 dim=128, log_path=None):
        self.data_dir = data_dir
        self.mode = mode
        self.args = dict(threads=threads, M=M, ef_construction=ef_construction, cap=cap, dim=dim)
        self.proc = None
        self.conn = None
        self.log_path = log_path
        os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------- processes
    def start(self, build_seed=1, load_faults=None):
        port = _free_port()
        cmd = [sys.executable, "-m", "plimsoll.refengine.server", "--dir", self.data_dir,
               "--mode", self.mode, "--port", str(port), "--build_seed", str(build_seed),
               "--load_faults", json.dumps(load_faults or [])]
        for k, v in self.args.items():
            cmd += ["--" + k, str(v)]
        env = dict(os.environ, PYTHONPATH=CODE_DIR, OMP_NUM_THREADS="1")
        log = open(self.log_path, "a") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, env=env, text=True)
        line = self.proc.stdout.readline()
        if line.strip() != "READY":
            raise RuntimeError("engine failed to start: %r" % line)
        deadline = time.time() + 30
        while True:
            try:
                self.conn = Client(("127.0.0.1", port), authkey=b"plimsoll")
                break
            except ConnectionRefusedError:
                if time.time() > deadline:
                    raise
                time.sleep(0.05)

    def signal_kill(self):
        """SIGKILL from another thread; leaves the connection to the caller's thread."""
        if self.proc is not None and self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGKILL)

    def kill(self):
        """SIGKILL: the process ends at once, with no flush."""
        if self.proc is not None and self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGKILL)
        if self.proc is not None:
            self.proc.wait()
        self._close()

    def clean_restart(self, load_faults=None):
        self.call("shutdown")
        self.proc.wait(timeout=120)
        self._close()
        self.start(load_faults=load_faults)

    def recover(self, load_faults=None):
        self.start(load_faults=load_faults)

    def stop(self):
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.call("shutdown")
                self.proc.wait(timeout=120)
        except (EngineUnavailable, EngineError):
            self.kill()
        self._close()

    def _close(self):
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None

    # ------------------------------------------------------------------- RPC
    def call(self, rpc, **kw):
        if self.conn is None:
            raise EngineUnavailable("not connected")
        try:
            self.conn.send((rpc, kw))
            status, res = self.conn.recv()
        except (EOFError, ConnectionError, BrokenPipeError, OSError) as e:
            self._close()
            raise EngineUnavailable(repr(e)) from e
        if status != "ok":
            raise EngineError(res)
        return res

    def write(self, op, ids, vecs=None):
        return self.call("write", op=op, ids=list(map(int, ids)), vecs=vecs)

    def search(self, Q, k, ef):
        return self.call("search", Q=Q, k=int(k), ef=int(ef))

    def lookup(self, ids):
        return self.call("lookup", ids=list(map(int, ids)))

    def flush(self):
        return self.call("flush")

    def rebuild(self, seed):
        return self.call("rebuild", seed=int(seed))

    def arm_desync(self, count):
        return self.call("arm_desync", count=int(count))
