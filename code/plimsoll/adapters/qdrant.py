"""Adapter for Qdrant: one server process per run, killed with SIGKILL.

Engine: the official Qdrant release binary (x86_64-unknown-linux-gnu), driven through
qdrant-client over gRPC. One collection "items": dense vectors, Euclid distance,
HNSW m = 16 and ef_construct = 100 (Qdrant defaults); other settings default.

  write    upsert or delete with wait = true; the reply is the acknowledgement
  search   query_batch_points with hnsw_ef = ef; the reported Euclidean score is squared
           in float64 for the oracles
  lookup   retrieve by ID with vectors
  flush    create a collection snapshot (read-only; Qdrant has no flush call)
  restart  SIGTERM, wait for exit, start
  rebuild  change hnsw ef_construct to 101 and back to 100; each change rebuilds the
           segments in the background (documented), and the run waits for both
  kill     SIGKILL; recover = start (WAL replay)
  quiesce  before each measurement, wait until the collection status is green
           (a grey status is triggered with an empty optimizer update, as documented)
"""
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from .. import paths
from ..refengine.client import EngineError, EngineUnavailable

QDRANT_BIN = paths.qdrant_binary()
COLL = "items"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Qdrant:
    name = "qdrant"

    def __init__(self, run_dir, seed, dim=128, m=16, ef_construct=100, root="/var/tmp/plim_qd"):
        tag = os.path.basename(run_dir.rstrip("/")) or ("run%d" % seed)
        self.root = os.path.join(root, tag)
        self.dim, self.m, self.efc = dim, m, ef_construct
        self.proc = None
        self.client = None
        self.http = self.grpc = None

    # ----------------------------------------------------------- processes
    def _launch(self):
        env = dict(os.environ,
                   QDRANT__STORAGE__STORAGE_PATH=os.path.join(self.root, "storage"),
                   QDRANT__STORAGE__SNAPSHOTS_PATH=os.path.join(self.root, "snapshots"),
                   QDRANT__SERVICE__HOST="127.0.0.1",
                   QDRANT__SERVICE__HTTP_PORT=str(self.http),
                   QDRANT__SERVICE__GRPC_PORT=str(self.grpc),
                   QDRANT__TELEMETRY_DISABLED="true", QDRANT__LOG_LEVEL="WARN")
        log = open(os.path.join(self.root, "server.log"), "a")
        self.proc = subprocess.Popen([getattr(self, "binary", QDRANT_BIN)], cwd=self.root, env=env,
                                     stdout=log, stderr=log)
        deadline = time.time() + 600
        while True:
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/readyz" % self.http, timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("qdrant exited during start")
            if time.time() > deadline:
                raise RuntimeError("qdrant not ready")
            time.sleep(0.1)
        self.client = QdrantClient(host="127.0.0.1", port=self.http, grpc_port=self.grpc,
                                   prefer_grpc=True, timeout=600)

    def start(self, build_seed=1, load_faults=None):
        assert not load_faults, "no injected faults on real engines"
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root)
        self.http, self.grpc = _free_port(), _free_port()
        self._launch()
        self.client.create_collection(
            COLL, vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.EUCLID),
            hnsw_config=models.HnswConfigDiff(m=self.m, ef_construct=self.efc))

    def signal_kill(self):
        if self.proc is not None and self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGKILL)

    def kill(self):
        self.signal_kill()
        if self.proc is not None:
            self.proc.wait()
        self._close()

    def clean_restart(self, load_faults=None):
        assert not load_faults
        self._close()
        self.proc.send_signal(signal.SIGTERM)
        self.proc.wait(timeout=600)
        self._launch()

    def recover(self, load_faults=None):
        assert not load_faults
        self._close()
        self._launch()

    def stop(self):
        self._close()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                self.kill()
        shutil.rmtree(self.root, ignore_errors=True)

    def _close(self):
        try:
            if self.client is not None:
                self.client.close()
        except Exception:
            pass
        self.client = None

    # ---------------------------------------------------------- operations
    def _guard(self, fn):
        if self.client is None:
            raise EngineUnavailable("not connected")
        try:
            return fn()
        except UnexpectedResponse as e:
            raise EngineError(repr(e)) from e
        except Exception as e:  # gRPC errors when the server dies, timeouts
            if self.proc is None or self.proc.poll() is not None:
                self._close()
                raise EngineUnavailable(repr(e)) from e
            raise EngineError(repr(e)) from e

    def write(self, op, ids, vecs=None):
        ids = [int(i) for i in ids]

        def go():
            if op in ("insert", "update"):
                self.client.upsert(COLL, points=models.Batch(
                    ids=ids, vectors=np.asarray(vecs, dtype=np.float32).tolist()), wait=True)
            elif op == "delete":
                self.client.delete(COLL, points_selector=models.PointIdsList(points=ids), wait=True)
            else:
                raise ValueError(op)
        return self._guard(go)

    def search(self, Q, k, ef, exact=False):
        Q = np.asarray(Q, dtype=np.float32)

        def go():
            ids, ds = [], []
            for b in range(0, len(Q), 100):
                reqs = [models.QueryRequest(query=q.tolist(), limit=int(k),
                                            params=models.SearchParams(hnsw_ef=int(ef), exact=exact),
                                            with_payload=False, with_vector=False)
                        for q in Q[b:b + 100]]
                for r in self.client.query_batch_points(COLL, requests=reqs):
                    ids.append([int(p.id) for p in r.points])
                    ds.append([float(p.score) * float(p.score) for p in r.points])
            return ids, ds
        return self._guard(go)

    def exact_search(self, Q, k):
        """The engine's exact path (exact = true), for baseline B2."""
        return self.search(Q, k, ef=1, exact=True)

    def lookup(self, ids):
        ids = [int(i) for i in ids]

        def go():
            got = {}
            for b in range(0, len(ids), 1000):
                for r in self.client.retrieve(COLL, ids=ids[b:b + 1000], with_vectors=True,
                                              with_payload=False):
                    got[int(r.id)] = np.asarray(r.vector, dtype=np.float32)
            return [i in got for i in ids], [got.get(i) for i in ids]
        return self._guard(go)

    def flush(self):
        return self._guard(lambda: self.client.create_snapshot(COLL, wait=True))

    def rebuild(self, seed):
        def go():
            for efc in (self.efc + 1, self.efc):
                self.client.update_collection(COLL, hnsw_config=models.HnswConfigDiff(ef_construct=efc))
                self._wait_green()
        return self._guard(go)

    def _wait_green(self, timeout=3600):
        deadline = time.time() + timeout
        time.sleep(0.5)
        while True:
            info = self.client.get_collection(COLL)
            st = info.status
            if st == models.CollectionStatus.GREEN:
                return info
            if st == models.CollectionStatus.GREY:
                self.client.update_collection(COLL, optimizers_config=models.OptimizersConfigDiff())
            if st == models.CollectionStatus.RED:
                raise EngineError("collection status red")
            if time.time() > deadline:
                raise EngineError("collection not green after %ds" % timeout)
            time.sleep(0.5)

    def quiesce(self):
        return self._guard(self._wait_green)

    def after_initial_load(self):
        self.quiesce()
