"""Adapter for Chroma (single-node server) with an integer metadata field and filtered
search (campaigns).

Engine: chromadb 1.5.9 from PyPI, started with its own CLI,
`chroma run --path <dir> --port <p>`, from a separate virtual environment
(plimsoll/paths.py, chroma_binary). The harness talks to it over the REST API v2 with
plain HTTP, so the experiment's own Python environment is unchanged. One collection "items":
HNSW with space l2 (squared Euclidean distance), max_neighbors 16 and ef_construction
100 (the documented defaults).

Search effort. ef_search is a collection setting; the documentation says it "can be
modified after creation", but in our probe (results/calib/chroma/ef_probe.json) a changed
value is stored and reported at once yet takes effect for queries only after the server
restarts. Restarting to escalate would itself be a maintenance operation, so the
collection is created with ef_search = EF_FIXED, the collection capacity (the default
theta_max of Sec. III-D), and every search runs at that effort; the per-call ef is ignored.

  write    insert = upsert (ids, embeddings, metadatas {"cat": a}), so an insert sets the
           item's state as in the logical model; update = update (ids, embeddings), which
           keeps the metadata; set_attr = update (ids, metadatas); delete = delete (ids).
           One request per write (the server's maximum batch is 5,461 records); the
           reply is the acknowledgement.
  search   query with n_results = k and include = distances; the l2 distance is reported
           squared, as float32, and is compared as is
  lookup   get by ids with embeddings and metadatas
  restart  SIGTERM, wait for exit, start
  kill     SIGKILL (also kills a running vacuum); recover = start
  vacuum   stop the server, `chroma vacuum --path <dir> --force`, start (documented
           maintenance: "Vacuuming shrinks and optimizes your database")
  sync     set the collection's sync_threshold to 100, or back to the default 1,000
           (documented: "determines when to synchronize the index with persistent
           storage ... This field can be modified after creation")
  batch    set batch_size to 10, or back to the default 100 (documented as modifiable)
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

from .. import paths
from ..refengine.client import EngineError, EngineUnavailable

CHROMA_BIN = paths.chroma_binary()
EF_FIXED = 20_000       # campaign collection capacity (CP["N_POOL"])
BASE = "/api/v2/tenants/default_tenant/databases/default_database/collections"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _where(filt):
    if filt is None:
        return None
    if filt[0] == "eq":
        return {"cat": {"$eq": int(filt[1])}}
    return {"cat": {"$lt": int(filt[1])}}


class ChromaF:
    name = "chroma"
    # Operations whose answers must not change; set from the calibration probe
    # (experiments/chroma_calib.py, results/calib/chroma/), not assumed.
    PRESERVING = ()
    ITERATIVE = (None,)
    UNORDERED = ()

    def __init__(self, run_dir, seed, dim=128, root="/var/tmp/plim_chroma"):
        tag = os.path.basename(run_dir.rstrip("/")) or ("run%d" % seed)
        self.root = os.path.join(root, tag)
        self.data = os.path.join(self.root, "data")
        self.dim = dim
        self.proc = None
        self.vac = None
        self.port = None
        self.cid = None
        self.ef = None
        self.sync = 1000
        self.batch = 100

    # ----------------------------------------------------------- processes
    def _launch(self):
        log = open(os.path.join(self.root, "server.log"), "a")
        env = dict(os.environ, ANONYMIZED_TELEMETRY="False")
        self.proc = subprocess.Popen([CHROMA_BIN, "run", "--path", self.data, "--port", str(self.port)],
                                     cwd=self.root, env=env, stdout=log, stderr=log)
        deadline = time.time() + 600
        while True:
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/api/v2/heartbeat" % self.port, timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("chroma exited during start")
            if time.time() > deadline:
                raise RuntimeError("chroma not ready")
            time.sleep(0.1)
        self.ef = None          # the server reads ef_search from the stored configuration
        if self.cid is not None:
            got = self._req("GET", BASE + "/by-id/" + self.cid)
            self.ef = (got.get("configuration_json") or {}).get("hnsw", {}).get("ef_search")

    def start(self, build_seed=1, load_faults=None):
        assert not load_faults, "no injected faults on real engines"
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.data)
        self.port = _free_port()
        self.cid = None
        self._launch()
        out = self._req("POST", BASE, {"name": "items", "configuration": {"hnsw": {
            "space": "l2", "ef_construction": 100, "max_neighbors": 16, "ef_search": EF_FIXED}}})
        self.cid = out["id"]
        self.ef = EF_FIXED

    def signal_kill(self):
        for p in (self.vac, self.proc):
            if p is not None and p.poll() is None:
                os.kill(p.pid, signal.SIGKILL)

    def kill(self):
        self.signal_kill()
        for p in (self.vac, self.proc):
            if p is not None:
                p.wait()
        self.vac = None

    def _term(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=600)

    def clean_restart(self, load_faults=None):
        assert not load_faults
        self._term()
        self._launch()

    def recover(self, load_faults=None):
        assert not load_faults
        self._launch()

    def stop(self):
        try:
            self._term()
        except Exception:
            self.kill()
        shutil.rmtree(self.root, ignore_errors=True)

    # ---------------------------------------------------------------- HTTP
    def _req(self, method, path, body=None, timeout=600):
        url = "http://127.0.0.1:%d%s" % (self.port, path)
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise EngineError("HTTP %d %s: %s" % (e.code, path.rsplit("/", 1)[-1], e.read()[:500])) from e
        except Exception as e:        # connection refused or reset: is the server alive?
            if self.proc is None or self.proc.poll() is not None:
                raise EngineUnavailable(repr(e)) from e
            raise EngineError(repr(e)) from e
        return json.loads(raw) if raw else None

    def _coll(self, verb):
        return BASE + "/" + self.cid + "/" + verb

    # ---------------------------------------------------------- operations
    def write(self, op, ids, vecs=None, attrs=None):
        sid = [str(int(i)) for i in ids]
        if op == "insert":
            body = {"ids": sid, "embeddings": np.asarray(vecs, np.float32).tolist(),
                    "metadatas": [{"cat": int(a)} for a in attrs]}
            return self._req("POST", self._coll("upsert"), body)
        if op == "update":
            return self._req("POST", self._coll("update"),
                             {"ids": sid, "embeddings": np.asarray(vecs, np.float32).tolist()})
        if op == "set_attr":
            return self._req("POST", self._coll("update"),
                             {"ids": sid, "metadatas": [{"cat": int(a)} for a in attrs]})
        if op == "delete":
            return self._req("POST", self._coll("delete"), {"ids": sid})
        raise ValueError(op)

    def _set_hnsw(self, **kw):
        self._req("PUT", BASE + "/" + self.cid, {"new_configuration": {"hnsw": kw}})

    def search(self, Q, k, ef, filt=None, iterative=None, exact=False):
        assert not exact, "Chroma has no exact-search switch"
        Q = np.asarray(Q, dtype=np.float32)
        assert int(ef) <= EF_FIXED
        ids, ds = [], []
        for b in range(0, len(Q), 100):
            body = {"query_embeddings": Q[b:b + 100].tolist(), "n_results": int(k), "include": ["distances"]}
            w = _where(filt)
            if w is not None:
                body["where"] = w
            out = self._req("POST", self._coll("query"), body)
            for i_, d_ in zip(out["ids"], out["distances"]):
                ids.append([int(x) for x in i_])
                # the reply carries float32 values as decimal text; round back to float32
                ds.append([float(np.float32(x)) for x in d_])
        return ids, ds

    def lookup(self, ids):
        ids = [int(i) for i in ids]
        got = {}
        for b in range(0, len(ids), 1000):
            out = self._req("POST", self._coll("get"), {"ids": [str(i) for i in ids[b:b + 1000]],
                                                        "include": ["embeddings", "metadatas"]})
            for i, e, m in zip(out["ids"], out["embeddings"], out["metadatas"]):
                got[int(i)] = (np.asarray(e, dtype=np.float32), int((m or {}).get("cat", -1)))
        return [i in got for i in ids], [got.get(i) for i in ids]

    def indexing_status(self):
        return self._req("GET", self._coll("indexing_status"))

    # ------------------------------------------------- maintenance operations
    def vacuum(self):
        self._term()
        log = open(os.path.join(self.root, "vacuum.log"), "a")
        self.vac = subprocess.Popen([CHROMA_BIN, "vacuum", "--path", self.data, "--force"],
                                    cwd=self.root, stdout=log, stderr=log)
        rc = self.vac.wait()
        self.vac = None
        if rc != 0:
            self._launch()
            raise EngineError("vacuum exited with %d" % rc)
        self._launch()

    def toggle_sync(self):
        self.sync = 100 if self.sync == 1000 else 1000
        self._set_hnsw(sync_threshold=self.sync)

    def toggle_batch(self):
        self.batch = 10 if self.batch == 100 else 100
        self._set_hnsw(batch_size=self.batch)

    def maintenance_ops(self):
        return {"restart": self.clean_restart, "vacuum": self.vacuum, "sync": self.toggle_sync,
                "batch": self.toggle_batch}

    def concurrent_writer(self):
        """A second client: plain HTTP needs no shared state, so a copy will do."""
        import copy
        return copy.copy(self)
