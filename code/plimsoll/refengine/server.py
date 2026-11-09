"""Reference engine server for Plimsoll experiments.

One process holds a row store and, in mode "hnsw", an hnswlib index. Every write is
appended to a write-ahead log (WAL) and fsynced before the client receives an
acknowledgement. A flush writes an atomic snapshot and truncates the WAL. Recovery
loads the snapshot and replays the complete WAL records; a partial record at the
tail is ignored.

Faults are injected only through explicit options ("load faults" applied right after
recovery, or "armed" faults set by RPC). Every injected effect is written to
faults.jsonl in the data directory. The tester never reads that file; it is used only
to evaluate detection.

Contract of this engine (stored in plimsoll/contracts.py):
  exact: scores are squared L2 distances from row data in float64 by the expansion
         formula; order follows the reported score; result size min(k, n).
  hnsw:  scores are squared L2 distances from index data in float32 by the direct
         formula; order follows the reported score; result size not promised.
  both:  an acknowledged write is visible at once and durable; no error is documented
         as having no effect.
"""
import argparse
import json
import os
import pickle
import re
import struct
import sys
import time
import zlib
from multiprocessing.connection import Listener

import numpy as np

NEVER, LIVE, DELETED = 0, 1, 2
_HDR = struct.Struct("<II")  # payload length, crc32


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Engine:
    def __init__(self, a):
        self.dir = a.dir
        self.mode = a.mode
        self.cap = a.cap
        self.dim = a.dim
        self.M = a.M
        self.efc = a.ef_construction
        self.threads = a.threads
        self.build_seed = a.build_seed
        self.vecs = np.zeros((self.cap, self.dim), np.float32)
        self.state = np.zeros(self.cap, np.int8)
        self.index = None
        self.seq = 0            # last WAL sequence number applied
        self.snap_seq = 0       # sequence number covered by the snapshot
        self.gen = 0            # snapshot generation
        self.armed_desync = 0   # remaining vector updates whose index part is skipped
        self.wal_path = os.path.join(self.dir, "wal.log")
        self.fault_path = os.path.join(self.dir, "faults.jsonl")
        self._recover()
        self.wal_fd = os.open(self.wal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    # ------------------------------------------------------------------ index
    def _new_index(self, seed):
        import hnswlib
        idx = hnswlib.Index(space="l2", dim=self.dim)
        idx.init_index(max_elements=self.cap, M=self.M, ef_construction=self.efc,
                       random_seed=int(seed))
        idx.set_num_threads(self.threads)
        return idx

    def _index_add(self, ids, vecs):
        if self.index is not None and len(ids):
            self.index.add_items(vecs, np.asarray(ids, dtype=np.int64), num_threads=self.threads)

    def _index_delete(self, ids):
        if self.index is not None:
            for i in ids:
                try:
                    self.index.mark_deleted(int(i))
                except RuntimeError:
                    pass  # entry already absent from the index (only after an injected fault)

    # ---------------------------------------------------------------- recovery
    def _recover(self):
        meta_p = os.path.join(self.dir, "meta.json")
        if os.path.exists(meta_p):
            meta = json.load(open(meta_p))
            self.gen = int(meta["gen"])
            z = np.load(os.path.join(self.dir, "rows.%d.npz" % self.gen))
            self.vecs[:] = z["vecs"]
            self.state[:] = z["state"]
            self.snap_seq = self.seq = int(meta["seq"])
            if self.mode == "hnsw":
                import hnswlib
                self.index = hnswlib.Index(space="l2", dim=self.dim)
                self.index.load_index(os.path.join(self.dir, "index.%d.bin" % self.gen),
                                      max_elements=self.cap)
                self.index.set_num_threads(self.threads)
        elif self.mode == "hnsw":
            self.index = self._new_index(self.build_seed)
        # replay complete WAL records after the snapshot
        if os.path.exists(self.wal_path):
            data = open(self.wal_path, "rb").read()
            pos = 0
            while pos + _HDR.size <= len(data):
                n, crc = _HDR.unpack_from(data, pos)
                body = data[pos + _HDR.size: pos + _HDR.size + n]
                if len(body) < n or zlib.crc32(body) != crc:
                    break  # partial or torn tail record: never acknowledged
                rec = pickle.loads(body)
                if rec[0] > self.seq:
                    self._apply(rec[1], rec[2], rec[3], replay=True)
                    self.seq = rec[0]
                pos += _HDR.size + n

    # ------------------------------------------------------------------ writes
    def _log(self, op, ids, vecs):
        self.seq += 1
        body = pickle.dumps((self.seq, op, ids, vecs), protocol=4)
        os.write(self.wal_fd, _HDR.pack(len(body), zlib.crc32(body)) + body)
        os.fsync(self.wal_fd)

    def _apply(self, op, ids, vecs, replay=False):
        ids = np.asarray(ids, dtype=np.int64)
        if op == "insert":
            self.vecs[ids] = vecs
            self.state[ids] = LIVE
            self._index_add(ids, vecs)
        elif op == "update":
            self.vecs[ids] = vecs
            if self.armed_desync > 0 and not replay and self.index is not None:
                skip = min(self.armed_desync, len(ids))
                self.armed_desync -= skip
                self._fault_log("desync", ids[:skip].tolist())
                self._index_add(ids[skip:], vecs[skip:])
            else:
                self._index_add(ids, vecs)
        elif op == "delete":
            self.state[ids] = DELETED
            self._index_delete(ids)
        else:
            raise ValueError(op)

    def write(self, op, ids, vecs=None):
        ids = np.asarray(ids, dtype=np.int64)
        if op in ("insert",):
            if np.any(self.state[ids] == LIVE):
                raise ValueError("insert of a live id")
        if op in ("update", "delete"):
            if np.any(self.state[ids] != LIVE):
                raise ValueError(op + " of an id that is not live")
        if vecs is not None:
            vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        self._log(op, ids, vecs)          # durable before the acknowledgement
        self._apply(op, ids, vecs)
        return self.seq

    # ------------------------------------------------------------------- reads
    def search(self, Q, k, ef):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        if self.mode == "exact":
            return self._search_exact(Q, k)
        self.index.set_ef(int(ef))
        try:
            lab, dist = self.index.knn_query(Q, k=k, num_threads=self.threads)
            return [l.tolist() for l in lab], [d.tolist() for d in dist]
        except RuntimeError:
            out_l, out_d = [], []
            for q in Q:
                for kk in range(k, 0, -1):
                    try:
                        l, d = self.index.knn_query(q[None, :], k=kk, num_threads=1)
                        out_l.append(l[0].tolist())
                        out_d.append(d[0].tolist())
                        break
                    except RuntimeError:
                        continue
                else:
                    out_l.append([])
                    out_d.append([])
            return out_l, out_d

    def _search_exact(self, Q, k):
        live = np.flatnonzero(self.state == LIVE)
        X = self.vecs[live].astype(np.float64)
        xn = np.einsum("ij,ij->i", X, X)
        out_l, out_d = [], []
        for s in range(0, len(Q), 200):
            q = Q[s:s + 200].astype(np.float64)
            D = np.einsum("ij,ij->i", q, q)[:, None] + xn[None, :] - 2.0 * (q @ X.T)
            kk = min(k, len(live))
            for row in D:
                if kk == 0:
                    out_l.append([])
                    out_d.append([])
                    continue
                part = np.argpartition(row, kk - 1)[:kk] if kk < len(row) else np.arange(len(row))
                order = sorted(part.tolist(), key=lambda j: (row[j], live[j]))
                out_l.append([int(live[j]) for j in order])
                out_d.append([float(row[j]) for j in order])
        return out_l, out_d

    def lookup(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        live = self.state[ids] == LIVE
        return live.tolist(), [self.vecs[i].copy() if l else None for i, l in zip(ids, live)]

    # ------------------------------------------------------------- maintenance
    def flush(self):
        """Atomic snapshot: generation files first, then meta.json as the commit point."""
        gen = self.gen + 1
        buf = _NpzBuffer(self.vecs, self.state)
        _atomic_write_bytes(os.path.join(self.dir, "rows.%d.npz" % gen), buf.data)
        if self.index is not None:
            final = os.path.join(self.dir, "index.%d.bin" % gen)
            tmp = final + ".tmp"
            self.index.save_index(tmp)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp, final)
        _fsync_dir(self.dir)
        _atomic_write_bytes(os.path.join(self.dir, "meta.json"),
                            json.dumps({"seq": self.seq, "gen": gen}).encode())
        _fsync_dir(self.dir)
        for name in os.listdir(self.dir):
            m = re.match(r"^(rows|index)\.(\d+)\.(npz|bin)$", name)
            if m and int(m.group(2)) != gen:
                os.remove(os.path.join(self.dir, name))
        self.gen = gen
        self.snap_seq = self.seq
        os.ftruncate(self.wal_fd, 0)
        os.fsync(self.wal_fd)
        return self.seq

    def rebuild(self, seed):
        if self.mode == "hnsw":
            live = np.flatnonzero(self.state == LIVE)
            idx = self._new_index(seed)
            idx.add_items(self.vecs[live], live.astype(np.int64), num_threads=self.threads)
            self.index = idx
        return self.flush()

    # ------------------------------------------------------------------ faults
    def _fault_log(self, kind, ids):
        with open(self.fault_path, "a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, "ids": [int(i) for i in ids]}) + "\n")

    def apply_load_faults(self, faults):
        for fl in faults or []:
            rng = np.random.default_rng(int(fl["seed"]))
            if fl["kind"] == "lost_entries" and self.index is not None:
                live = np.flatnonzero(self.state == LIVE)
                n = int(round(fl["frac"] * len(live)))
                ids = np.sort(rng.choice(live, size=n, replace=False))
                for i in ids:
                    self.index.mark_deleted(int(i))   # rows stay live: entries lost from the index
                self._fault_log("lost_entries", ids.tolist())
            elif fl["kind"] == "resurrect":
                dead = np.flatnonzero(self.state == DELETED)
                n = int(round(fl["frac"] * len(dead)))
                ids = np.sort(rng.choice(dead, size=n, replace=False))
                self.state[ids] = LIVE                # tombstones lost in rows ...
                if self.index is not None:
                    for i in ids:
                        self.index.unmark_deleted(int(i))  # ... and in the index
                self._fault_log("resurrect", ids.tolist())
            else:
                raise ValueError(fl)


class _NpzBuffer:
    def __init__(self, vecs, state):
        import io
        b = io.BytesIO()
        np.savez(b, vecs=vecs, state=state)
        self.data = b.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--mode", choices=["exact", "hnsw"], required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--authkey", default="plimsoll")
    p.add_argument("--cap", type=int, default=100_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef_construction", type=int, default=100)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--build_seed", type=int, default=1)
    p.add_argument("--load_faults", default="[]")
    a = p.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    eng = Engine(a)
    eng.apply_load_faults(json.loads(a.load_faults))
    listener = Listener(("127.0.0.1", a.port), authkey=a.authkey.encode())
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    while True:
        conn = listener.accept()
        try:
            while True:
                try:
                    op, kw = conn.recv()
                except (EOFError, ConnectionError):
                    break
                try:
                    if op == "ping":
                        res = {"seq": eng.seq, "mode": eng.mode}
                    elif op == "write":
                        res = eng.write(kw["op"], kw["ids"], kw.get("vecs"))
                    elif op == "search":
                        res = eng.search(kw["Q"], kw["k"], kw["ef"])
                    elif op == "lookup":
                        res = eng.lookup(kw["ids"])
                    elif op == "flush":
                        res = eng.flush()
                    elif op == "rebuild":
                        res = eng.rebuild(kw["seed"])
                    elif op == "arm_desync":
                        eng.armed_desync = int(kw["count"])
                        res = eng.armed_desync
                    elif op == "count_live":
                        res = int(np.count_nonzero(eng.state == LIVE))
                    elif op == "shutdown":
                        eng.flush()
                        conn.send(("ok", eng.seq))
                        conn.close()
                        os._exit(0)
                    else:
                        raise ValueError("unknown op " + str(op))
                    conn.send(("ok", res))
                except Exception as e:  # reported to the client as an engine error
                    conn.send(("err", repr(e)))
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
