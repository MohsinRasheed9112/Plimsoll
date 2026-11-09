"""Adapter for pgvector on PostgreSQL: one server per run, killed with SIGKILL.

Engine: PostgreSQL 16 (Ubuntu 24.04 package) with pgvector 0.8.6 built from source.
Table items(id bigint primary key, embedding vector(d)). The HNSW index
(vector_l2_ops; m and ef_construction default to pgvector's 16 and 64) is created
after the initial load. Autovacuum is off, so the index changes only through the
workload's own operations.

  write    one transaction per batch; the commit is the acknowledgement
  search   ORDER BY embedding <-> q LIMIT k under hnsw.ef_search = ef; the reported
           L2 distance is squared in float64 for the oracles
  lookup   by primary key (heap)
  flush    CHECKPOINT
  restart  pg_ctl stop -m fast, then start
  rebuild  REINDEX INDEX (a new build from the table rows)
  kill     SIGKILL to the postmaster and every child; recover = start (WAL replay)
"""
import os
import shutil
import signal
import socket
import subprocess
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .. import paths
from ..refengine.client import EngineError, EngineUnavailable

PG_BIN = paths.PG_BIN
PG_USER = paths.PG_USER     # PostgreSQL refuses to run as root; the harness runs it as this user
PG_ROOT = "/var/tmp/plim_pg"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _np32(v):
    v = v.to_numpy() if hasattr(v, "to_numpy") else v
    return np.asarray(v, dtype=np.float32)


def _as_user(cmd):
    return ["runuser", "-u", PG_USER, "--"] + cmd


class PgVector:
    name = "pgvector"

    def __init__(self, run_dir, seed, dim=128, m=16, ef_construction=64, log_path=None):
        tag = os.path.basename(run_dir.rstrip("/")) or ("run%d" % seed)
        self.root = os.path.join(PG_ROOT, tag)
        self.pgdata = os.path.join(self.root, "data")
        self.dim, self.m, self.efc = dim, m, ef_construction
        self.port = None
        self.conn = None
        self.started_once = False
        self.ef = None
        self.log = os.path.join(self.root, "server.log")

    # ----------------------------------------------------------- processes
    def _sh(self, cmd, timeout=600):
        r = subprocess.run(_as_user(cmd), capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError("%s failed: %s %s" % (cmd[0], r.stdout[-2000:], r.stderr[-2000:]))
        return r

    def _initdb(self):
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root)
        shutil.chown(self.root, PG_USER, PG_USER)
        self._sh([PG_BIN + "/initdb", "-D", self.pgdata, "-U", "postgres", "--auth=trust",
                  "-E", "UTF8", "--no-instructions"])
        self.port = _free_port()
        conf = "\n".join([
            "port = %d" % self.port, "listen_addresses = '127.0.0.1'",
            "unix_socket_directories = '%s'" % self.root,
            "shared_buffers = 512MB", "maintenance_work_mem = 1GB",
            "max_parallel_maintenance_workers = 1", "max_parallel_workers_per_gather = 0",
            "max_wal_size = 8GB", "checkpoint_timeout = 30min",
            "fsync = on", "synchronous_commit = on", "full_page_writes = on",
            "autovacuum = off", "jit = off", "log_min_messages = warning", ""])
        with open(os.path.join(self.pgdata, "postgresql.conf"), "a") as f:
            f.write(conf)

    def _pg_start(self):
        self._sh([PG_BIN + "/pg_ctl", "-D", self.pgdata, "-l", self.log, "-w", "-t", "600", "start"])
        self._connect()

    def _connect(self):
        self.conn = psycopg.connect(host="127.0.0.1", port=self.port, user="postgres",
                                    dbname="postgres", autocommit=True)
        if self.started_once:
            register_vector(self.conn)
            self._session()

    def _session(self):
        with self.conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")   # the index path is under test
            cur.execute("SET jit = off")
        self.ef = None

    def start(self, build_seed=1, load_faults=None):
        assert not load_faults, "no injected faults on real engines"
        self._initdb()
        self._pg_start()
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION vector")
            cur.execute("CREATE TABLE items (id bigint PRIMARY KEY, embedding vector(%d))" % self.dim)
        self.started_once = True
        register_vector(self.conn)
        self._session()

    def after_initial_load(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE INDEX items_embedding_idx ON items USING hnsw (embedding vector_l2_ops) "
                        "WITH (m = %d, ef_construction = %d)" % (self.m, self.efc))
            cur.execute("ANALYZE items")
            plan = "\n".join(r[0] for r in cur.execute(
                "EXPLAIN SELECT id FROM items ORDER BY embedding <-> %b LIMIT 10",
                (np.zeros(self.dim, np.float32),)).fetchall())
        if "items_embedding_idx" not in plan:
            raise RuntimeError("search plan does not use the HNSW index:\n" + plan)

    def _pids(self):
        try:
            with open(os.path.join(self.pgdata, "postmaster.pid")) as f:
                pm = int(f.readline().strip())
        except (OSError, ValueError):
            return []
        kids = subprocess.run(["pgrep", "-P", str(pm)], capture_output=True, text=True).stdout.split()
        return [pm] + [int(k) for k in kids]

    def signal_kill(self):
        for pid in self._pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def kill(self):
        pids = self._pids()
        self.signal_kill()
        for pid in pids:
            for _ in range(600):
                if not os.path.exists("/proc/%d" % pid):
                    break
                time.sleep(0.05)
        self._close()

    def clean_restart(self, load_faults=None):
        assert not load_faults
        self._close()
        self._sh([PG_BIN + "/pg_ctl", "-D", self.pgdata, "-m", "fast", "-w", "-t", "600", "stop"])
        self._pg_start()

    def recover(self, load_faults=None):
        assert not load_faults
        self._close()
        self._pg_start()

    def stop(self):
        self._close()
        if os.path.exists(os.path.join(self.pgdata, "postmaster.pid")):
            try:
                self._sh([PG_BIN + "/pg_ctl", "-D", self.pgdata, "-m", "fast", "-w", "-t", "600", "stop"])
            except Exception:
                self.kill()
        shutil.rmtree(self.root, ignore_errors=True)

    def _close(self):
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None

    # ---------------------------------------------------------- operations
    def _guard(self, fn):
        if self.conn is None:
            raise EngineUnavailable("not connected")
        try:
            return fn()
        except psycopg.OperationalError as e:
            self._close()
            raise EngineUnavailable(repr(e)) from e
        except psycopg.Error as e:
            raise EngineError(repr(e)) from e

    def write(self, op, ids, vecs=None):
        ids = [int(i) for i in ids]

        def go():
            with self.conn.transaction():
                with self.conn.cursor() as cur:
                    if op == "insert":
                        cur.executemany("INSERT INTO items (id, embedding) VALUES (%s, %b)",
                                        list(zip(ids, vecs)))
                    elif op == "update":
                        cur.executemany("UPDATE items SET embedding = %b WHERE id = %s",
                                        list(zip(vecs, ids)))
                    elif op == "delete":
                        cur.execute("DELETE FROM items WHERE id = ANY(%s)", (ids,))
                    else:
                        raise ValueError(op)
        return self._guard(go)

    def _set_ef(self, cur, ef):
        if self.ef != ef:
            cur.execute("SET hnsw.ef_search = %d" % int(ef))
            self.ef = ef

    def search(self, Q, k, ef):
        def go():
            ids, ds = [], []
            with self.conn.cursor() as cur:
                self._set_ef(cur, ef)
                for q in np.asarray(Q, dtype=np.float32):
                    rows = cur.execute("SELECT id, embedding <-> %b FROM items "
                                       "ORDER BY embedding <-> %b LIMIT %s",
                                       (q, q, int(k)), prepare=True).fetchall()
                    ids.append([int(r[0]) for r in rows])
                    ds.append([float(r[1]) * float(r[1]) for r in rows])
            return ids, ds
        return self._guard(go)

    def exact_search(self, Q, k):
        """The engine's exact path (no index), for baseline B2."""
        def go():
            ids, ds = [], []
            with self.conn.cursor() as cur:
                cur.execute("SET enable_indexscan = off")
                cur.execute("SET enable_seqscan = on")
                try:
                    for q in np.asarray(Q, dtype=np.float32):
                        rows = cur.execute("SELECT id, embedding <-> %b FROM items "
                                           "ORDER BY embedding <-> %b LIMIT %s", (q, q, int(k))).fetchall()
                        ids.append([int(r[0]) for r in rows])
                        ds.append([float(r[1]) * float(r[1]) for r in rows])
                finally:
                    cur.execute("SET enable_indexscan = on")
                    cur.execute("SET enable_seqscan = off")
            return ids, ds
        return self._guard(go)

    def lookup(self, ids):
        ids = [int(i) for i in ids]

        def go():
            with self.conn.cursor() as cur:
                rows = cur.execute("SELECT id, embedding FROM items WHERE id = ANY(%s)", (ids,)).fetchall()
            got = {int(r[0]): _np32(r[1]) for r in rows}
            return [i in got for i in ids], [got.get(i) for i in ids]
        return self._guard(go)

    def flush(self):
        return self._guard(lambda: self.conn.execute("CHECKPOINT"))

    def rebuild(self, seed):
        return self._guard(lambda: self.conn.execute("REINDEX INDEX items_embedding_idx"))
