"""Adapter for ClickHouse with a vector similarity index (campaigns).

Engine: the official ClickHouse static binary, one server per run, killed with
SIGKILL. Table items(id UInt64, vec Array(Float32), cat Int32) with a
vector_similarity('hnsw', 'L2Distance', d, <quantization>, 16, 100) index, MergeTree
ordered by id, block number and offset columns on (needed by lightweight UPDATE).
Queries go over the HTTP interface.

  write    insert: INSERT (one part); delete: lightweight DELETE;
           update/set_attr: lightweight UPDATE (beta), one statement per batch with
           transform(); the reply is the acknowledgement
  search   ORDER BY L2Distance(vec, q) LIMIT k with hnsw_candidate_list_size_for_search;
           the reported distance is squared in float64 for the oracles
  lookup   by id
  ops      optimize (OPTIMIZE TABLE FINAL), materialize (MATERIALIZE INDEX), attr_index
           (minmax skipping index on cat), restart (SIGTERM, start)
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .. import paths
from ..refengine.client import EngineError, EngineUnavailable

CH_BIN = paths.clickhouse_binary()

CONFIG = """<clickhouse>
  <logger><level>warning</level><log>{root}/log/server.log</log><errorlog>{root}/log/err.log</errorlog></logger>
  <http_port>{http}</http_port>
  <tcp_port>{tcp}</tcp_port>
  <listen_host>127.0.0.1</listen_host>
  <path>{root}/data/</path>
  <tmp_path>{root}/tmp/</tmp_path>
  <user_files_path>{root}/user_files/</user_files_path>
  <format_schema_path>{root}/format_schemas/</format_schema_path>
  <mark_cache_size>268435456</mark_cache_size>
  <max_server_memory_usage_to_ram_ratio>0.5</max_server_memory_usage_to_ram_ratio>
  <users_config>{root}/users.xml</users_config>
  <default_profile>default</default_profile>
  <default_database>default</default_database>
</clickhouse>
"""
USERS = """<clickhouse>
  <profiles><default></default></profiles>
  <users><default><password></password><networks><ip>127.0.0.1</ip></networks><profile>default</profile><quota>default</quota><access_management>1</access_management></default></users>
  <quotas><default></default></quotas>
</clickhouse>
"""


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _vec_sql(v):
    return "CAST([" + ",".join(repr(float(x)) for x in np.asarray(v, dtype=np.float32)) + "] AS Array(Float32))"


class ClickHouseF:
    name = "clickhouse"
    PRESERVING = ("restart",)
    ITERATIVE = (None,)
    UNORDERED = ()

    def __init__(self, run_dir, seed, dim=128, quantization="f32", root="/var/tmp/plim_ch"):
        tag = os.path.basename(run_dir.rstrip("/")) or ("run%d" % seed)
        self.root = os.path.join(root, tag)
        self.dim, self.quant = dim, quantization
        self.proc = None
        self.http = self.tcp = None
        self.binary = CH_BIN
        self.up = False

    # ----------------------------------------------------------- processes
    def _launch(self):
        log = open(os.path.join(self.root, "stdout.log"), "a")
        self.proc = subprocess.Popen([self.binary, "server", "--config-file", os.path.join(self.root, "config.xml")],
                                     cwd=self.root, stdout=log, stderr=log)
        deadline = time.time() + 300
        while True:
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/ping" % self.http, timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("clickhouse exited during start")
            if time.time() > deadline:
                raise RuntimeError("clickhouse not ready")
            time.sleep(0.1)
        self.up = True

    def start(self, build_seed=1, load_faults=None):
        assert not load_faults
        shutil.rmtree(self.root, ignore_errors=True)
        for d in ("data", "tmp", "user_files", "format_schemas", "log"):
            os.makedirs(os.path.join(self.root, d))
        self.http, self.tcp = _free_port(), _free_port()
        with open(os.path.join(self.root, "config.xml"), "w") as f:
            f.write(CONFIG.format(root=self.root, http=self.http, tcp=self.tcp))
        with open(os.path.join(self.root, "users.xml"), "w") as f:
            f.write(USERS)
        self._launch()
        self._q("CREATE TABLE items (id UInt64, vec Array(Float32), cat Int32, "
                "INDEX vidx vec TYPE vector_similarity('hnsw', 'L2Distance', %d, '%s', 16, 100)) "
                "ENGINE = MergeTree ORDER BY id SETTINGS enable_block_number_column = 1, "
                "enable_block_offset_column = 1" % (self.dim, self.quant))

    def signal_kill(self):
        if self.proc is not None and self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGKILL)

    def kill(self):
        self.signal_kill()
        if self.proc is not None:
            self.proc.wait()
        self.up = False

    def clean_restart(self, load_faults=None):
        assert not load_faults
        self.proc.send_signal(signal.SIGTERM)
        self.proc.wait(timeout=600)
        self.up = False
        self._launch()

    def recover(self, load_faults=None):
        assert not load_faults
        self._launch()

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                self.kill()
        self.up = False
        shutil.rmtree(self.root, ignore_errors=True)

    # ---------------------------------------------------------------- HTTP
    def _q(self, sql, data=None, settings=None):
        if not self.up:
            raise EngineUnavailable("not running")
        params = dict(settings or {})
        if data is not None:          # INSERT: statement in the URL, rows in the body
            params["query"] = sql
            body = data.encode() if isinstance(data, str) else data
        else:                         # other statements can be long: send them in the body
            body = sql.encode()
        url = "http://127.0.0.1:%d/?%s" % (self.http, urllib.parse.urlencode(params))
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            raise EngineError("HTTP %d: %s" % (e.code, e.read().decode()[:500])) from e
        except Exception as e:
            if self.proc is None or self.proc.poll() is not None:
                self.up = False
                raise EngineUnavailable(repr(e)) from e
            raise EngineError(repr(e)) from e

    # ---------------------------------------------------------- operations
    def write(self, op, ids, vecs=None, attrs=None):
        ids = [int(i) for i in ids]
        if op == "insert":
            rows = "\n".join(json.dumps(dict(id=i, vec=[float(x) for x in np.asarray(v, np.float32)], cat=int(a)))
                             for i, v, a in zip(ids, vecs, attrs))
            return self._q("INSERT INTO items FORMAT JSONEachRow", data=rows)
        # Statements carry their values as literals; a vector update of 3,000 rows is about
        # 4.5 MB of SQL, above the default max_query_size of 256 KiB.
        big = {"max_query_size": 64 * 1024 * 1024}
        if op == "delete":
            return self._q("DELETE FROM items WHERE id IN (%s)" % ",".join(map(str, ids)), settings=big)
        if op == "set_attr":
            return self._q("UPDATE items SET cat = transform(id, [%s], [%s], cat) WHERE id IN (%s)" % (
                ",".join(map(str, ids)), ",".join(str(int(a)) for a in attrs), ",".join(map(str, ids))),
                settings=big)
        if op == "update":
            arr = "CAST([" + ",".join("[" + ",".join(repr(float(x)) for x in np.asarray(v, np.float32)) + "]"
                                      for v in vecs) + "] AS Array(Array(Float32)))"
            return self._q("UPDATE items SET vec = arrayElement(%s, indexOf([%s], id)) WHERE id IN (%s)" % (
                arr, ",".join(map(str, ids)), ",".join(map(str, ids))), settings=big)
        raise ValueError(op)

    def search(self, Q, k, ef, filt=None, iterative=None):
        where = ""
        if filt is not None:
            where = "WHERE cat = %d " % int(filt[1]) if filt[0] == "eq" else "WHERE cat < %d " % int(filt[1])
        ids, ds = [], []
        for q in np.asarray(Q, dtype=np.float32):
            out = self._q("SELECT id, L2Distance(vec, %s) AS d FROM items %sORDER BY d ASC LIMIT %d FORMAT TSV"
                          % (_vec_sql(q), where, int(k)),
                          settings={"hnsw_candidate_list_size_for_search": int(ef)})
            row_ids, row_d = [], []
            for line in out.splitlines():
                a, b = line.split("\t")
                row_ids.append(int(a))
                row_d.append(float(b) * float(b))
            ids.append(row_ids)
            ds.append(row_d)
        return ids, ds

    def concurrent_writer(self):
        """HTTP requests are independent, so the adapter itself serves a second client."""
        return self

    def quiesce(self, timeout=600):
        """Wait until no merge or mutation of the table is running."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = self._q("SELECT count() FROM system.merges WHERE table = 'items'").strip()
            u = self._q("SELECT count() FROM system.mutations WHERE table = 'items' AND NOT is_done").strip()
            if m == "0" and u == "0":
                return
            time.sleep(0.2)
        raise EngineError("merges or mutations still running after %ds" % timeout)

    def exactness_promised(self):
        """The documentation says that parts with deleted rows are read in full (exact
        search). When every active part has deleted rows, every unfiltered answer must
        therefore be exact."""
        out = self._q("SELECT count(), countIf(has_lightweight_delete) FROM system.parts "
                      "WHERE table = 'items' AND active AND rows > 0 FORMAT TSV").strip().split("\t")
        n, d = int(out[0]), int(out[1])
        return n > 0 and n == d

    def state_token(self):
        """The active parts: the vector indexes are unchanged exactly when this is unchanged."""
        return self._q("SELECT groupArray(name) FROM (SELECT name FROM system.parts WHERE table = 'items' "
                       "AND active ORDER BY name)").strip()

    def exact_search(self, Q, k, filt=None):
        """The exact path (skipping indexes off), for triage and baseline B2."""
        where = ""
        if filt is not None:
            where = "WHERE cat = %d " % int(filt[1]) if filt[0] == "eq" else "WHERE cat < %d " % int(filt[1])
        ids, ds = [], []
        for q in np.asarray(Q, dtype=np.float32):
            out = self._q("SELECT id, L2Distance(vec, %s) AS d FROM items %sORDER BY d ASC LIMIT %d FORMAT TSV"
                          % (_vec_sql(q), where, int(k)), settings={"use_skip_indexes": 0})
            r = [l.split("\t") for l in out.splitlines()]
            ids.append([int(a) for a, _ in r])
            ds.append([float(b) * float(b) for _, b in r])
        return ids, ds

    def diagnose(self, i):
        """Where row i lives: its part, and that part's row and deleted-row state."""
        part = self._q("SELECT _part FROM items WHERE id = %d" % i).strip()
        info = self._q("SELECT name, rows, has_lightweight_delete, active FROM system.parts WHERE table = 'items' "
                       "AND name = '%s' FORMAT TSV" % part).strip() if part else ""
        return dict(part=part, part_info=info,
                    parts=self._q("SELECT count() FROM system.parts WHERE table = 'items' AND active").strip(),
                    patch_parts=self._q("SELECT count() FROM system.parts WHERE table = 'items' AND active "
                                        "AND startsWith(name, 'patch')").strip())

    def lookup(self, ids):
        ids = [int(i) for i in ids]
        got = {}
        for b in range(0, len(ids), 1000):
            part = ids[b:b + 1000]
            out = self._q("SELECT id, vec, cat FROM items WHERE id IN (%s) FORMAT JSONEachRow" % ",".join(map(str, part)))
            for line in out.splitlines():
                r = json.loads(line)
                got[int(r["id"])] = (np.asarray(r["vec"], dtype=np.float32), int(r["cat"]))
        return [i in got for i in ids], [got.get(i) for i in ids]

    def optimize(self):
        return self._q("OPTIMIZE TABLE items FINAL")

    def materialize(self):
        return self._q("ALTER TABLE items MATERIALIZE INDEX vidx", settings={"mutations_sync": 2})

    def attr_index(self):
        self._q("ALTER TABLE items ADD INDEX IF NOT EXISTS cidx cat TYPE minmax GRANULARITY 1")
        return self._q("ALTER TABLE items MATERIALIZE INDEX cidx", settings={"mutations_sync": 2})

    def maintenance_ops(self):
        return {"restart": self.clean_restart, "optimize": self.optimize, "materialize": self.materialize,
                "attr_index": self.attr_index}
