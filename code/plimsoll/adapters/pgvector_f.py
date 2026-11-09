"""pgvector adapter with an integer attribute and filtered search (campaigns).

Extends adapters/pgvector.py: items(id bigint primary key, embedding vector(d), cat int).
Filtered searches add WHERE cat = c or WHERE cat < c. Each search may run with
hnsw.iterative_scan off (default), strict_order or relaxed_order; relaxed order
is documented to allow results "slightly out of order by distance", so the order
rule is not applied to those answers. Extra maintenance operations: VACUUM, a B-tree
index on cat, and REINDEX INDEX CONCURRENTLY.
"""
import numpy as np

from .pgvector import PgVector, _np32


class PgVectorF(PgVector):
    name = "pgvector"

    def start(self, build_seed=1, load_faults=None):
        assert not load_faults
        self._initdb()
        self._pg_start()
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION vector")
            cur.execute("CREATE TABLE items (id bigint PRIMARY KEY, embedding vector(%d), cat int)" % self.dim)
        self.started_once = True
        from pgvector.psycopg import register_vector
        register_vector(self.conn)
        self._session()
        self.iter_mode = None

    def _session(self):
        super()._session()
        self.iter_mode = None

    def write(self, op, ids, vecs=None, attrs=None):
        ids = [int(i) for i in ids]

        def go():
            with self.conn.transaction():
                with self.conn.cursor() as cur:
                    if op == "insert":
                        cur.executemany("INSERT INTO items (id, embedding, cat) VALUES (%s, %b, %s)",
                                        [(i, v, int(a)) for i, v, a in zip(ids, vecs, attrs)])
                    elif op == "update":
                        cur.executemany("UPDATE items SET embedding = %b WHERE id = %s", list(zip(vecs, ids)))
                    elif op == "set_attr":
                        cur.executemany("UPDATE items SET cat = %s WHERE id = %s",
                                        [(int(a), i) for a, i in zip(attrs, ids)])
                    elif op == "delete":
                        cur.execute("DELETE FROM items WHERE id = ANY(%s)", (ids,))
                    else:
                        raise ValueError(op)
        return self._guard(go)

    def _set_iter(self, cur, mode):
        mode = mode or "off"
        if self.iter_mode != mode:
            cur.execute("SET hnsw.iterative_scan = %s" % mode)
            self.iter_mode = mode

    def search(self, Q, k, ef, filt=None, iterative=None):
        where, arg = "", ()
        if filt is not None:
            where = "WHERE cat = %s " if filt[0] == "eq" else "WHERE cat < %s "
            arg = (int(filt[1]),)

        def go():
            ids, ds = [], []
            sql = ("SELECT id, embedding <-> %b FROM items " + where +
                   "ORDER BY embedding <-> %b LIMIT %s")
            with self.conn.cursor() as cur:
                self._set_ef(cur, ef)
                self._set_iter(cur, iterative)
                for q in np.asarray(Q, dtype=np.float32):
                    rows = cur.execute(sql, (q,) + arg + (q, int(k)), prepare=True).fetchall()
                    ids.append([int(r[0]) for r in rows])
                    ds.append([float(r[1]) * float(r[1]) for r in rows])
            return ids, ds
        return self._guard(go)

    def exact_search(self, Q, k, filt=None):
        """The exact path (no index scan), for triage and baseline B2."""
        where, arg = "", ()
        if filt is not None:
            where = "WHERE cat = %s " if filt[0] == "eq" else "WHERE cat < %s "
            arg = (int(filt[1]),)

        def go():
            ids, ds = [], []
            with self.conn.cursor() as cur:
                cur.execute("SET enable_indexscan = off")
                cur.execute("SET enable_seqscan = on")
                try:
                    for q in np.asarray(Q, dtype=np.float32):
                        rows = cur.execute("SELECT id, embedding <-> %b FROM items " + where +
                                           "ORDER BY embedding <-> %b LIMIT %s", (q,) + arg + (q, int(k))).fetchall()
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
                rows = cur.execute("SELECT id, embedding, cat FROM items WHERE id = ANY(%s)", (ids,)).fetchall()
            got = {int(r[0]): (_np32(r[1]), int(r[2])) for r in rows}
            return [i in got for i in ids], [got.get(i) for i in ids]
        return self._guard(go)

    def concurrent_writer(self):
        """A second client (its own connection) for concurrent writes."""
        import copy
        w = copy.copy(self)
        w.conn = None
        w._connect()
        w.close = w._close
        return w

    # extra maintenance operations
    def vacuum(self):
        return self._guard(lambda: self.conn.execute("VACUUM items"))

    def attr_index(self):
        return self._guard(lambda: self.conn.execute("CREATE INDEX IF NOT EXISTS items_cat_idx ON items (cat)"))

    def reindex_concurrently(self):
        return self._guard(lambda: self.conn.execute("REINDEX INDEX CONCURRENTLY items_embedding_idx"))

    def maintenance_ops(self):
        return {"flush": self.flush, "restart": self.clean_restart, "rebuild": lambda: self.rebuild(0),
                "vacuum": self.vacuum, "attr_index": self.attr_index,
                "reindex_concurrently": self.reindex_concurrently}

    # operations after which the logical state and the index are unchanged
    PRESERVING = ("flush", "restart")
    ITERATIVE = (None, "strict_order", "relaxed_order")
    UNORDERED = ("relaxed_order",)
