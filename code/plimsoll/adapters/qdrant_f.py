"""Qdrant adapter with an integer payload field and filtered search (campaigns).

Extends adapters/qdrant.py: every point has payload {"cat": int}. Filtered searches use
FieldCondition match (cat = c) or range (cat < c). Vector updates use update_vectors
(the payload is kept); attribute updates use set_payload, grouped by value in one
batch request. Extra maintenance operations: an integer payload index on cat, an
optimizer trigger (empty optimizer update), and snapshot then restore.
"""
import os

import numpy as np
from qdrant_client import models

from .qdrant import COLL, Qdrant


class QdrantF(Qdrant):
    name = "qdrant"

    def write(self, op, ids, vecs=None, attrs=None):
        ids = [int(i) for i in ids]

        def go():
            if op == "insert":
                self.client.upsert(COLL, points=models.Batch(
                    ids=ids, vectors=np.asarray(vecs, dtype=np.float32).tolist(),
                    payloads=[{"cat": int(a)} for a in attrs]), wait=True)
            elif op == "update":
                self.client.update_vectors(COLL, points=[
                    models.PointVectors(id=i, vector=np.asarray(v, dtype=np.float32).tolist())
                    for i, v in zip(ids, vecs)], wait=True)
            elif op == "set_attr":
                groups = {}
                for i, a in zip(ids, attrs):
                    groups.setdefault(int(a), []).append(i)
                ops = [models.SetPayloadOperation(set_payload=models.SetPayload(payload={"cat": a}, points=p))
                       for a, p in sorted(groups.items())]
                self.client.batch_update_points(COLL, update_operations=ops, wait=True)
            elif op == "delete":
                self.client.delete(COLL, points_selector=models.PointIdsList(points=ids), wait=True)
            else:
                raise ValueError(op)
        return self._guard(go)

    @staticmethod
    def _filter(filt):
        if filt is None:
            return None
        if filt[0] == "eq":
            cond = models.FieldCondition(key="cat", match=models.MatchValue(value=int(filt[1])))
        else:
            cond = models.FieldCondition(key="cat", range=models.Range(lt=int(filt[1])))
        return models.Filter(must=[cond])

    def search(self, Q, k, ef, filt=None, iterative=None, exact=False):
        Q = np.asarray(Q, dtype=np.float32)
        flt = self._filter(filt)

        def go():
            ids, ds = [], []
            for b in range(0, len(Q), 100):
                reqs = [models.QueryRequest(query=q.tolist(), limit=int(k), filter=flt,
                                            params=models.SearchParams(hnsw_ef=int(ef), exact=exact),
                                            with_payload=False, with_vector=False)
                        for q in Q[b:b + 100]]
                for r in self.client.query_batch_points(COLL, requests=reqs):
                    ids.append([int(p.id) for p in r.points])
                    ds.append([float(p.score) * float(p.score) for p in r.points])
            return ids, ds
        return self._guard(go)

    def lookup(self, ids):
        ids = [int(i) for i in ids]

        def go():
            got = {}
            for b in range(0, len(ids), 1000):
                for r in self.client.retrieve(COLL, ids=ids[b:b + 1000], with_vectors=True, with_payload=True):
                    got[int(r.id)] = (np.asarray(r.vector, dtype=np.float32), int((r.payload or {}).get("cat", -1)))
            return [i in got for i in ids], [got.get(i) for i in ids]
        return self._guard(go)

    def concurrent_writer(self):
        """A second client (its own gRPC channel) for concurrent writes."""
        import copy
        from qdrant_client import QdrantClient
        w = copy.copy(self)
        w.client = QdrantClient(host="127.0.0.1", port=self.http, grpc_port=self.grpc, prefer_grpc=True,
                                timeout=600)
        w.close = w.client.close
        return w

    def configure_campaign(self):
        """Campaign collections are small (20,000 vectors), so both size thresholds are
        scaled down from their defaults: build HNSW once a segment holds more than about
        2,000 vectors (indexing_threshold, kB), and search segments above 1,000 kB through
        the HNSW graph instead of a full scan (full_scan_threshold, kB). With the default
        full_scan_threshold of 10,000 kB, every search of a 20,000-vector collection is a
        full scan: recall@10 is 1.0 even at hnsw_ef = 1. Setting full_scan_kb to 0 keeps
        the default (the full-scan configuration of Table III)."""
        self.client.update_collection(COLL, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000))
        if getattr(self, "full_scan_kb", 1000):
            self.client.update_collection(COLL, hnsw_config=models.HnswConfigDiff(full_scan_threshold=self.full_scan_kb))

    full_scan_kb = 1000

    # extra maintenance operations
    def attr_index(self):
        return self._guard(lambda: self.client.create_payload_index(
            COLL, field_name="cat", field_schema=models.PayloadSchemaType.INTEGER, wait=True))

    def attr_index_drop(self):
        return self._guard(lambda: self.client.delete_payload_index(COLL, field_name="cat", wait=True))

    def optimize(self):
        return self._guard(lambda: self.client.update_collection(
            COLL, optimizers_config=models.OptimizersConfigDiff()))

    def snapshot_restore(self):
        def go():
            snap = self.client.create_snapshot(COLL, wait=True)
            path = os.path.join(self.root, "snapshots", COLL, snap.name)
            self.client.recover_snapshot(COLL, location="file://" + path, wait=True)
        return self._guard(go)

    def maintenance_ops(self):
        return {"flush": self.flush, "restart": self.clean_restart, "rebuild": lambda: self.rebuild(0),
                "attr_index": self.attr_index, "attr_index_drop": self.attr_index_drop,
                "optimize": self.optimize,
                "snapshot_restore": self.snapshot_restore}

    PRESERVING = ("flush", "restart")
    # snapshot then restore reverts the collection to the snapshot's time, so writes
    # concurrent with it may be reverted (snapshots documentation: a snapshot holds "a
    # specific collection on a specific node at a specific time")
    STATE_REPLACING = ("snapshot_restore",)
    ITERATIVE = (None,)
    UNORDERED = ()
