"""Tests of the reference engine: durability, recovery, index preservation and faults."""
import json
import os

import numpy as np
import pytest

from plimsoll.refengine.client import RefEngine

DIM = 16
CAP = 2000


def _vecs(n, seed):
    return np.random.default_rng(seed).integers(0, 50, size=(n, DIM)).astype(np.float32)


def _engine(tmp_path, mode):
    e = RefEngine(str(tmp_path / mode), mode, threads=2, M=8, ef_construction=50, cap=CAP, dim=DIM)
    e.start(build_seed=3)
    return e


@pytest.mark.parametrize("mode", ["exact", "hnsw"])
def test_acknowledged_writes_survive_kill(tmp_path, mode):
    e = _engine(tmp_path, mode)
    X = _vecs(600, 1)
    e.write("insert", range(0, 300), X[:300])
    e.flush()
    e.write("insert", range(300, 600), X[300:])          # in the WAL only
    e.write("delete", [5, 305])
    U = _vecs(2, 2)
    e.write("update", [6, 306], U)
    e.kill()
    e.recover()
    live, vecs = e.lookup([5, 305, 6, 306, 7, 307])
    assert live == [False, False, True, True, True, True]
    assert np.array_equal(vecs[2], U[0]) and np.array_equal(vecs[3], U[1])
    assert np.array_equal(vecs[4], X[7]) and np.array_equal(vecs[5], X[307])
    e.stop()


@pytest.mark.parametrize("mode", ["exact", "hnsw"])
def test_torn_wal_tail_is_ignored(tmp_path, mode):
    e = _engine(tmp_path, mode)
    X = _vecs(200, 3)
    e.write("insert", range(0, 100), X[:100])
    e.write("insert", range(100, 200), X[100:])
    e.kill()
    wal = os.path.join(e.data_dir, "wal.log")
    size = os.path.getsize(wal)
    with open(wal, "r+b") as f:                            # tear the last record
        f.truncate(size - 7)
    e.recover()
    live, _ = e.lookup([0, 99, 100, 199])
    assert live == [True, True, False, False]
    e.stop()


def test_clean_restart_preserves_hnsw_answers(tmp_path):
    e = _engine(tmp_path, "hnsw")
    X = _vecs(1500, 4)
    e.write("insert", range(1500), X)
    e.write("delete", range(0, 1500, 7))
    Q = _vecs(50, 5)
    before = e.search(Q, 10, 32)
    e.clean_restart()
    after = e.search(Q, 10, 32)
    assert before == after
    e.stop()


def test_exact_engine_returns_exact_top_k(tmp_path):
    e = _engine(tmp_path, "exact")
    X = _vecs(800, 6)
    e.write("insert", range(800), X)
    Q = _vecs(20, 7)
    ids, dists = e.search(Q, 10, 0)
    D = ((Q[:, None, :].astype(np.float64) - X[None, :, :]) ** 2).sum(-1)
    for i in range(len(Q)):
        kth = np.sort(D[i])[9]
        assert all(D[i][j] <= kth for j in ids[i])
        assert np.allclose(dists[i], D[i][ids[i]])
    e.stop()


def test_load_faults_and_desync(tmp_path):
    e = _engine(tmp_path, "hnsw")
    X = _vecs(1000, 8)
    e.write("insert", range(1000), X)
    e.write("delete", range(900, 1000))
    e.clean_restart(load_faults=[{"kind": "lost_entries", "frac": 0.05, "seed": 1},
                                 {"kind": "resurrect", "frac": 0.1, "seed": 2}])
    log = [json.loads(l) for l in open(os.path.join(e.data_dir, "faults.jsonl"))]
    lost = log[0]["ids"]
    res = log[1]["ids"]
    assert len(lost) == 45 and len(res) == 10
    ids, _ = e.search(X[lost], 1, 2000)
    assert all(i[0] != l for i, l in zip(ids, lost))           # lost entries are not returned
    live, _ = e.lookup(lost)
    assert all(live)                                             # ... but their rows are live
    ids, d = e.search(X[res], 1, 2000)
    assert [i[0] for i in ids] == res                            # resurrected deletes are returned
    e.arm_desync(3)
    U = _vecs(5, 9) + 100
    upd = [i for i in range(5) if i not in lost]
    e.write("update", upd, U[:len(upd)])
    ids, d = e.search(U[:len(upd)], 1, 2000)
    hits = [ids[j][0] == upd[j] for j in range(len(upd))]
    assert hits == [False] * 3 + [True] * (len(upd) - 3)        # first 3 updates skipped the index
    e.stop()
