"""The blocked ground truth (plimsoll/gt.py) must agree with the float64 ground truth
of plimsoll/oracles.py, and update() must agree with a fresh compute()."""
import os

import numpy as np
import pytest

from plimsoll import gt as G
from plimsoll import oracles as O
from plimsoll.contracts import CONTRACTS
from plimsoll.model import Model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")


@pytest.fixture(scope="module")
def setup():
    if not os.path.exists(os.path.join(DATA, "sift100k_base.npy")):
        pytest.skip("SIFT data not downloaded: run python scripts/get_data.py sift")
    X = np.load(os.path.join(DATA, "sift100k_base.npy"))[:30_000]
    Q = np.load(os.path.join(DATA, "sift100k_queries.npy"))[:300]
    m = Model(len(X), X.shape[1])
    m.ack("insert", np.arange(len(X)), X)
    rng = np.random.default_rng(7)
    dead = rng.choice(len(X), 1_500, replace=False)
    m.ack("delete", dead)
    return X, Q, m, rng


def _same_answer(a, b, k):
    assert np.allclose(a["dk"], b["dk"], rtol=1e-12, atol=0)
    # identical top-k sets except where distances tie at the k-th position
    for r in range(len(a["dk"])):
        if set(a["topk"][r]) != set(b["topk"][r]):
            assert np.isclose(a["dk"][r], b["dk"][r])


def test_compute_matches_oracles_ground_truth(setup):
    X, Q, m, rng = setup
    c = CONTRACTS["hnsw"]
    ref = O.ground_truth(m, c, Q, 10, rng=np.random.default_rng(1))
    new = G.compute(Q, m.vec, m.definite_live_ids(), 10, c, rng=np.random.default_rng(1))
    _same_answer(ref, new, 10)
    assert np.allclose(ref["epsk"], new["epsk"])
    assert new["checked"] == 20


def test_update_matches_fresh_compute(setup):
    X, Q, m, rng = setup
    c = CONTRACTS["hnsw"]
    live = m.definite_live_ids()
    g0 = G.compute(Q, m.vec, live, 10, c)
    # desync-like change: 600 live items get new vectors; 200 deleted items come back
    ch = rng.choice(live, 600, replace=False)
    m2 = Model(m.cap, m.dim)
    m2.status[:] = m.status
    m2.vec[:] = m.vec
    src = rng.integers(0, len(X), 600)
    m2.vec[ch] = (X[src] + rng.normal(0, 1, (600, X.shape[1]))).astype(np.float32)
    back = rng.choice(m.deleted_ids(), 200, replace=False)
    m2.status[back] = 1
    upd = G.update(g0, Q, m2.vec, removed=ch, added=np.concatenate([ch, back]), k=10,
                   contract=c, rng=np.random.default_rng(3))
    fresh = G.compute(Q, m2.vec, m2.definite_live_ids(), 10, c)
    assert np.array_equal(upd["live"], m2.definite_live_ids())
    _same_answer(fresh, upd, 10)
