"""Tests of the model, tolerance, statistics and oracles."""
import numpy as np
import pytest

from plimsoll import oracles as O
from plimsoll import stats as S
from plimsoll.contracts import CONTRACTS
from plimsoll.model import DELETED, INDET, LIVE, Model
from plimsoll.tolerance import distances_and_eps, gamma


def test_gamma():
    assert gamma(130, 2 ** -24) == pytest.approx(130 * 2 ** -24 / (1 - 130 * 2 ** -24))
    with pytest.raises(ValueError):
        gamma(2 ** 24, 2 ** -24)


def test_float32_direct_l2_within_tolerance():
    rng = np.random.default_rng(0)
    q = (rng.random(128) * 200).astype(np.float32)
    V = (rng.random((2000, 128)) * 200).astype(np.float32)
    s32 = ((V - q) ** 2).sum(axis=1, dtype=np.float32)          # float32 direct evaluation
    delta, eps = distances_and_eps(q, V, CONTRACTS["hnsw"])
    assert np.all(np.abs(s32 - delta) <= eps)


def test_float64_expansion_within_tolerance():
    rng = np.random.default_rng(1)
    q = (rng.random(128) * 200).astype(np.float32)
    V = (rng.random((2000, 128)) * 200).astype(np.float32)
    q64, V64 = q.astype(np.float64), V.astype(np.float64)
    s = q64 @ q64 + np.einsum("ij,ij->i", V64, V64) - 2 * V64 @ q64
    delta, eps = distances_and_eps(q, V, CONTRACTS["exact"])
    assert np.all(np.abs(s - delta) <= eps)


def test_model_unknown_and_resolve():
    m = Model(10, 2)
    v1, v2 = np.array([1, 1], np.float32), np.array([2, 2], np.float32)
    m.ack("insert", [0, 1], np.stack([v1, v1]))
    m.unknown("update", [0], v2[None, :])
    m.unknown("delete", [1])
    m.unknown("insert", [2], v2[None, :])
    assert m.status[0] == INDET and m.in_forall(0) and m.in_exists(0)
    assert m.status[1] == INDET and m.in_exists(1) and not m.in_forall(1)
    assert m.status[2] == INDET and m.in_exists(2) and not m.in_forall(2)
    assert m.resolve(0, v2) and m.status[0] == LIVE and np.array_equal(m.vec[0], v2)
    assert m.resolve(1, None) and m.status[1] == DELETED and np.array_equal(m.vec[1], v1)
    assert m.resolve(2, None) and m.status[2] == 0
    m.unknown("update", [0], v1[None, :])
    assert not m.resolve(0, np.array([9, 9], np.float32))      # impossible observation


def test_ei_flags_deleted_item_and_wrong_score():
    m = Model(5, 2)
    V = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [9, 9]], np.float32)
    m.ack("insert", range(5), V)
    m.ack("delete", [4])
    Q = np.array([[0, 0]], np.float32)
    ok, n = O.ei_check_search(m, CONTRACTS["hnsw"], Q, [[0, 1, 2]], [[0.0, 1.0, 1.0]], 3, "t")
    assert ok == [] and n == 0
    v, n = O.ei_check_search(m, CONTRACTS["hnsw"], Q, [[0, 4]], [[0.0, 162.0]], 3, "t")
    assert [x["rule"] for x in v] == ["existence"] and n == 1
    v, n = O.ei_check_search(m, CONTRACTS["hnsw"], Q, [[0, 1]], [[0.0, 1.5]], 3, "t")
    assert [x["rule"] for x in v] == ["score"]
    v, n = O.ei_check_search(m, CONTRACTS["hnsw"], Q, [[1, 0]], [[1.0, 0.0]], 3, "t")
    assert "order" in [x["rule"] for x in v]
    v, n = O.ei_check_search(m, CONTRACTS["exact"], Q, [[0, 1]], [[0.0, 1.0]], 3, "t")
    assert [x["rule"] for x in v] == ["size"]          # exact engine promises min(k, n) = 3


def test_ei_allows_any_possible_version():
    m = Model(3, 2)
    m.ack("insert", [0], np.array([[3, 0]], np.float32))
    m.unknown("update", [0], np.array([[0, 4]], np.float32))
    Q = np.array([[0, 0]], np.float32)
    for s in (9.0, 16.0):
        v, _ = O.ei_check_search(m, CONTRACTS["hnsw"], Q, [[0]], [[s]], 1, "t")
        assert v == []
    v, _ = O.ei_check_search(m, CONTRACTS["hnsw"], Q, [[0]], [[10.0]], 1, "t")
    assert [x["rule"] for x in v] == ["score"]


def test_holm():
    rej = S.holm({"a": 0.001, "b": 0.02, "c": 0.04}, 0.05)
    assert rej == {"a": 0.05 / 3, "b": 0.05 / 2, "c": 0.05}
    assert S.holm({"a": 0.02, "b": 0.03}, 0.05) == {"a": 0.025, "b": 0.05}
    assert S.holm({"a": 0.03, "b": 0.04}, 0.05) == {}


def test_mean_pvalue_and_count_pvalue_calibration():
    rng = np.random.default_rng(3)
    calib = rng.normal(0, 1, 20)
    assert S.mean_pvalue(10.0, calib) < 1e-5
    assert S.mean_pvalue(-10.0, calib) > 0.999
    # count p-values from healthy draws should be roughly uniform: check validity at 0.1
    n = 2000
    a, b = 2.0, 1500.0
    bad = 0
    trials = 200
    for t in range(trials):
        x = rng.binomial(n, rng.beta(a, b, size=21))
        p, _, _ = S.count_pvalue(int(x[20]), x[:20], n, 300, rng)
        bad += p <= 0.1
    assert bad / trials <= 0.16       # 0.1 plus sampling slack


def test_ground_truth_and_recall():
    rng = np.random.default_rng(4)
    m = Model(3000, 16)
    X = rng.integers(0, 30, size=(3000, 16)).astype(np.float32)
    m.ack("insert", range(3000), X)
    Q = rng.integers(0, 30, size=(50, 16)).astype(np.float32)
    gt = O.ground_truth(m, CONTRACTS["hnsw"], Q, 10, rng=rng)
    D = ((Q[:, None, :].astype(np.float64) - X[None]) ** 2).sum(-1)
    assert np.allclose(gt["dk"], np.sort(D, axis=1)[:, 9])
    ids = [list(np.argsort(D[i], kind="stable")[:10]) for i in range(50)]
    r = O.recall_per_query(m, CONTRACTS["hnsw"], Q, ids, gt, 10)
    assert np.all(r == 1.0)
