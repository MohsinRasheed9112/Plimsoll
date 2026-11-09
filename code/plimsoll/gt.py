"""Exact k-NN ground truth for large collections (Experiment 4 onward).

compute() finds, for each query, the exact k nearest live items in squared L2.
Candidates come from a float32 BLAS pass over the live items in chunks; the
candidates are then ranked in float64 by the direct formula. Each query carries a
certificate; a query whose certificate fails is recomputed by a full float64 pass.
update() gives the ground truth after some items leave the live set and others
join it (with their current vectors), reusing the candidates of an earlier result.

Certificate. Let s(x) be the float32 score fl(||x||^2 - 2 q.x) and w the largest
score among the C+1 candidates. Every non-candidate x has s(x) >= w, so its exact
squared distance is at least lb = ||q||^2 + w - e32(q), where
e32(q) = 2 gamma_{d+2}(2^-24) (max ||x||^2 + 2 |q| . max|x|) bounds the float32
error for any summation order (Higham 2002, with a factor 2 of slack). The k-th
float64 direct distance dk has relative error at most gamma_{d+2}(2^-53). The
candidates hold the true k nearest items when dk (1 + 2 gamma64) < lb.
A second path recomputes a sample of queries by a full float64 pass and must agree.
"""
import numpy as np

from .tolerance import U64, distances_and_eps, gamma

U32 = 2.0 ** -24
CHUNK = 50_000


def _direct64(q, V):
    diff = np.asarray(V, dtype=np.float64) - np.asarray(q, dtype=np.float64)[None, :]
    return np.einsum("ij,ij->i", diff, diff)


def _topc32(Q, vec, ids, C):
    """Per query, the C+1 ids with the smallest float32 scores among `ids`, the
    largest of those scores (w), and the float32 error bound e32 per query.
    If len(ids) <= C+1, every id is a candidate and w = +inf."""
    nq, d = Q.shape
    m = len(ids)
    if m == 0:
        return np.zeros((nq, 0), np.int64), np.full(nq, np.inf), np.zeros(nq)
    Q32 = np.asarray(Q, dtype=np.float32)
    keep = min(C + 1, m)
    cand = np.zeros((nq, 0), np.int64)
    cscore = np.zeros((nq, 0), np.float32)
    xn_max = 0.0
    xa_max = np.zeros(d)
    for b in range(0, m, CHUNK):
        sub = ids[b:b + CHUNK]
        X = vec[sub]
        xn = np.einsum("ij,ij->i", X.astype(np.float64), X.astype(np.float64))
        xn_max = max(xn_max, float(xn.max()))
        xa_max = np.maximum(xa_max, np.abs(X).max(axis=0).astype(np.float64))
        S = Q32 @ X.T
        S *= np.float32(-2.0)
        S += xn.astype(np.float32)[None, :]
        kk = min(keep, len(sub))
        if kk < len(sub):
            part = np.argpartition(S, kk - 1, axis=1)[:, :kk]
        else:
            part = np.broadcast_to(np.arange(len(sub)), (nq, len(sub)))
        ps = np.take_along_axis(S, part, axis=1)
        cand = np.concatenate([cand, sub[part]], axis=1)
        cscore = np.concatenate([cscore, ps], axis=1)
        if cand.shape[1] > keep:
            sel = np.argpartition(cscore, keep - 1, axis=1)[:, :keep]
            cand = np.take_along_axis(cand, sel, axis=1)
            cscore = np.take_along_axis(cscore, sel, axis=1)
    g32 = gamma(d + 2, U32)
    e32 = 2.0 * g32 * (xn_max + 2.0 * (np.abs(Q.astype(np.float64)) @ xa_max))
    if m <= C + 1:
        w = np.full(nq, np.inf)
    else:
        w = cscore.max(axis=1).astype(np.float64)
    return cand, w, e32


def _full64(q, vec, ids, k):
    """Full float64 direct pass over `ids`: (top-k ids, their distances)."""
    best_i = np.zeros(0, np.int64)
    best_d = np.zeros(0)
    for b in range(0, len(ids), CHUNK):
        sub = ids[b:b + CHUNK]
        dd = _direct64(q, vec[sub])
        best_i = np.concatenate([best_i, sub])
        best_d = np.concatenate([best_d, dd])
        if len(best_d) > k:
            sel = np.argpartition(best_d, k - 1)[:k]
            best_i, best_d = best_i[sel], best_d[sel]
    o = np.lexsort((best_i, best_d))
    return best_i[o], best_d[o]


def _finish(gt, Q, vec, contract, k):
    epsk = np.zeros(len(Q))
    for r in range(len(Q)):
        _, e = distances_and_eps(Q[r], vec[gt["topk"][r, k - 1]][None, :], contract)
        epsk[r] = e[0]
    gt["epsk"] = epsk
    return gt


def _rank(Q, vec, cand_lists, lb, live, k, g64, contract, fallback_ok=True):
    nq = len(Q)
    topk = np.zeros((nq, k), np.int64)
    dk = np.zeros(nq)
    cd = []
    n_fallback = 0
    for r in range(nq):
        c = cand_lists[r]
        d = _direct64(Q[r], vec[c])
        cd.append(d)
        o = np.lexsort((c, d))
        if len(o) >= k and d[o[k - 1]] * (1.0 + 2.0 * g64) < lb[r]:
            topk[r] = c[o[:k]]
            dk[r] = d[o[k - 1]]
            continue
        if not fallback_ok:
            raise RuntimeError("ground truth certificate failed for query %d" % r)
        n_fallback += 1
        ti, td = _full64(Q[r], vec, live, k)
        topk[r] = ti
        dk[r] = td[k - 1]
    return topk, dk, cd, n_fallback


def compute(Q, vec, live, k, contract, C=50, n_check=20, rng=None):
    """Exact top-k over the items `live` (sorted int64 ids), vectors from `vec`."""
    Q = np.asarray(Q, dtype=np.float32)
    live = np.asarray(live, dtype=np.int64)
    d = Q.shape[1]
    g64 = gamma(d + 2, U64)
    cand, w, e32 = _topc32(Q, vec, live, C)
    qn = np.einsum("ij,ij->i", Q.astype(np.float64), Q.astype(np.float64))
    lb = qn + w - e32
    topk, dk, cd, nfb = _rank(Q, vec, list(cand), lb, live, k, g64, contract)
    gt = dict(topk=topk, dk=dk, cand=list(cand), cand_d=cd, lb=lb, live=live,
              n_fallback=nfb, checked=0)
    gt["checked"] = _second_path(Q, vec, gt, k, g64, n_check, rng)
    return _finish(gt, Q, vec, contract, k)


def update(gt0, Q, vec, removed, added, k, contract, C=50, n_check=1, rng=None):
    """Ground truth after `removed` ids leave the live set of gt0 and `added` ids
    (with their vectors in `vec`) join it. An id in both is re-scored."""
    Q = np.asarray(Q, dtype=np.float32)
    d = Q.shape[1]
    g64 = gamma(d + 2, U64)
    removed = np.unique(np.asarray(removed, dtype=np.int64))
    added = np.unique(np.asarray(added, dtype=np.int64))
    live = np.union1d(np.setdiff1d(gt0["live"], removed, assume_unique=True), added)
    gone = np.zeros(int(max(gt0["live"].max(), added.max() if len(added) else 0)) + 1, bool)
    gone[removed] = True
    acand, aw, ae32 = _topc32(Q, vec, added, C)
    qn = np.einsum("ij,ij->i", Q.astype(np.float64), Q.astype(np.float64))
    lb = np.minimum(gt0["lb"], qn + aw - ae32)
    cand = []
    for r in range(len(Q)):
        c = gt0["cand"][r]
        c = c[~gone[c]]
        cand.append(np.concatenate([c, acand[r]]))
    topk, dk, cd, nfb = _rank(Q, vec, cand, lb, live, k, g64, contract)
    gt = dict(topk=topk, dk=dk, cand=cand, cand_d=cd, lb=lb, live=live, n_fallback=nfb,
              checked=0)
    gt["checked"] = _second_path(Q, vec, gt, k, g64, n_check, rng)
    return _finish(gt, Q, vec, contract, k)


def _second_path(Q, vec, gt, k, g64, n_check, rng):
    if rng is None or not n_check:
        return 0
    agree = 0
    for r in rng.choice(len(Q), size=min(n_check, len(Q)), replace=False):
        _, td = _full64(Q[r], vec, gt["live"], k)
        if abs(td[k - 1] - gt["dk"][r]) <= 2.0 * g64 * gt["dk"][r]:
            agree += 1
        else:
            raise RuntimeError("ground-truth paths disagree on query %d" % r)
    return agree
