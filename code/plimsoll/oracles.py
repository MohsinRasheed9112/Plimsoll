"""Oracles EI, WQ and SE, and exact ground truth (paper Secs. III-C to III-E).

All checks use only the recorded answers and the logical model; none reads engine
internals or the engine's fault log.
"""
import numpy as np

from .model import INDET, LIVE
from .tolerance import U64, batch_distances_and_eps, distances_and_eps, gamma


def _pad(lists, k, fill):
    out = np.full((len(lists), k), fill, dtype=np.float64 if isinstance(fill, float) else np.int64)
    lens = np.zeros(len(lists), dtype=np.int64)
    for i, row in enumerate(lists):
        n = min(len(row), k)
        lens[i] = len(row)
        if n:
            out[i, :n] = row[:n]
    return out, lens


# --------------------------------------------------------------------------- EI
def ei_check_search(model, contract, Q, ids_lists, dists_lists, k, where):
    """EI rules on a batch of search answers. Returns (violations, n_answers_with_violation)."""
    viols = []
    bad_answer = np.zeros(len(ids_lists), dtype=bool)
    size_on = contract["size"]["promised"]
    if size_on:
        n_all, n_ex = model.n_forall(), model.n_exists()
    for qi, (ids, ds) in enumerate(zip(ids_lists, dists_lists)):
        m = len(ids)
        if m != len(ds):
            viols.append(dict(rule="malformed", where=where, query=qi))
            bad_answer[qi] = True
        if len(set(ids)) != m:
            viols.append(dict(rule="uniqueness", where=where, query=qi))
            bad_answer[qi] = True
        if contract["order"]["follows"] == "reported" and any(ds[j] > ds[j + 1] for j in range(m - 1)):
            viols.append(dict(rule="order", where=where, query=qi))
            bad_answer[qi] = True
        if size_on and not (min(k, n_all) <= m <= min(k, n_ex)):
            viols.append(dict(rule="size", where=where, query=qi, m=m))
            bad_answer[qi] = True
    ids_arr, lens = _pad(ids_lists, k, -1)
    ds_arr, _ = _pad(dists_lists, k, 0.0)
    valid = ids_arr >= 0
    st = np.where(valid, model.status[np.clip(ids_arr, 0, None)], -1)
    # existence: every returned id must be live in some possible state
    exists = (st == LIVE)
    indet_mask = (st == INDET)
    if indet_mask.any():
        for qi, j in zip(*np.nonzero(indet_mask)):
            exists[qi, j] = model.in_exists(int(ids_arr[qi, j]))
    for qi, j in zip(*np.nonzero(valid & ~exists)):
        viols.append(dict(rule="existence", where=where, query=int(qi), id=int(ids_arr[qi, j])))
        bad_answer[qi] = True
    # score: some possible version must explain the reported score
    if contract["score"]["on"]:
        sel = valid & (st == LIVE)
        qi_idx, j_idx = np.nonzero(sel)
        if len(qi_idx):
            ids_sel = ids_arr[qi_idx, j_idx]
            delta, eps = batch_distances_and_eps(Q[qi_idx], model.vec[ids_sel], contract)
            s = ds_arr[qi_idx, j_idx]
            badm = np.abs(s - delta) > eps
            for t in np.nonzero(badm)[0]:
                viols.append(dict(rule="score", where=where, query=int(qi_idx[t]), id=int(ids_sel[t]),
                                  reported=float(s[t]), model=float(delta[t]), eps=float(eps[t])))
                bad_answer[qi_idx[t]] = True
        for qi, j in zip(*np.nonzero(valid & indet_mask)):
            i = int(ids_arr[qi, j])
            vs = model.possible_vectors(i)
            if not vs:
                continue
            delta, eps = distances_and_eps(Q[qi], np.stack(vs), contract)
            if not np.any(np.abs(ds_arr[qi, j] - delta) <= eps):
                viols.append(dict(rule="score", where=where, query=int(qi), id=i))
                bad_answer[qi] = True
    return viols, int(bad_answer.sum())


def ei_check_lookup(model, ids, live, vecs, where):
    """Lookup rule for items with a definite outcome."""
    viols = []
    for i, l, v in zip(ids, live, vecs):
        s = model.status[i]
        if s == INDET:
            continue  # handled by resolution
        if s == LIVE:
            if not l or not np.array_equal(v, model.vec[i]):
                viols.append(dict(rule="lookup", where=where, id=int(i), expected="live"))
        elif l:
            viols.append(dict(rule="lookup", where=where, id=int(i), expected="absent"))
    return viols


# --------------------------------------------------------------------------- WQ
def wq_hits(model, contract, W, first_ids):
    """Self-query hit test for definite witnesses W (L2: d* = 0, attained by w itself).

    Hit iff the first result has a possible version within distance eps(first) + eps(w)
    of the witness vector."""
    hits = np.zeros(len(W), dtype=bool)
    Vw = model.vec[W]
    _, eps_self = batch_distances_and_eps(Vw, Vw, contract)
    for t, (w, r) in enumerate(zip(W, first_ids)):
        if r < 0:
            continue
        vs = model.possible_vectors(int(r))
        if not vs:
            continue
        delta, eps = distances_and_eps(Vw[t], np.stack(vs), contract)
        hits[t] = bool(np.any(delta <= eps + eps_self[t]))
    return hits


# ------------------------------------------------------------- ground truth / SE
def ground_truth(model, contract, Q, k, n_cand=50, n_check=20, rng=None):
    """Exact k-th neighbor distance and its tolerance for each query, over the definite
    live items. Candidates come from a float64 expansion pass; the final ranking uses
    the float64 direct formula. A second path (full direct pass) checks a sample."""
    assert not model.has_indeterminate()
    live = model.definite_live_ids()
    X = model.vec[live].astype(np.float64)
    xn = np.einsum("ij,ij->i", X, X)
    xn_max = xn.max()
    xa_max = np.abs(X).max(axis=0)
    d = X.shape[1]
    g = gamma(d + 2, U64)
    dk = np.zeros(len(Q))
    epsk = np.zeros(len(Q))
    topk = np.zeros((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), 200):
        q = Q[s:s + 200].astype(np.float64)
        qn = np.einsum("ij,ij->i", q, q)
        D = qn[:, None] + xn[None, :] - 2.0 * (q @ X.T)
        cand = np.argpartition(D, n_cand, axis=1)[:, :n_cand + 1]
        for r in range(len(q)):
            c = cand[r]
            diff = X[c] - q[r]
            dd = np.einsum("ij,ij->i", diff, diff)
            order = np.argsort(dd, kind="stable")
            kth_direct = dd[order[k - 1]]
            # every non-candidate has expansion distance >= the largest candidate's;
            # err bounds the expansion error of any item and the direct error of the k-th
            worst_cand_exp = D[r, c].max()
            err = g * (qn[r] + xn_max + 2.0 * (xa_max @ np.abs(q[r]))) + g * kth_direct
            if not worst_cand_exp - err > kth_direct + err:
                raise RuntimeError("ground truth candidate margin too small; raise n_cand")
            row = s + r
            dk[row] = kth_direct
            topk[row] = live[c[order[:k]]]
            _, e = distances_and_eps(Q[row], model.vec[topk[row, k - 1]][None, :], contract)
            epsk[row] = e[0]
    # second path on a sample: full direct distances must give the same k-th distance
    agree = 0
    if rng is not None and n_check:
        for row in rng.choice(len(Q), size=min(n_check, len(Q)), replace=False):
            full = np.zeros(len(live))
            for b in range(0, len(live), 20000):
                diff = X[b:b + 20000] - Q[row].astype(np.float64)
                full[b:b + 20000] = np.einsum("ij,ij->i", diff, diff)
            if abs(np.partition(full, k - 1)[k - 1] - dk[row]) <= 2.0 * g * dk[row]:
                agree += 1
            else:
                raise RuntimeError("ground-truth paths disagree on query %d" % row)
    return dict(dk=dk, epsk=epsk, topk=topk, checked=agree)


def recall_per_query(model, contract, Q, ids_lists, gt, k):
    """recall@k with the near-tie rule: a returned live item counts when its exact
    distance exceeds the k-th true distance by at most the sum of both tolerances."""
    rec = np.zeros(len(Q))
    for qi, ids in enumerate(ids_lists):
        uniq = [i for i in dict.fromkeys(ids) if i >= 0 and model.status[i] == LIVE]
        if not uniq:
            continue
        delta, eps = distances_and_eps(Q[qi], model.vec[uniq], contract)
        ok = delta <= gt["dk"][qi] + eps + gt["epsk"][qi]
        rec[qi] = min(int(ok.sum()), k) / k
    return rec


def answers_identical(model, contract, Q, a, b):
    """Same identifiers with scores within tolerance, ties in any order."""
    ids_a, ds_a = a
    ids_b, ds_b = b
    changed = np.zeros(len(Q), dtype=bool)
    for qi in range(len(Q)):
        if set(ids_a[qi]) != set(ids_b[qi]) or len(ids_a[qi]) != len(ids_b[qi]):
            changed[qi] = True
            continue
        sa = dict(zip(ids_a[qi], ds_a[qi]))
        sb = dict(zip(ids_b[qi], ds_b[qi]))
        for i in sa:
            if model.status[i] == LIVE:
                _, e = distances_and_eps(Q[qi], model.vec[i][None, :], contract)
                tol = 2.0 * e[0]
            else:
                tol = 0.0
            if abs(sa[i] - sb[i]) > tol:
                changed[qi] = True
                break
    return changed
