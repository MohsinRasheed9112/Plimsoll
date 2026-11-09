"""Experiment 5: baselines on known bugs at equal budget (Sec. V-C5, Table II).

Usage (from code/):
  python experiments/exp5.py qdrant <version> <first_seed> <n>      Qdrant lost-points scenario
  python experiments/exp5.py clickhouse <version> <first_seed> <n>  ClickHouse delete scenario
  python experiments/exp5.py summarize
Each trial writes results/exp5/<system>-<version>/<seed>.json; existing trials are kept.
Every method sees the same 1,000 queries (k = 10) and the same lookups of all items.
"""
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from plimsoll import gt as G  # noqa: E402
from plimsoll import paths  # noqa: E402
from plimsoll import oracles as O  # noqa: E402
from plimsoll import stats as S  # noqa: E402
from plimsoll.contracts import CONTRACTS  # noqa: E402
from plimsoll.model import DELETED, LIVE, Model  # noqa: E402

OUT = os.environ.get("PLIMSOLL_EXP5_OUT", os.path.join(ROOT, "results", "exp5"))   # test runs can go elsewhere
N, K, NQ, LEVEL = 15_000, 10, 1_000, 0.001
FLOOR = 1.0 / (K * NQ)
X = np.load(os.path.join(ROOT, "data", "sift100k_base.npy"))[:N]
Q = np.load(os.path.join(ROOT, "data", "sift100k_queries.npy"))[:NQ]


def env():
    return dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), python=platform.python_version(),
                numpy=np.__version__, cpus=os.cpu_count())


def measure(eng, model, contract, size_promised, exact_fn):
    """One state: EI on the 1,000 answers and on lookups of every id, B1 recall, B2 overlap."""
    ids, ds = eng.search(Q, K, 64)
    c = dict(contract, size=dict(contract["size"], promised=size_promised))
    viol, n_bad = O.ei_check_search(model, c, Q, ids, ds, K, "q")
    live_ids = np.flatnonzero(model.status == LIVE)
    gt = G.compute(Q, model.vec, live_ids, K, contract, n_check=20, rng=np.random.default_rng(1))
    rec = O.recall_per_query(model, contract, Q, ids, gt, K)
    eids, _ = exact_fn(Q, K)
    overlap = np.array([len(set(a) & set(b)) / K for a, b in zip(ids, eids)])
    lk = list(range(N))
    lv, vv = eng.lookup(lk)
    vecs = [None if v is None else (v[0] if isinstance(v, tuple) else v) for v in vv]
    lviol = O.ei_check_lookup(model, lk, lv, vecs, "lookup")
    by_rule = {}
    for v in viol + lviol:
        by_rule[v["rule"]] = by_rule.get(v["rule"], 0) + 1
    absent_live = sum(1 for v in lviol if v.get("expected") == "live")
    present_dead = sum(1 for v in lviol if v.get("expected") == "absent")
    return dict(ei_answers_flagged=n_bad, ei_by_rule=by_rule, recall=float(rec.mean()),
                overlap=float(overlap.mean()), short=int(sum(len(i) < K for i in ids)),
                lookups=len(lk), b4_absent_live=absent_live, b4_present_deleted=present_dead,
                gt_fallback=int(gt["n_fallback"]), gt_checked=int(gt["checked"]))


# ------------------------------------------------------------------- Qdrant
def qdrant_trial(version, seed):
    from plimsoll.adapters.qdrant_f import QdrantF
    eng = QdrantF(paths.work_dir("exp5_qd_%s_%d" % (version, seed)), seed, root="/var/tmp/plim_qc")
    eng.binary = paths.qdrant_binary(version)
    eng.full_scan_kb = None   # keep Qdrant's default full_scan_threshold (the bug is in storage)
    contract = json.loads(json.dumps(CONTRACTS["qdrant_hnsw"]))
    rng = np.random.default_rng([seed, 5])
    model = Model(N, 128)
    out = dict(system="qdrant", version=version, seed=seed)
    t0 = time.time()
    try:
        eng.start()
        eng.configure_campaign()
        for b in range(0, N, 1000):
            ids = np.arange(b, b + 1000)
            eng.write("insert", ids, X[ids], rng.integers(0, 100, 1000))
        model.status[:] = LIVE
        model.vec[:] = X
        eng.quiesce()
        live = set(range(N))
        for n in (2018, 123):
            d = np.array(sorted(int(i) for i in rng.choice(sorted(live), n, replace=False)))
            eng.write("delete", d)
            live -= set(int(i) for i in d)
            model.status[d] = DELETED
        eng.optimize()
        tgt = np.array(sorted(int(i) for i in rng.choice(sorted(live), 1366, replace=False)))
        eng.write("set_attr", tgt, attrs=rng.integers(0, 100, len(tgt)))
        eng.quiesce()
        eng.kill()
        eng.recover()
        eng.quiesce()
        out["after_recovery"] = measure(eng, model, contract, False,
                                        lambda q, k: eng.search(q, k, 64, exact=True))
    finally:
        eng.stop()
    out["secs"] = round(time.time() - t0, 1)
    out["environment"] = env()
    return out


# --------------------------------------------------------------- ClickHouse
def clickhouse_trial(version, seed):
    from plimsoll.adapters.clickhouse_f import ClickHouseF
    eng = ClickHouseF(paths.work_dir("exp5_ch_%s_%d" % (version, seed)), seed)
    eng.binary = paths.clickhouse_binary(version)
    contract = json.loads(json.dumps(CONTRACTS["clickhouse_hnsw_f32"]))
    rng = np.random.default_rng([seed, 5])
    model = Model(N, 128)
    out = dict(system="clickhouse", version=version, seed=seed)
    t0 = time.time()
    try:
        eng.start()
        out["server_version"] = eng._q("SELECT version()").strip()
        for b in range(0, N, 1000):
            ids = np.arange(b, b + 1000)
            eng.write("insert", ids, X[ids], rng.integers(0, 100, 1000))
        model.status[:] = LIVE
        model.vec[:] = X
        eng.optimize()
        eng.quiesce()
        ex = lambda q, k: eng.exact_search(q, k)  # noqa: E731
        out["state_A"] = measure(eng, model, contract, True, ex)
        dead = np.sort(rng.choice(N, 100, replace=False))
        eng.write("delete", dead)
        model.status[dead] = DELETED
        eng.quiesce()
        out["state_B"] = measure(eng, model, contract, True, ex)
    finally:
        eng.stop()
    out["secs"] = round(time.time() - t0, 1)
    out["environment"] = env()
    return out


def run(system, version, first, n):
    d = os.path.join(OUT, "%s-%s" % (system, version))
    os.makedirs(d, exist_ok=True)
    for seed in range(first, first + n):
        path = os.path.join(d, "%d.json" % seed)
        if os.path.exists(path):
            continue
        r = None
        for attempt in range(3):     # a server that fails to start is retried with the same seed
            try:
                r = (qdrant_trial if system == "qdrant" else clickhouse_trial)(version, seed)
                break
            except Exception as e:    # noqa: BLE001
                print(time.strftime("%H:%M:%S"), system, version, seed, "attempt", attempt, "failed:", repr(e)[:300],
                      flush=True)
        if r is None:
            continue
        r["attempts"] = attempt + 1
        with open(path + ".tmp", "w") as f:
            json.dump(r, f, indent=1)
        os.replace(path + ".tmp", path)
        print(time.strftime("%H:%M:%S"), system, version, seed, json.dumps(
            {k: v for k, v in r.items() if k.startswith(("after", "state"))})[:400], flush=True)


# ------------------------------------------------------------------ summary
def _load(name):
    d = os.path.join(OUT, name)
    if not os.path.isdir(d):
        return []
    return [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")]


def _decide(x, calib, stat):
    return S.mean_pvalue(x[stat], [c[stat] for c in calib], lower=True, floor=FLOOR) <= LEVEL


def _methods(states, healthy):
    rows = []
    for s in states:
        rows.append(dict(EI=s["ei_answers_flagged"] > 0 or s["b4_absent_live"] > 0 or s["b4_present_deleted"] > 0,
                         B1=_decide(s, healthy, "recall"), B2=_decide(s, healthy, "overlap"),
                         B4=s["b4_absent_live"] > 0 or s["b4_present_deleted"] > 0))
    return {m: sum(r[m] for r in rows) for m in ("EI", "B1", "B2", "B4")}


def _loo(healthy):
    fa = {"B1": 0, "B2": 0}
    for i, s in enumerate(healthy):
        rest = healthy[:i] + healthy[i + 1:]
        fa["B1"] += int(_decide(s, rest, "recall"))
        fa["B2"] += int(_decide(s, rest, "overlap"))
    return fa


def summarize():
    res = {}
    qb, qh = _load("qdrant-1.18.3"), _load("qdrant-1.19.2")
    if qb and qh:
        sb, sh = [t["after_recovery"] for t in qb], [t["after_recovery"] for t in qh]
        res["qdrant_lost_points"] = dict(
            trials=len(sb), healthy_trials=len(sh), triggered=sum(s["b4_absent_live"] > 0 for s in sb),
            detected=_methods(sb, sh), healthy_false_alarms_loo=_loo(sh),
            healthy_ei=_methods(sh, sh)["EI"],
            recall_buggy_triggered=[s["recall"] for s in sb if s["b4_absent_live"] > 0],
            recall_healthy=dict(mean=float(np.mean([s["recall"] for s in sh])),
                                min=float(np.min([s["recall"] for s in sh]))))
    c9, c3 = _load("clickhouse-26.9.13.15"), _load("clickhouse-26.3.45.3")
    if c9:
        sa, sbb = [t["state_A"] for t in c9], [t["state_B"] for t in c9]
        res["clickhouse_117286"] = dict(trials=len(sbb), detected=_methods(sbb, sa),
                                        short_answers=[s["short"] for s in sbb],
                                        healthy_false_alarms_loo=_loo(sa), state_A_ei=_methods(sa, sa)["EI"],
                                        recall_drop_points=100 * (np.mean([s["recall"] for s in sa])
                                                                  - np.mean([s["recall"] for s in sbb])))
    if c9 and c3:
        sa9, sa3 = [t["state_A"] for t in c9], [t["state_A"] for t in c3]
        res["clickhouse_102480"] = dict(trials=len(sa3), detected=_methods(sa3, sa9),
                                        score_flags=[s["ei_by_rule"].get("score", 0) for s in sa3],
                                        recall_26_3=float(np.mean([s["recall"] for s in sa3])),
                                        recall_26_9=float(np.mean([s["recall"] for s in sa9])))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "summarize":
        summarize()
    else:
        run(a[0], a[1], int(a[2]), int(a[3]))
