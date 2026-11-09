"""Chroma calibration for campaigns: which maintenance operations preserve answers,
and whether the WQ zero rule applies. Also the ef_search probe (Sec. V-C7).

Rules, fixed before the run:
  - An operation is index-preserving if, in all 10 trials, it changed no answer to 200
    unfiltered queries (same ids and the same reported distances) and caused no new
    persistent miss among 2,000 witnesses.
  - The WQ zero rule applies if no measured state of any trial has a persistent miss
    (a self-query whose first result is not the witness at ef_search = 20,000).
Trial t (seed 9700 + t): load 15,000 of the campaign's 20,000 SIFT vectors with attributes
uniform in 0..99, then three rounds of 200 vector updates (Gaussian noise, sd 1.0), 200
deletes, 200 inserts and 200 attribute updates; measure; clean restart; measure; vacuum;
measure.
Usage (from code/): python experiments/chroma_calib.py [ef] [calib]
Writes results/calib/chroma/{ef_probe,calib}.json.
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
from plimsoll import paths  # noqa: E402
from plimsoll.adapters.chroma_f import BASE, EF_FIXED, ChromaF  # noqa: E402

OUT = os.path.join(ROOT, "results", "calib", "chroma")
X = np.load(os.path.join(ROOT, "data", "sift100k_base.npy"))[:20_000]
QS = np.load(os.path.join(ROOT, "data", "sift100k_queries.npy"))


def env():
    return dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), python=platform.python_version(),
                numpy=np.__version__,
                chromadb="%s (%s; server via `chroma run`)" % (paths.CHROMA_VERSION, paths.chroma_binary()),
                machine=platform.machine(), cpus=os.cpu_count())


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    obj["environment"] = env()
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


def recall(ids, gt):
    return float(np.mean([len(set(a) & set(g)) / 10 for a, g in zip(ids, gt)]))


def ef_probe():
    """Is a changed ef_search applied to queries at once, or only after a restart?"""
    Q = QS[:200]
    X64, Q64 = X.astype(np.float64), Q.astype(np.float64)
    d = (Q64 ** 2).sum(1)[:, None] + (X64 ** 2).sum(1)[None, :] - 2.0 * Q64 @ X64.T
    gt = np.argsort(d, axis=1)[:, :10]
    out = []
    for ef0, ef1 in ((4, 400), (400, 4)):
        e = ChromaF(paths.work_dir("chroma_ef_probe"), 1)
        e.start()
        try:
            e._req("DELETE", BASE + "/items")
            c = e._req("POST", BASE, {"name": "items", "configuration": {"hnsw": {
                "space": "l2", "ef_construction": 100, "max_neighbors": 16, "ef_search": ef0}}})
            e.cid = c["id"]
            for b in range(0, 20_000, 1000):
                e.write("insert", np.arange(b, b + 1000), X[b:b + 1000], np.zeros(1000, int))
            r0 = recall(e.search(Q, 10, 0)[0], gt)
            e._set_hnsw(ef_search=ef1)
            stored = e._req("GET", BASE + "/by-id/" + e.cid)["configuration_json"]["hnsw"]["ef_search"]
            r1 = recall(e.search(Q, 10, 0)[0], gt)
            e.clean_restart()
            r2 = recall(e.search(Q, 10, 0)[0], gt)
            out.append(dict(created_ef=ef0, modified_to=ef1, stored_after_modify=stored, recall_created=r0,
                            recall_after_modify=r1, recall_after_restart=r2))
        finally:
            e.stop()
    save("ef_probe", dict(n=20_000, queries=200, k=10, trials=out))
    print(json.dumps(out, indent=1))


def measure(e, Q, W):
    ids, ds = e.search(Q, 10, EF_FIXED)
    wi, _ = e.search(X[W], 10, EF_FIXED)   # witnesses were never updated: their vector is X[w]
    persistent = sorted(int(w) for w, r in zip(W, wi) if not r or r[0] != w)
    return dict(ids=ids, ds=ds, persistent=persistent)


def changed(a, b):
    n = 0
    for ia, da, ib, db in zip(a["ids"], a["ds"], b["ids"], b["ds"]):
        if ia != ib or any(x != y for x, y in zip(da, db)):
            n += 1
    return n


def calib(trials=10, first_seed=9700):
    rows = []
    tdir = os.path.join(OUT, "trials")
    os.makedirs(tdir, exist_ok=True)
    for t in range(trials):
        seed = first_seed + t
        tpath = os.path.join(tdir, "%d.json" % seed)
        if os.path.exists(tpath):          # resumable: a finished trial is not repeated
            rows.append(json.load(open(tpath)))
            continue
        rng = np.random.default_rng([seed, 1])
        e = ChromaF(paths.work_dir("chroma_calib_%d" % seed), seed)
        t0 = time.time()
        e.start()
        try:
            perm = rng.permutation(20_000)
            live = set(int(i) for i in perm[:15_000])
            free = [int(i) for i in perm[15_000:]]
            attrs = rng.integers(0, 100, 20_000)
            init = np.sort(perm[:15_000])
            for b in range(0, 15_000, 1000):
                ids = init[b:b + 1000]
                e.write("insert", ids, X[ids], attrs[ids])
            updated = set()
            for _ in range(3):
                L = np.array(sorted(live))
                up = np.sort(rng.choice(L, 200, replace=False))
                e.write("update", up, (X[up] + rng.normal(0, 1.0, (200, 128))).astype(np.float32))
                updated |= set(int(i) for i in up)
                L = np.array(sorted(live - set(int(i) for i in up)))
                de = np.sort(rng.choice(L, 200, replace=False))
                e.write("delete", de)
                live -= set(int(i) for i in de)
                ins = np.array(sorted(free[:200]))
                free = free[200:]
                e.write("insert", ins, X[ins], attrs[ins])
                live |= set(int(i) for i in ins)
                L = np.array(sorted(live))
                sa = np.sort(rng.choice(L, 200, replace=False))
                e.write("set_attr", sa, attrs=rng.integers(0, 100, 200))
            # witnesses: live items whose vector was never updated (their vector is X[w])
            cand = np.array(sorted(live - updated))
            W = np.sort(rng.choice(cand, 2000, replace=False))
            Q = QS[rng.choice(len(QS), 200, replace=False)]
            m0 = measure(e, Q, W)
            e.clean_restart()
            m1 = measure(e, Q, W)
            e.vacuum()
            m2 = measure(e, Q, W)
            rows.append(dict(seed=seed, secs=round(time.time() - t0, 1),
                             persistent=[len(m0["persistent"]), len(m1["persistent"]), len(m2["persistent"])],
                             persistent_ids=[m0["persistent"][:20], m1["persistent"][:20], m2["persistent"][:20]],
                             restart_changed=changed(m0, m1),
                             restart_new_persistent=len(set(m1["persistent"]) - set(m0["persistent"])),
                             vacuum_changed=changed(m1, m2),
                             vacuum_new_persistent=len(set(m2["persistent"]) - set(m1["persistent"]))))
            with open(tpath, "w") as f:
                json.dump(dict(rows[-1], environment=env()), f)
            print(rows[-1], flush=True)
        finally:
            e.stop()
    preserving = [op for op in ("restart", "vacuum")
                  if all(r[op + "_changed"] == 0 and r[op + "_new_persistent"] == 0 for r in rows)]
    zero = all(sum(r["persistent"]) == 0 for r in rows)
    save("calib", dict(trials=rows, preserving=preserving, zero_rule=zero, ef_search=EF_FIXED))
    print("preserving", preserving, "zero_rule", zero)


if __name__ == "__main__":
    args = sys.argv[1:] or ["ef", "calib"]
    if "ef" in args:
        ef_probe()
    if "calib" in args:
        calib()
