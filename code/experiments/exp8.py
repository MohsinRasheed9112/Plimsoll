"""Experiment 8: what checking costs (Sec. V-F2 and V-F3).

Measures, on the machine it runs on:
  1. oracle cost: EI on 1,000 answers (k = 10), WQ hit test on 2,000 witnesses, recall with
     the near-tie rule on 1,000 answers, against an hnswlib index over SIFT 100K;
  2. ground truth: exact top-10 for 1,000 queries over SIFT 100K, SIFT 1M (128-d) and
     Fashion-MNIST 60K (784-d) (plimsoll/gt.py), with and without the second float64 path;
  3. engine capacity: queries per second at ef = 64 for one and two client threads on each
     engine adapter given on the command line, over SIFT 100K (or the campaign's 20K);
  4. audit overhead: 2,000 witness queries per hour (0.56 per second), with their measured
     escalation rate, as a share of the measured capacity.
  5. ground truth at scale (32-vCPU cloud instance): certified exact top-10 for 1,000
     queries over SIFT 1M (128-d) and GIST 1M (960-d, ANN-Benchmarks
     gist-960-euclidean.hdf5, fetched by scripts/get_data.py gist).
Usage: python experiments/exp8.py [oracles] [gt] [gt_scale] [capacity pgvector qdrant clickhouse]
Writes results/exp8/<part>.json. Run on an otherwise idle machine.
"""
import json
import os
import platform
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from plimsoll import gt as G  # noqa: E402
from plimsoll import oracles as O  # noqa: E402
from plimsoll import paths  # noqa: E402
from plimsoll.contracts import CONTRACTS  # noqa: E402
from plimsoll.model import LIVE, Model  # noqa: E402

OUT = os.path.join(ROOT, "results", "exp8")


def env():
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                cpus=os.cpu_count(), machine=platform.machine(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                load=os.getloadavg())


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    obj["environment"] = env()
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print(json.dumps(obj, indent=1)[:3000])


def best(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return dict(min=min(ts), median=float(np.median(ts)), reps=reps)


def oracles():
    import hnswlib
    X = np.load(os.path.join(ROOT, "data", "sift100k_base.npy"))
    Q = np.load(os.path.join(ROOT, "data", "sift100k_queries.npy"))
    m = Model(len(X), X.shape[1])
    m.status[:] = LIVE
    m.vec[:] = X
    idx = hnswlib.Index("l2", 128)
    idx.init_index(len(X), M=16, ef_construction=100, random_seed=1)
    idx.add_items(X, np.arange(len(X)), num_threads=2)
    idx.set_ef(64)
    c = CONTRACTS["hnsw"]
    lab, dis = idx.knn_query(Q, k=10)
    ids, ds = [r.astype(np.int64) for r in lab], list(dis)
    W = np.random.default_rng(1).choice(len(X), 2000, replace=False)
    wl, _ = idx.knn_query(X[W], k=10)
    first = wl[:, 0].astype(np.int64)
    gt = G.compute(Q, m.vec, np.arange(len(X)), 10, c)
    out = dict(
        ei_1000_answers=best(lambda: O.ei_check_search(m, c, Q, ids, ds, 10, "q")),
        wq_hits_2000=best(lambda: O.wq_hits(m, c, W, first)),
        recall_1000=best(lambda: O.recall_per_query(m, c, Q, ids, gt, 10)),
        search_1000_ef64_2threads=best(lambda: idx.knn_query(Q, k=10, num_threads=2)))
    out["per_answer_us"] = dict(ei=1e6 * out["ei_1000_answers"]["median"] / 1000,
                                wq=1e6 * out["wq_hits_2000"]["median"] / 2000,
                                recall=1e6 * out["recall_1000"]["median"] / 1000,
                                search=1e6 * out["search_1000_ef64_2threads"]["median"] / 1000)
    save("oracles", out)


def gt_cost():
    c = CONTRACTS["hnsw"]
    out = {}
    for name, files, qf in (("sift100k", ["sift100k_base.npy"], "sift100k_queries.npy"),
                            ("sift1m", ["sift1m_base_part%d.npy" % i for i in range(4)], "sift1m_queries.npy"),
                            ("fmnist60k", ["fmnist_base.npy"], "fmnist_queries.npy")):
        X = np.concatenate([np.load(os.path.join(ROOT, "data", f)) for f in files])
        Q = np.load(os.path.join(ROOT, "data", qf))[:1000]
        live = np.arange(len(X))
        t0 = time.perf_counter()
        G.compute(Q, X, live, 10, c, n_check=0)
        t1 = time.perf_counter()
        G.compute(Q, X, live, 10, c, n_check=20, rng=np.random.default_rng(1))
        t2 = time.perf_counter()
        out[name] = dict(n=len(X), dim=X.shape[1], queries=len(Q), secs_certified=t1 - t0,
                         secs_with_second_path_20=t2 - t1)
        del X
    save("ground_truth", out)


def gt_scale():
    """Certified ground truth over SIFT 1M and GIST 1M, timed as secs_certified in gt_cost
    (one run each, no second path), on the machine it runs on."""
    import h5py
    c = CONTRACTS["hnsw"]
    out = {}
    for name in ("sift1m", "gist1m"):
        if name == "sift1m":
            X = np.concatenate([np.load(os.path.join(ROOT, "data", "sift1m_base_part%d.npy" % i)) for i in range(4)])
            Q = np.load(os.path.join(ROOT, "data", "sift1m_queries.npy"))[:1000]
        else:
            with h5py.File(os.path.join(ROOT, "data", "gist-960-euclidean.hdf5"), "r") as f:
                X = np.ascontiguousarray(np.asarray(f["train"], dtype=np.float32))
                Q = np.ascontiguousarray(np.asarray(f["test"], dtype=np.float32)[:1000])
        live = np.arange(len(X))
        t0 = time.perf_counter()
        g = G.compute(Q, X, live, 10, c, n_check=0)
        t1 = time.perf_counter()
        out[name] = dict(n=len(X), dim=X.shape[1], queries=len(Q), secs_certified=t1 - t0,
                         n_fallback=int(g["n_fallback"]))
        del X
    save("ground_truth_scale", out)


def capacity(engines):
    sys.path.insert(0, HERE)
    from campaign import factory
    X = np.load(os.path.join(ROOT, "data", "sift100k_base.npy"))[:20000]
    Q = np.load(os.path.join(ROOT, "data", "sift100k_queries.npy"))
    out = {}
    for e in engines:
        make, cname, theta = factory(e)
        eng = make(paths.work_dir("exp8_" + e), 1)
        try:
            eng.start()
            if hasattr(eng, "configure_campaign"):   # Qdrant: HNSW path, as in the campaigns
                eng.configure_campaign()
            rng = np.random.default_rng(8)
            for b in range(0, 20000, 1000):
                eng.write("insert", np.arange(b, b + 1000), X[b:b + 1000], rng.integers(0, 100, 1000))
            if hasattr(eng, "after_initial_load"):
                eng.after_initial_load()
            if hasattr(eng, "quiesce"):
                eng.quiesce()
            eng.search(Q[:50], 10, 64)   # warm up
            res = {}
            for threads in (1, 2):
                writers = [eng] + [eng.concurrent_writer() for _ in range(threads - 1)]
                n_per = 500
                t0 = time.perf_counter()
                th = [threading.Thread(target=lambda w=w, j=j, n=n_per: w.search(Q[j * n:(j + 1) * n], 10, 64))
                      for j, w in enumerate(writers)]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                secs = time.perf_counter() - t0
                res["qps_%d_client" % threads] = n_per * threads / secs
                for w in writers[1:]:
                    if hasattr(w, "close") and w is not eng:
                        w.close()
            # escalation: self-queries at ef = 64, misses re-asked at theta_max
            W = rng.choice(20000, 300, replace=False)
            misses = 0
            for w in W:
                ids, _ = eng.search(X[w][None, :], 10, 64)
                misses += int(not ids[0] or ids[0][0] != w)
            res["self_query_miss_rate_ef64"] = misses / len(W)
            cap = max(res["qps_1_client"], res["qps_2_client"])
            rate = 2000 / 3600 * (1 + res["self_query_miss_rate_ef64"])
            res["audit_queries_per_sec"] = rate
            res["audit_overhead_share"] = rate / cap
            out[e] = res
        finally:
            eng.stop()
    save("capacity", out)


if __name__ == "__main__":
    args = sys.argv[1:] or ["oracles", "gt"]
    if "oracles" in args:
        oracles()
    if "gt" in args:
        gt_cost()
    if "gt_scale" in args:
        gt_scale()
    if "capacity" in args:
        capacity([a for a in args if a in ("pgvector", "qdrant", "clickhouse")])
