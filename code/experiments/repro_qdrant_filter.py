"""Triage of Experiment 7 candidates (qdrant/exp7_sift seed 1001): filtered self-queries on
Qdrant 1.19.2 that miss at every hnsw_ef, while lookups show the item live with the
filtered value.

Directed run, campaign collection settings (m 16, ef_construct 100, indexing_threshold
1,000 kB, full_scan_threshold 1,000 kB, no payload index): 15,000 SIFT vectors with payload
cat drawn uniformly from 0 to 99 (seeded), loaded in batches of 1,000.
States: A after the load; B after inserting 110 more points and a rebuild (ef_construct
101, then 100, waiting for green each time, as the campaign's rebuild); C after creating
an integer payload index on cat.
In each state, for every value c: self-queries of up to 5 live items with cat = c,
filtered on cat = c, k = 10, at hnsw_ef 64 and 20,000 and with exact = true. Recorded:
answers shorter than min(10, number of live items with cat = c), self-misses, and the
collection's segment and index counts.
Options: --data sift|fmnist, --n N items loaded (default 15,000), --full-scan-kb KB
(default 1,000; 0 keeps Qdrant's defaults for both thresholds), --seed, --binary.
Usage (from code/): python experiments/repro_qdrant_filter.py [options]
Writes results/repro/qdrant_filter/<version>_<data>_n<N>_fs<KB>_seed<seed>.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from plimsoll.adapters import qdrant as qbase  # noqa: E402
from plimsoll.adapters.qdrant_f import QdrantF  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
K, EF_LOW, EF_MAX, PER_VALUE = 10, 64, 20_000, 5
QUERIES = None


def measure(eng, live, cat, base, rng):
    out = dict(values=[], short=dict(ef64=0, ef_max=0, exact=0), self_miss=dict(ef64=0, ef_max=0, exact=0),
               queries=0)
    for c in range(100):
        ids = np.array(sorted(i for i in live if cat[i] == c))
        if len(ids) == 0:
            continue
        pick = rng.choice(ids, size=min(PER_VALUE, len(ids)), replace=False)
        expect = min(K, len(ids))
        row = dict(c=c, eligible=int(len(ids)), queries=int(len(pick)))
        for name, kw in (("ef64", dict(ef=EF_LOW)), ("ef_max", dict(ef=EF_MAX)), ("exact", dict(ef=EF_LOW, exact=True))):
            res, _ = eng.search(base[pick], K, filt=("eq", c), **kw)
            short = sum(len(r) < expect for r in res)
            miss = sum(int(i) not in set(r) for i, r in zip(pick, res))
            row[name] = dict(short=short, self_miss=miss, sizes=sorted({len(r) for r in res}),
                             answers=[[int(x) for x in r] for r in res] if (short or miss) else None)
            out["short"][name] += short
            out["self_miss"][name] += miss
        out["queries"] += int(len(pick))
        out["values"].append(row)
    info = eng.client.get_collection(qbase.COLL)
    out["collection"] = dict(points=info.points_count, indexed_vectors=info.indexed_vectors_count,
                             segments=info.segments_count, status=str(info.status),
                             payload_schema=sorted((info.payload_schema or {}).keys()))
    out["segments"] = segments(eng)
    out["b2"] = b2(eng, live, cat, np.random.default_rng(12345))  # same queries and filters in every state
    return out


def b2(eng, live, cat, rng, n_q=1000):
    """What a recall monitor would see: 1,000 test queries, each with a random filter as in
    the campaigns (none, cat = c, cat < 10, cat < 50), k = 10, hnsw_ef 64; recall@10 against
    the engine's exact path (exact = true), overall and per filter kind."""
    Q = QUERIES[:n_q]
    kinds = rng.integers(0, 4, size=n_q)
    cs = rng.integers(0, 100, size=n_q)
    rec = {k: [] for k in ("none", "eq", "lt10", "lt50")}
    for kind in range(4):
        idx = np.nonzero(kinds == kind)[0]
        if kind == 1:  # one request per value of c
            groups = {}
            for i in idx:
                groups.setdefault(int(cs[i]), []).append(i)
        else:
            groups = {None: list(idx)}
        for c, ii in groups.items():
            filt = [None, ("eq", c), ("lt", 10), ("lt", 50)][kind]
            a, _ = eng.search(Q[ii], K, filt=filt, ef=EF_LOW)
            e, _ = eng.search(Q[ii], K, filt=filt, ef=EF_LOW, exact=True)
            name = ["none", "eq", "lt10", "lt50"][kind]
            for x, y in zip(a, e):
                if y:
                    rec[name].append(len(set(x) & set(y)) / len(y))
    allr = [r for v in rec.values() for r in v]
    return dict(recall=float(np.mean(allr)), n=len(allr),
                by_kind={k: (float(np.mean(v)) if v else None) for k, v in rec.items()},
                below_half=sum(r < 0.5 for r in allr))


def segments(eng):
    """Per-segment point and indexed-vector counts from Qdrant's telemetry endpoint."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/telemetry?details_level=6" % eng.http, timeout=30) as r:
            t = json.load(r)["result"]
        segs = []
        for c in t["collections"]["collections"]:
            for sh in c.get("shards") or []:
                for sg in ((sh.get("local") or {}).get("segments") or []):
                    i = sg.get("info", {})
                    segs.append(dict(type=i.get("segment_type"), points=i.get("num_points"),
                                     vectors=i.get("num_vectors"), indexed=i.get("num_indexed_vectors")))
        return segs
    except Exception as e:  # diagnostics only
        return repr(e)[:200]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=qbase.QDRANT_BIN)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", choices=("sift", "fmnist"), default="sift")
    ap.add_argument("--n", type=int, default=15_000)
    ap.add_argument("--full-scan-kb", type=int, default=1000)
    a = ap.parse_args(argv[1:])
    version = subprocess.run([a.binary, "--version"], capture_output=True, text=True).stdout.strip().split()[-1]
    rng = np.random.default_rng(a.seed)
    fn = {"sift": "sift100k_base.npy", "fmnist": "fmnist_base.npy"}[a.data]
    base = np.load(os.path.join(ROOT, "data", fn))[:a.n + 1_000].astype(np.float32)
    global QUERIES
    QUERIES = np.load(os.path.join(ROOT, "data", fn.replace("_base", "_queries")))[:1000].astype(np.float32)
    cat = rng.integers(0, 100, size=len(base))
    perm = rng.permutation(len(base))
    tag = "%s_%s_n%d_fs%d_seed%d" % (version, a.data, a.n, a.full_scan_kb, a.seed)
    eng = QdrantF("/var/tmp/plim_qd/repro_filter_" + tag, a.seed, dim=int(base.shape[1]))
    eng.binary = a.binary
    eng.start()
    if a.full_scan_kb:
        eng.full_scan_kb = a.full_scan_kb
        eng.configure_campaign()
    live = set()
    t0 = time.time()
    for b in range(0, a.n, 1_000):
        ids = perm[b:min(b + 1_000, a.n)]
        eng.write("insert", ids, base[ids], cat[ids])
        live.update(int(i) for i in ids)
    eng.quiesce()
    cfg = eng.client.get_collection(qbase.COLL).config
    res = dict(version=version, seed=a.seed, binary=a.binary, data=a.data, n=a.n, settings=dict(
        m=cfg.hnsw_config.m, ef_construct=cfg.hnsw_config.ef_construct,
        full_scan_threshold_kb=cfg.hnsw_config.full_scan_threshold,
        indexing_threshold_kb=cfg.optimizer_config.indexing_threshold,
        default_segment_number=cfg.optimizer_config.default_segment_number,
        k=K, ef=[EF_LOW, EF_MAX], per_value=PER_VALUE), states={})
    res["states"]["A_load"] = measure(eng, live, cat, base, np.random.default_rng(a.seed + 1))
    ids = perm[a.n:a.n + 110]
    eng.write("insert", ids, base[ids], cat[ids])
    live.update(int(i) for i in ids)
    eng.rebuild(0)
    eng.quiesce()
    res["states"]["B_insert_rebuild"] = measure(eng, live, cat, base, np.random.default_rng(a.seed + 1))
    eng.attr_index()
    eng.quiesce()
    res["states"]["C_payload_index"] = measure(eng, live, cat, base, np.random.default_rng(a.seed + 1))
    res["secs"] = round(time.time() - t0, 1)
    eng.stop()
    out = os.path.join(ROOT, "results", "repro", "qdrant_filter")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, tag + ".json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    for st, m in res["states"].items():
        bad = [(v["c"], v["eligible"], v["ef_max"]["sizes"]) for v in m["values"] if v["ef_max"]["short"]]
        print(st, "queries", m["queries"], "short", m["short"], "self_miss", m["self_miss"], m["collection"], "values", bad)
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv)
