"""Campaign driver (Experiments 2, 3 and 7): seeded campaign runs on one engine, with
replay of every run that raised a candidate (2 of 3 replays must raise the same key),
except runs whose every candidate has the signature of a bug of Sec. V-C: short answers
after a DELETE (ClickHouse #117286), the lost lightweight update (ClickHouse #116686),
or the Qdrant query-planner misroute.

Usage (from code/):
  python experiments/campaign.py <engine> <first_seed> <n_runs> [--tag T] [options]
  engine: pgvector | qdrant | qdrant-<version> | clickhouse | clickhouse-<version> | chroma
Results: results/campaigns/<engine>/<tag>/run_<seed>.json (and _rep<k>.json replays).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from plimsoll import paths                      # noqa: E402
from plimsoll.campaign import CampaignRun        # noqa: E402
from plimsoll.contracts import CONTRACTS         # noqa: E402

WORK = paths.work_dir("campaigns")


def _version(engine, default):
    """qdrant-1.18.3 -> 1.18.3; qdrant -> the default release."""
    return engine.split("-", 1)[1] if "-" in engine else default


def factory(engine, qdrant_full_scan=False):
    if engine == "pgvector":
        from plimsoll.adapters.pgvector_f import PgVectorF
        return (lambda d, s, dim=128: PgVectorF(d, s, dim=dim)), "pgvector_hnsw", 1000
    if engine.startswith("qdrant"):
        from plimsoll.adapters.qdrant_f import QdrantF
        binary = paths.qdrant_binary(_version(engine, paths.QDRANT_VERSION))
        assert os.path.exists(binary), binary

        def make(d, s, dim=128):
            e = QdrantF(d, s, dim=dim, root="/var/tmp/plim_qc")
            e.binary = binary
            if qdrant_full_scan:
                e.full_scan_kb = None      # keep Qdrant's default full_scan_threshold
            return e
        return make, "qdrant_hnsw", 20_000
    if engine.startswith("clickhouse"):
        from plimsoll.adapters.clickhouse_f import ClickHouseF
        binary = paths.clickhouse_binary(_version(engine, paths.CLICKHOUSE_VERSION))
        assert os.path.exists(binary), binary

        def make_ch(d, s, dim=128):
            e = ClickHouseF(d, s, dim=dim)
            e.binary = binary
            return e
        return make_ch, "clickhouse_hnsw_f32", 20_000
    if engine == "chroma":
        from plimsoll.adapters.chroma_f import EF_FIXED, ChromaF
        cal = json.load(open(os.path.join(ROOT, "results", "calib", "chroma", "calib.json")))
        ChromaF.PRESERVING = tuple(cal["preserving"])     # from experiments/chroma_calib.py
        return (lambda d, s, dim=128: ChromaF(d, s, dim=dim)), "chroma_hnsw", EF_FIXED
    raise SystemExit("unknown engine " + engine)


def zero_rule(engine):
    """WQ zero rule only where healthy runs showed no persistent miss at theta_max: Qdrant
    (0 in 157 checks of the 1.18.3 campaign before its first finding) and Chroma if its
    calibration says so. ClickHouse leaves about 0.2% of rows unreachable at any search
    effort, so its counts are only recorded."""
    if engine.startswith("qdrant"):
        return True
    if engine == "chroma":
        return bool(json.load(open(os.path.join(ROOT, "results", "calib", "chroma", "calib.json")))["zero_rule"])
    return False


def jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("first_seed", type=int)
    ap.add_argument("n_runs", type=int)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--concurrent", type=int, default=0, help="write batches by a second client during checks")
    ap.add_argument("--op-concurrent", type=int, default=0, help="write batches by a second client during maintenance operations")
    ap.add_argument("--data", default="sift", choices=["sift", "fmnist"], help="first 20,000 base vectors and the test queries")
    ap.add_argument("--no-kill", action="store_true", help="healthy runs: no process kills (Experiment 7)")
    ap.add_argument("--qdrant-full-scan", action="store_true",
                    help="keep Qdrant's default full_scan_threshold (Table III, 'kills, full scan')")
    a = ap.parse_args()
    make, cname, theta = factory(a.engine, qdrant_full_scan=a.qdrant_full_scan)
    contract = json.loads(json.dumps(CONTRACTS[cname]))
    contract["theta_max_ef"] = theta
    contract["wq_zero_rule"] = zero_rule(a.engine)
    out = os.path.join(ROOT, "results", "campaigns", a.engine, a.tag)
    os.makedirs(out, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    files = {"sift": ("sift100k_base.npy", "sift100k_queries.npy"), "fmnist": ("fmnist_base.npy", "fmnist_queries.npy")}
    base = np.load(os.path.join(ROOT, "data", files[a.data][0]))[:20_000]
    queries = np.load(os.path.join(ROOT, "data", files[a.data][1]))
    params = {} if a.steps is None else dict(STEPS=a.steps)
    params["DIM"] = int(base.shape[1])
    if a.no_kill:
        params["P_KILL"] = 0.0
    if a.op_concurrent:
        params["OP_CONCURRENT"] = a.op_concurrent
    if a.concurrent:
        params["CONCURRENT_BATCHES"] = a.concurrent

    def one(seed, tag):
        path = os.path.join(out, "run_%d%s.json" % (seed, "" if tag == "main" else "_" + tag))
        if os.path.exists(path):
            return json.load(open(path))
        t = time.time()
        dim = int(base.shape[1])
        r = CampaignRun(lambda d, s: make(d, s, dim=dim), contract, seed, (base, queries), WORK, tag=tag,
                        params=params).execute()
        r["data"] = a.data
        r = jsonable(r)
        r["engine_name"] = a.engine
        with open(path + ".tmp", "w") as f:
            json.dump(r, f)
        os.replace(path + ".tmp", path)
        print("%s %s seed %d %s: %.0fs, %d violations, %d candidates %s" % (
            time.strftime("%H:%M:%S"), a.engine, seed, tag, time.time() - t, len(r["violations"]),
            len(r["candidates"]), [c["key"] for c in r["candidates"]][:6]), flush=True)
        return r

    def known_signature(r, c):
        """An EI size candidate on ClickHouse 26.9.13.15 at or after a DELETE has the
        signature of #117286 (short answers after a lightweight DELETE, Sec. V-C4). A WQ
        zero-rule candidate on Qdrant 1.19.2 whose persistent misses are all in filtered
        self-queries, at a step with no payload index on the attribute, has the signature
        of the query-planner misroute (Sec. V-C1)."""
        if a.engine == "clickhouse" and tuple(c["key"][:2]) == ("EI", "size"):
            if any(h.get("op") == "delete" and h["step"] <= c["key"][2] for h in r["history"]):
                return True
        if a.engine == "clickhouse" and c["key"][0] == "EI":
            # from the first step at which an acknowledged concurrent lightweight UPDATE
            # (vector or attribute) was not visible right after a mutation (attr_index or
            # materialize) or OPTIMIZE, every later EI alarm follows from that lost update
            # (Sec. V-C2, ClickHouse #116686 through index materialization)
            s0 = None
            for v in r["violations"]:
                w = v.get("where", "")
                if (v["rule"] == "lookup" and w.endswith(":concurrent_lookup")
                        and w.split(":")[1] in ("attr_index", "materialize", "optimize")
                        and (v.get("got") == "other vector" or str(v.get("expected", "")).startswith("attr"))):
                    s0 = v["step"] if s0 is None else min(s0, v["step"])
            return s0 is not None and c["key"][2] >= s0
        if a.engine == "qdrant" and tuple(c["key"][:2]) == ("WQ", "wq_persistent_zero"):
            s = c["key"][2]
            vs = [v for v in r["violations"] if v["rule"] == "wq_persistent_zero" and v["step"] == s]
            if not vs or not all(v["where"].endswith(":filtered") for v in vs):
                return False
            last = None
            for h in r["history"]:
                if h.get("op") in ("attr_index", "attr_index_drop") and h["step"] <= s:
                    last = h
            return last is None or (last["op"] == "attr_index_drop" and last["action"] == "op"
                                    and last.get("err") is None)
        return False

    for seed in range(a.first_seed, a.first_seed + a.n_runs):
        r = one(seed, "main")
        if not r["candidates"]:
            continue
        rep0 = os.path.join(out, "run_%d_rep0.json" % seed)
        if all(known_signature(r, c) for c in r["candidates"]) and not os.path.exists(rep0):
            for c in r["candidates"]:
                c["replay"], c["confirmed"] = None, None
                c["replay_skipped"] = ("known signature: #117286 or lost update" if a.engine == "clickhouse"
                                       else "known signature: planner misroute")
            with open(os.path.join(out, "run_%d.json" % seed), "w") as f:
                json.dump(r, f)
            print("  replay skipped: every candidate has a known signature", flush=True)
            continue
        reps = [one(seed, "rep%d" % k) for k in range(3)]
        keys = [set(tuple(c["key"]) for c in x["candidates"]) for x in reps]
        for c in r["candidates"]:
            c["replay"] = sum(tuple(c["key"]) in k for k in keys)
            c["confirmed"] = c["replay"] >= 2
        with open(os.path.join(out, "run_%d.json" % seed), "w") as f:
            json.dump(r, f)
        print("  replay:", [(c["key"], c["replay"]) for c in r["candidates"]], flush=True)


if __name__ == "__main__":
    main()
