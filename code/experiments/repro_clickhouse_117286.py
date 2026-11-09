"""Directed run for ClickHouse issue #117286 with Plimsoll's contract rule for documented
exact search (Experiment 2, directed mode).

The ClickHouse documentation says that parts with deleted rows are read in full (exact
search). Scenario: insert 15,000 SIFT rows, OPTIMIZE FINAL (one part), check 200
queries; lightweight-DELETE 100 rows; check the same 200 queries again. After the delete
every part has deleted rows, so every unfiltered answer must be exact: min(k, live) rows,
with the i-th distance equal to the i-th exact distance within tolerance.

Usage: python experiments/repro_clickhouse_117286.py <clickhouse-version>
Writes results/repro/clickhouse_117286/<version>.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from plimsoll import paths  # noqa: E402
from plimsoll.adapters.clickhouse_f import ClickHouseF  # noqa: E402
from plimsoll.contracts import CONTRACTS  # noqa: E402
from plimsoll.tolerance import distances_and_eps  # noqa: E402


def check(eng, Q, vec, live, contract, k=10):
    out = dict(answers=0, short=0, wrong_rank=0, examples=[])
    ids, ds = eng.search(Q, k, 64)
    for q, i_, d_ in zip(Q, ids, ds):
        out["answers"] += 1
        delta, eps = distances_and_eps(q, vec[live], contract)
        order = np.argsort(delta, kind="stable")[:k]
        if len(i_) != min(k, len(live)):
            out["short"] += 1
            if len(out["examples"]) < 3:
                out["examples"].append(dict(rule="size", got=len(i_), ids=[int(x) for x in i_]))
            continue
        for r in range(k):
            if abs(d_[r] - delta[order[r]]) > 2.0 * eps[order[r]] + 1e-9:
                out["wrong_rank"] += 1
                if len(out["examples"]) < 3:
                    out["examples"].append(dict(rule="rank", rank=r, reported=d_[r], exact=float(delta[order[r]])))
                break
    return out


def main(label):
    binary = paths.clickhouse_binary(label)
    X = np.load(os.path.join(ROOT, "data", "sift100k_base.npy"))[:15000]
    Q = np.load(os.path.join(ROOT, "data", "sift100k_queries.npy"))[:200]
    contract = json.loads(json.dumps(CONTRACTS["clickhouse_hnsw_f32"]))
    rng = np.random.default_rng(117286)
    eng = ClickHouseF(paths.work_dir("ch117286_" + label), 1)
    eng.binary = binary
    res = dict(label=label, binary=binary)
    try:
        eng.start()
        res["version"] = eng._q("SELECT version()").strip()
        for b in range(0, 15000, 1000):
            eng.write("insert", np.arange(b, b + 1000), X[b:b + 1000], rng.integers(0, 100, 1000))
        eng.optimize()
        eng.quiesce()
        live = np.arange(15000)
        res["before_delete"] = dict(exactness_promised=eng.exactness_promised(),
                                    parts=eng._q("SELECT count() FROM system.parts WHERE table='items' AND active").strip(),
                                    **check(eng, Q, X, live, contract))
        dead = np.sort(rng.choice(15000, 100, replace=False))
        eng.write("delete", dead)
        live = np.setdiff1d(live, dead)
        res["after_delete"] = dict(exactness_promised=eng.exactness_promised(),
                                   parts=eng._q("SELECT count() FROM system.parts WHERE table='items' AND active").strip(),
                                   **check(eng, Q, X, live, contract))
    finally:
        eng.stop()
    d = os.path.join(ROOT, "results", "repro", "clickhouse_117286")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, label + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:2500])


if __name__ == "__main__":
    main(sys.argv[1])
