"""Triage of ClickHouse exp3o seed 850 (step 18): acknowledged lightweight UPDATEs that a
second client made while ALTER TABLE ... ADD INDEX / MATERIALIZE INDEX ran were not
visible afterwards.

Directed run on ClickHouse 26.9.13.15 with the campaign adapter's table (vector index
hnsw f32, MergeTree ordered by id, block number and offset columns, lightweight UPDATE):
load 15,000 SIFT rows in batches of 1,000; then, in each trial, one of
  attr_index   ADD INDEX cidx cat TYPE minmax + MATERIALIZE INDEX cidx (mutations_sync 2)
  materialize  MATERIALIZE INDEX vidx (mutations_sync 2)
  optimize     OPTIMIZE TABLE items FINAL
  none         no operation (control)
  add_index    ADD INDEX cidx only (a metadata change, no MATERIALIZE)
  materialize_cidx  cidx added before the updates; MATERIALIZE INDEX cidx alongside them
runs while a second client sends 5 lightweight UPDATEs of the vectors of 100 rows each.
After both finish, every updated row is read back and compared with its old and new
vector: right after, after 10 s, and after a final OPTIMIZE TABLE FINAL.
Usage (from code/): python experiments/repro_clickhouse_concurrent_update.py <mode> <trials> [seed]
Writes results/repro/clickhouse_concurrent_update/<mode>_seed<seed>.json
"""
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from plimsoll.adapters.clickhouse_f import ClickHouseF  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))


def classify(eng, ids, old, new):
    live, vals = eng.lookup(ids)
    out = dict(new=0, old=0, other=0, absent=0)
    for i, l, v in zip(ids, live, vals):
        if not l:
            out["absent"] += 1
            continue
        vec = np.asarray(v[0], dtype=np.float32)
        if np.array_equal(vec, new[i]):
            out["new"] += 1
        elif np.array_equal(vec, old[i]):
            out["old"] += 1
        else:
            out["other"] += 1
    return out


def trial(mode, seed, base):
    rng = np.random.default_rng(seed)
    eng = ClickHouseF("/var/tmp/plim_ch/repro_cu_%s_%d" % (mode, seed), seed, dim=128)
    eng.start()
    cat = rng.integers(0, 100, size=15_000)
    for b in range(0, 15_000, 1_000):
        ids = np.arange(b, b + 1_000)
        eng.write("insert", ids, base[ids], cat[ids])
    eng.quiesce()
    upd = rng.choice(15_000, size=500, replace=False)
    newv = {int(i): (base[int(i)] + rng.normal(0, 1.0, size=128)).astype(np.float32) for i in upd}
    oldv = {int(i): base[int(i)].astype(np.float32) for i in upd}
    acks = []
    if mode == "materialize_cidx":   # the minmax index exists before the updates; only MATERIALIZE runs alongside
        eng._q("ALTER TABLE items ADD INDEX IF NOT EXISTS cidx cat TYPE minmax GRANULARITY 1")
    op = {"attr_index": eng.attr_index, "materialize": eng.materialize, "optimize": eng.optimize,
          "none": (lambda: None),
          "add_index": (lambda: eng._q("ALTER TABLE items ADD INDEX IF NOT EXISTS cidx cat TYPE minmax GRANULARITY 1")),
          "materialize_cidx": (lambda: eng._q("ALTER TABLE items MATERIALIZE INDEX cidx",
                                               settings={"mutations_sync": 2}))}[mode]
    err = {}

    def run_op():
        try:
            op()
        except Exception as e:  # recorded, not raised
            err["op"] = repr(e)[:300]

    t0 = time.time()
    th = threading.Thread(target=run_op)
    th.start()
    for b in range(5):
        ids = upd[b * 100:(b + 1) * 100]
        try:
            eng.write("update", ids, np.stack([newv[int(i)] for i in ids]))
            acks.append(True)
        except Exception as e:
            acks.append(repr(e)[:200])
    th.join()
    secs = round(time.time() - t0, 2)
    acked = [int(i) for b in range(5) if acks[b] is True for i in upd[b * 100:(b + 1) * 100]]
    res = dict(mode=mode, seed=seed, op_secs=secs, acks=acks, op_error=err.get("op"), acked_rows=len(acked))
    res["right_after"] = classify(eng, acked, oldv, newv)
    time.sleep(10)
    res["after_10s"] = classify(eng, acked, oldv, newv)
    eng.optimize()
    res["after_optimize"] = classify(eng, acked, oldv, newv)
    res["parts"] = eng._q("SELECT name, rows FROM system.parts WHERE table = 'items' AND active FORMAT JSONEachRow")
    res["mutations"] = eng._q("SELECT mutation_id, command, is_done, latest_fail_reason FROM system.mutations "
                              "WHERE table = 'items' FORMAT JSONEachRow")
    try:
        with open(os.path.join(eng.root, "log", "err.log")) as f:
            res["err_log"] = [l.strip()[:400] for l in f if "Error" in l or "Exception" in l][-30:]
    except OSError as e:
        res["err_log"] = repr(e)
    eng.stop()
    return res


def main(argv):
    mode, n = argv[1], int(argv[2])
    seed0 = int(argv[3]) if len(argv) > 3 else 1
    base = np.load(os.path.join(ROOT, "data", "sift100k_base.npy"))[:15_000].astype(np.float32)
    out = os.path.join(ROOT, "results", "repro", "clickhouse_concurrent_update")
    os.makedirs(out, exist_ok=True)
    for s in range(seed0, seed0 + n):
        r = trial(mode, s, base)
        with open(os.path.join(out, "%s_seed%d.json" % (mode, s)), "w") as f:
            json.dump(r, f, indent=1, default=str)
        print(mode, s, "acked", r["acked_rows"], "right_after", r["right_after"], "after_10s", r["after_10s"],
              "after_optimize", r["after_optimize"], "op_error", r["op_error"], flush=True)


if __name__ == "__main__":
    main(sys.argv)
