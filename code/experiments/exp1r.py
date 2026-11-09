"""Experiment 1 on real engines (Sec. V-B2, Table I): calibration, kill calibration,
soundness with kills, replays and baseline B1, with the Experiment 1 template.

Usage (from code/):  python experiments/exp1r.py <engine> [phase ...]
  engine: pgvector | qdrant
  phases: calib killcal envelopes sound (default: all, in this order)
  smoke:  python experiments/exp1r.py <engine> smoke <seed> [kill]
Every run is saved under results/exp1r/<engine>/runs/; existing runs are not repeated.
"""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from plimsoll import paths                               # noqa: E402
from plimsoll import stats as S                          # noqa: E402
from plimsoll.contracts import CONTRACTS                 # noqa: E402
from plimsoll.runner import OPS, P, run_once             # noqa: E402
from exp1 import B1_POINTS, _jsonable, _keys, _stats     # noqa: E402

WORK = paths.work_dir("exp1r")


def _pg(run_dir, seed):
    from plimsoll.adapters.pgvector import PgVector
    return PgVector(run_dir, seed)


def _qd(run_dir, seed):
    from plimsoll.adapters.qdrant import Qdrant
    return Qdrant(run_dir, seed)


ENGINES = {
    "pgvector": dict(factory=_pg, contract="pgvector_hnsw", theta=1000,
                     seeds=dict(calib=range(6000, 6020), killcal=range(6500, 6520), sound=range(6100, 6130))),
    "qdrant": dict(factory=_qd, contract="qdrant_hnsw", theta=P["CAP"],
                   seeds=dict(calib=range(7000, 7020), killcal=range(7500, 7520), sound=range(7100, 7130))),
}


class Ctx:
    def __init__(self, engine):
        self.engine = engine
        self.cfg = ENGINES[engine]
        self.res = os.path.join(ROOT, "results", "exp1r", engine)
        self.runs = os.path.join(self.res, "runs")
        os.makedirs(self.runs, exist_ok=True)
        os.makedirs(WORK, exist_ok=True)
        c = json.loads(json.dumps(CONTRACTS[self.cfg["contract"]]))
        c["theta_max_ef"] = self.cfg["theta"]
        self.contract = c
        self.data = (np.load(os.path.join(ROOT, "data", "sift100k_base.npy")),
                     np.load(os.path.join(ROOT, "data", "sift100k_queries.npy")))

    def log(self, msg):
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        with open(os.path.join(self.res, "progress.log"), "a") as f:
            f.write(line + "\n")

    def run_cached(self, name, **kw):
        path = os.path.join(self.runs, name + ".json")
        if os.path.exists(path):
            return json.load(open(path))
        t = time.time()
        mode = kw.pop("mode", self.engine)
        res = run_once(mode=mode, workdir=WORK, data=self.data, engine_factory=self.cfg["factory"],
                       contract=self.contract, **kw)
        res = _jsonable(res)
        res["engine"] = self.engine
        with open(path + ".tmp", "w") as f:
            json.dump(res, f)
        os.replace(path + ".tmp", path)
        self.log("run %s done in %.1fs: raw=%s candidates=%d" % (name, time.time() - t, res["raw"],
                                                                 len(res["candidates"])))
        return res


def calibrate(ctx, phase, kill):
    out = []
    for seed in ctx.cfg["seeds"][phase]:
        r = ctx.run_cached("%s_%d" % (phase, seed), seed=seed, kill=kill)
        if r["violations"]:
            ctx.log("%s run %d has EI violations; they are candidates for triage" % (phase, seed))
        out.append(r)
    return out


def build_envelopes(ctx):
    calib = [json.load(open(os.path.join(ctx.runs, "calib_%d.json" % s))) for s in ctx.cfg["seeds"]["calib"]]
    kc = [json.load(open(os.path.join(ctx.runs, "killcal_%d.json" % s))) for s in ctx.cfg["seeds"]["killcal"]]
    env = {"index_preserving": {}, "stats": {}}
    for name, kind in OPS:
        recs = [r["ops"][name] for r in calib]
        env["index_preserving"][kind] = all(len(x["se_changed"]) == 0 and len(x["wq_new_persistent"]) == 0
                                            for x in recs)
        env["stats"][kind] = _stats(recs)
    env["b1"] = {p: [r["points"][p]["recall"] for r in calib] for p in B1_POINTS}
    env["stats_crash"] = {k: _stats([r["ops"][k] for r in kc]) for k in ("k1", "k2", "k3")}
    env["b1_kill"] = {p: [r["points"][p]["recall"] for r in kc] for p in B1_POINTS}
    env["killcal_index_preserving_check"] = {
        name: all(len(r["ops"][name]["se_changed"]) == 0 and len(r["ops"][name]["wq_new_persistent"]) == 0
                  for r in kc) for name, _ in OPS}
    env["calib_ei_violation_runs"] = [r["seed"] for r in calib + kc if r["violations"]]
    with open(os.path.join(ctx.res, "envelopes.json"), "w") as f:
        json.dump(_jsonable(env), f, indent=1)
    return env


def replay(ctx, prefix, res, env):
    """As in Experiment 1: 2 of 3 replays; controls for confirmed exact-mode candidates
    (a real engine has no injected defect, so a control is a replay of the same workload
    with a new build); fresh seeds for statistical candidates."""
    cands = res["candidates"]
    if not cands:
        return res
    seed = res["seed"]
    same = [c for c in cands if c["mode"] in ("ei", "exact")]
    stat = [c for c in cands if c["mode"] in ("stat", "stat_zero")]
    if same:
        reps = [ctx.run_cached("%s_rep%d" % (prefix, r), seed=seed, kill=res["kill"],
                               build_seed=res["build_seed"] + 10_000 * (r + 1), envelopes=env,
                               tag="rep%d" % r) for r in range(3)]
        rk = [_keys(x) for x in reps]
        for c in same:
            c["replay"] = sum(tuple(c["key"]) in k for k in rk)
            c["confirmed"] = c["replay"] >= 2
            c["counted"] = c["confirmed"]
    if stat:
        reps = [ctx.run_cached("%s_srep%d" % (prefix, r), seed=seed + 100_000 * (r + 1),
                               kill=res["kill"], envelopes=env, tag="srep%d" % r) for r in range(3)]
        for c in stat:
            if c["mode"] == "stat_zero":
                c["replay"] = sum(tuple(c["key"]) in _keys(x) for x in reps)
            else:
                key = "%s|%s" % (c["key"][1], c["key"][2])
                c["replay"] = sum(x["tests"].get(key, 1.0) <= c["level"] for x in reps)
            c["confirmed"] = c["replay"] >= 2
            c["counted"] = c["confirmed"]
    return res


def b1_alarm(res, env):
    ref = env["b1_kill"] if res["kill"] else env["b1"]
    ps = {p: S.mean_pvalue(res["points"][p]["recall"], ref[p], lower=True) for p in B1_POINTS}
    return sorted(S.holm(ps, P["ALPHA_FW"])), ps


def environment(ctx):
    import importlib.metadata as md
    lines = ["date_utc: %s" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "platform: %s" % platform.platform(), "python: %s" % sys.version.split()[0],
             "numpy: %s" % np.__version__, "nproc: %s" % os.cpu_count()]
    if ctx.engine == "pgvector":
        v = subprocess.run([os.path.join(paths.PG_BIN, "postgres"), "--version"], capture_output=True,
                           text=True).stdout.strip()
        lines += ["server: %s" % v, "pgvector: 0.8.6 (source tarball sha256 "
                  "10bf9938906e5d643bbc4a7eea104b6f57ba4898e5b76b20e60484ea1d5a7f8f, built with "
                  "PGXS defaults)", "psycopg: %s" % md.version("psycopg"),
                  "pgvector-python: %s" % md.version("pgvector")]
    else:
        from plimsoll.adapters.qdrant import QDRANT_BIN
        v = subprocess.run([QDRANT_BIN, "--version"], capture_output=True, text=True).stdout.strip()
        lines += ["server: %s (release binary x86_64-unknown-linux-gnu, tarball sha256 "
                  "34a788a09a4cb278b5c6d2af96d9a6db30c8c7e88ae3550455f7a4d425bb8d4b)" % v,
                  "qdrant-client: %s" % md.version("qdrant-client")]
    return "\n".join(lines) + "\n"


def main(argv):
    ctx = Ctx(argv[1])
    phases = argv[2:] or ["calib", "killcal", "envelopes", "sound"]
    if phases[0] == "smoke":
        seed = int(phases[1])
        assert seed >= 9000
        kill = len(phases) > 2 and phases[2] == "kill"
        r = ctx.run_cached("smoke_%d%s" % (seed, "_kill" if kill else ""), seed=seed, kill=kill)
        print(json.dumps({k: r[k] for k in ("raw", "decisions", "times", "kills")}, indent=1))
        for op, rec in r["ops"].items():
            print(op, rec["kind"], "changed", len(rec["se_changed"]), "newpersist", len(rec["wq_new_persistent"]),
                  "D", rec["se_D"], "tail", rec["se_tail"], "wq_count", rec["wq_count"])
        for p, x in r["points"].items():
            print(p, x)
        print("violations", r["violations"][:5])
        return
    with open(os.path.join(ctx.res, "environment.txt"), "a") as f:
        f.write(environment(ctx))
    if "calib" in phases:
        calibrate(ctx, "calib", False)
    if "killcal" in phases:
        calibrate(ctx, "killcal", True)
    if "envelopes" in phases:
        env = build_envelopes(ctx)
        ctx.log("envelopes: index_preserving=%s killcal=%s ei_runs=%s" % (
            env["index_preserving"], env["killcal_index_preserving_check"], env["calib_ei_violation_runs"]))
    if "sound" in phases:
        env = json.load(open(os.path.join(ctx.res, "envelopes.json")))
        seeds = list(ctx.cfg["seeds"]["sound"])
        if os.environ.get("PLIMSOLL_SOUND_SEEDS"):   # orchestration only: a sub-range, "first-last"
            lo, hi = (int(x) for x in os.environ["PLIMSOLL_SOUND_SEEDS"].split("-"))
            seeds = [x for x in seeds if lo <= x <= hi]
        for seed in seeds:
            name = "sound_%d" % seed
            r = ctx.run_cached(name, seed=seed, kill=bool(seed % 2), envelopes=env)
            r = replay(ctx, name, r, env)
            r["b1"] = b1_alarm(r, env)[0]
            with open(os.path.join(ctx.runs, name + ".json"), "w") as f:
                json.dump(_jsonable(r), f)
    ctx.log("done %s" % phases)


if __name__ == "__main__":
    main(sys.argv)
