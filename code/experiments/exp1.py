"""Experiment 1 driver: test the tester on the reference engine (Sec. V-B1, Table I).

Phases: calibration (both engines), soundness, injected faults, replays, baseline B1,
summary. Every run is saved as JSON under results/exp1/runs/; the driver skips runs
whose file exists, so it can resume after an interruption.
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

from plimsoll import paths                           # noqa: E402
from plimsoll import stats as S                      # noqa: E402
from plimsoll.runner import OPS, P, run_once         # noqa: E402

RES = os.path.join(ROOT, "results", "exp1")
RUNS = os.path.join(RES, "runs")
WORK = paths.work_dir("exp1")
LOG = os.path.join(RES, "progress.log")

SEEDS = {
    ("exact", "calib"): range(2000, 2020), ("exact", "sound"): range(2100, 2130),
    ("hnsw", "calib"): range(1000, 1020), ("hnsw", "sound"): range(1100, 1130),
    ("hnsw", "killcal"): range(1500, 1520),   # calibration runs with kills
}
FAULTS = [("desync", 0.001, range(1200, 1205)), ("lost_entries", 0.01, range(1300, 1305)),
          ("resurrect", 0.005, range(1400, 1405))]
B1_POINTS = ["op1.pre", "op1.post", "op2.pre", "op2.post", "op3.pre", "op3.post"]


def log(msg):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run_cached(name, **kw):
    path = os.path.join(RUNS, name + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    t = time.time()
    res = run_once(workdir=WORK, data=DATA, log_path=os.path.join(RES, "server.log"), **kw)
    res = _jsonable(res)
    with open(path + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(path + ".tmp", path)
    log("run %s done in %.1fs: raw=%s candidates=%d" % (name, time.time() - t, res["raw"],
                                                        len(res["candidates"])))
    return res


# ------------------------------------------------------------------ envelopes
STEP = 1.0 / (P["K"] * 1000)   # resolution of D: one hit among K x |Q|


def _grid(x):
    """D on its exact grid, free of floating-point summation noise."""
    return round(x / STEP) / (P["K"] * 1000.0)


def _stats(recs):
    return {"wq_count": [x["wq_count"] for x in recs], "se_D": [_grid(x["se_D"]) for x in recs],
            "se_tail": [x["se_tail"] for x in recs]}


def build_envelopes(calib):
    env = {"index_preserving": {}, "stats": {}}
    for name, kind in OPS:
        recs = [r["ops"][name] for r in calib]
        env["index_preserving"][kind] = all(len(x["se_changed"]) == 0 and len(x["wq_new_persistent"]) == 0
                                            for x in recs)
        env["stats"][kind] = _stats(recs)
    env["b1"] = {p: [r["points"][p]["recall"] for r in calib] for p in B1_POINTS}
    return env


def calibrate(mode, phase="calib", kill=False):
    calib = []
    seeds = list(SEEDS[(mode, phase)])
    extra = 0
    for seed in seeds:
        r = run_cached("%s_%s_%d" % (phase, mode, seed), mode=mode, seed=seed, kill=kill)
        while r["violations"]:
            log("%s run %s/%d has EI violations: reported as candidates, replaced"
                % (phase, mode, r["seed"]))
            extra += 1
            s2 = seeds[-1] + extra
            r = run_cached("%s_%s_%d" % (phase, mode, s2), mode=mode, seed=s2, kill=kill)
        calib.append(r)
    return calib


# --------------------------------------------------------------------- replay
def _keys(res):
    return set(tuple(c["key"]) for c in res["candidates"])


def replay(prefix, res, env):
    """Replay a run's candidates: 2 of 3; controls without faults for exact mode."""
    cands = res["candidates"]
    if not cands:
        return res
    mode, seed = res["mode"], res["seed"]
    same = [c for c in cands if c["mode"] in ("ei", "exact")]
    stat = [c for c in cands if c["mode"] in ("stat", "stat_zero")]
    if same:
        reps = [run_cached("%s_rep%d" % (prefix, r), mode=mode, seed=seed, kill=res["kill"],
                           fault=res["fault"], build_seed=res["build_seed"] + 10_000 * (r + 1),
                           envelopes=env, tag="rep%d" % r) for r in range(3)]
        rk = [_keys(x) for x in reps]
        for c in same:
            c["replay"] = sum(tuple(c["key"]) in k for k in rk)
            c["confirmed"] = c["replay"] >= 2
            c["counted"] = c["confirmed"] if c["mode"] == "ei" else None
        exact_conf = [c for c in same if c["mode"] == "exact" and c["confirmed"]]
        if exact_conf:
            ctls = [run_cached("%s_ctl%d" % (prefix, r), mode=mode, seed=seed,
                               kill="draw_only" if res["kill"] else False,
                               fault=None, build_seed=res["build_seed"] + 10_000 * (r + 4),
                               envelopes=env, tag="ctl%d" % r) for r in range(3)]
            ck = [_keys(x) for x in ctls]
            for c in exact_conf:
                c["control_recur"] = sum(tuple(c["key"]) in k for k in ck)
                c["counted"] = c["control_recur"] == 0
        for c in same:
            if c["mode"] == "exact" and not c["confirmed"]:
                c["counted"] = False
    if stat:
        reps = [run_cached("%s_srep%d" % (prefix, r), mode=mode, seed=seed + 100_000 * (r + 1),
                           kill=res["kill"], fault=res["fault"], envelopes=env,
                           tag="srep%d" % r) for r in range(3)]
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
    ref = env["b1_kill"] if (res["kill"] and "b1_kill" in env) else env["b1"]
    ps = {p: S.mean_pvalue(res["points"][p]["recall"], ref[p], lower=True) for p in B1_POINTS}
    rej = S.holm(ps, P["ALPHA_FW"])
    return sorted(rej), ps


def environment():
    from importlib.metadata import version

    import scipy
    lines = ["date_utc: %s" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "platform: %s" % platform.platform(), "machine: %s" % platform.machine(),
             "python: %s" % sys.version.split()[0], "numpy: %s" % np.__version__,
             "scipy: %s" % scipy.__version__, "hnswlib: %s (pip)" % version("hnswlib"),
             "nproc: %s" % os.cpu_count()]
    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        lines += ["lscpu: " + line.strip() for line in cpu.splitlines() if "Model name" in line]
        mem = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()[1]
        lines.append("memory_mb: " + mem)
    except Exception:
        pass
    return "\n".join(lines) + "\n"


def main(engines):
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(RES, "environment.txt"), "w") as f:
        f.write(environment())
    log("start %s; constants %s" % (engines, json.dumps(P)))
    for mode in engines:
        calib = calibrate(mode)
        env = build_envelopes(calib)
        if mode == "hnsw":
            killcal = calibrate(mode, phase="killcal", kill=True)
            env["stats_crash"] = {k: _stats([r["ops"][k] for r in killcal])
                                  for k in ("k1", "k2", "k3")}
            env["b1_kill"] = {p: [r["points"][p]["recall"] for r in killcal] for p in B1_POINTS}
            env["killcal_index_preserving_check"] = {
                name: all(len(r["ops"][name]["se_changed"]) == 0 and
                          len(r["ops"][name]["wq_new_persistent"]) == 0 for r in killcal)
                for name, _ in OPS}
        with open(os.path.join(RES, "envelopes_%s.json" % mode), "w") as f:
            json.dump(_jsonable(env), f, indent=1)
        log("%s envelopes: index_preserving=%s" % (mode, env["index_preserving"]))
        for seed in SEEDS[(mode, "sound")]:
            name = "sound_%s_%d" % (mode, seed)
            r = run_cached(name, mode=mode, seed=seed, kill=bool(seed % 2), envelopes=env)
            r = replay(name, r, env)
            r["b1"] = b1_alarm(r, env)[0] if mode == "hnsw" else []
            with open(os.path.join(RUNS, name + ".json"), "w") as f:
                json.dump(_jsonable(r), f)
        if mode == "hnsw":
            for kind, frac, seeds in FAULTS:
                for seed in seeds:
                    name = "fault_%s_%d" % (kind, seed)
                    r = run_cached(name, mode="hnsw", seed=seed, fault=dict(kind=kind, frac=frac),
                                   envelopes=env)
                    r = replay(name, r, env)
                    r["b1"] = b1_alarm(r, env)[0]
                    with open(os.path.join(RUNS, name + ".json"), "w") as f:
                        json.dump(_jsonable(r), f)
        log("done %s" % mode)


if __name__ == "__main__":
    DATA = (np.load(os.path.join(ROOT, "data", "sift100k_base.npy")),
            np.load(os.path.join(ROOT, "data", "sift100k_queries.npy")))
    main(sys.argv[1:] or ["exact", "hnsw"])
