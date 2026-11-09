"""Summarize Experiment 4 from the per-trial JSON files (no new measurement).

Usage: python experiments/exp4_summarize.py <cfg>
Writes results/exp4/<cfg>/summary.json and results/exp4/<cfg>/tables.md. Every
decision is recomputed here from stored statistics and the calibration envelopes.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from plimsoll import stats as S  # noqa: E402
from exp4 import CFGS, GRID, P, RES  # noqa: E402

LEVEL = P["LEVEL"]
FLOOR = 1.0 / (P["K"] * P["NQ"])

TARGETS = {  # design doc, Experiment 4 (percent; degradation in recall points)
    "desync": ("EI", 0.05, 0.1),
    "resurrect": ("EI", 0.2, 0.5),
    "lost": ("WQ", 0.2, 0.5),
    "lost_rebuild": ("WQ", 0.2, 0.5),
    "degrade": ("SE", 1.0, 2.0),
}
# EI is expected to detect a fault at least this many times smaller than the best baseline
MIN_RATIO = 5
METHODS = {   # EI_q: EI on the answers to the 1,000 queries only, the queries B1 uses
    # (an additional analysis; EI's own decision uses every answer)
    "desync": ["EI", "EI_q", "B1", "B2"],
    "resurrect": ["EI", "EI_q", "B1", "B2"],
    "lost": ["WQ", "SE_exact", "EI", "B1", "B2"],
    "lost_rebuild": ["WQ", "SE", "EI", "B1", "B2"],
    "degrade": ["SE", "WQ", "EI", "B1", "B2"],
}
REF = {"desync": "S0", "resurrect": "S0", "lost": "S1", "lost_rebuild": "S2", "degrade": "S2"}


def p_low(x, calib):
    return S.mean_pvalue(x, calib, lower=True, floor=FLOOR)


def p_high(x, calib):
    return S.mean_pvalue(x, calib, lower=False, floor=FLOOR)


def decide(fam, lv, env):
    ref = env[REF[fam]]
    d = {}
    d["EI"] = lv["ei"]["bad"] > 0
    d["EI_q"] = lv["ei"]["by_source"]["se"] > 0
    d["B1"] = p_low(lv["recall"], ref["recall"]) <= LEVEL
    d["B2"] = p_low(lv["b2_hits"] / 1e4, ref["b2"]) <= LEVEL
    if fam in ("lost", "lost_rebuild", "degrade"):
        d["WQ"] = bool(lv["wq"]["alarm"])
    if fam == "lost":
        d["SE_exact"] = lv["changed"] > 0
    if fam in ("lost_rebuild", "degrade"):
        d["SE"] = p_high(lv["D"], env["S2"]["D"]) <= LEVEL
    return d


def f_star(levels, rates, n):
    """Smallest level whose rate, and the rate of every larger level, is >= 95/100."""
    need = 0.95 * n
    best = None
    for lvl, r in reversed(list(zip(levels, rates))):
        if r >= need:
            best = lvl
        else:
            break
    return best


def main(cfg):
    base = os.path.join(RES, cfg)
    env = json.load(open(os.path.join(base, "envelopes.json")))
    files = sorted(glob.glob(os.path.join(base, "trial", "*.json")))
    trials = [json.load(open(f)) for f in files]
    seeds = sorted(t["seed"] for t in trials)
    n = len(trials)
    out = dict(cfg=cfg, dataset=CFGS[cfg]["name"], n_trials=n,
               complete=seeds == list(CFGS[cfg]["trials"]), c_star=env.get("c_star"),
               c_star_rule=env.get("c_star_rule"),
               calibration=dict(index_preserving_restart=env["index_preserving_restart"],
                                ei_bad=env["ei_bad_total"], ei_decisions=env["ei_decisions_total"]))
    for st in ("S0", "S1", "S2"):
        if st in env:
            e = env[st]
            out["calibration"][st] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                                              min=float(np.min(v)), max=float(np.max(v)))
                                      for k, v in e.items()}
    # ---------------------------------------------------- healthy states
    fa = {}
    hs = [t["healthy"] for t in trials]
    fa["EI_S0_S1_S2"] = dict(states=sum(len(h) for h in hs),
                             with_violation=sum(int(x["ei"]["bad"] > 0) for h in hs for x in h.values()),
                             decisions=sum(x["ei"]["decisions"] for h in hs for x in h.values()))
    fa["WQ_exact_S1"] = sum(int(h["S1"]["new_persistent"] > 0) for h in hs)
    fa["SE_exact_S1"] = sum(int(h["S1"]["changed"] > 0) for h in hs)
    for st in ("S0", "S1", "S2"):
        if st not in hs[0]:
            continue
        fa["B1_" + st] = sum(int(p_low(h[st]["recall"], env[st]["recall"]) <= LEVEL) for h in hs)
        fa["B2_" + st] = sum(int(p_low(h[st]["b2_hits"] / 1e4, env[st]["b2"]) <= LEVEL) for h in hs)
    if "S2" in hs[0]:
        fa["SE_S2"] = sum(int(p_high(h["S2"]["D"], env["S2"]["D"]) <= LEVEL) for h in hs)
        fa["WQ_count_S2"] = sum(int(h["S2"]["wq"]["persistent"] >= env["c_star"]) for h in hs)
    # EI on fault states that EI cannot see (lost entries, degradation)
    quiet = [lv for t in trials for fam in ("lost", "lost_rebuild", "degrade")
             for lv in t["faults"].get(fam, [])]
    fa["EI_on_non_EI_faults"] = dict(states=len(quiet), with_violation=sum(int(lv["ei"]["bad"] > 0) for lv in quiet),
                                     decisions=sum(lv["ei"]["decisions"] for lv in quiet))
    out["false_alarms"] = fa
    # ------------------------------------------------------ fault levels
    fams = {}
    for fam in ("desync", "resurrect", "lost", "lost_rebuild", "degrade"):
        if fam not in trials[0]["faults"]:
            continue
        keys = [lv.get("pct", lv.get("M")) for lv in trials[0]["faults"][fam]]
        rows = []
        for j, key in enumerate(keys):
            lvs = [t["faults"][fam][j] for t in trials]
            assert all(lv.get("pct", lv.get("M")) == key for lv in lvs)
            dec = [decide(fam, lv, env) for lv in lvs]
            ref = [t["healthy"][REF[fam]] for t in trials]
            drop = [r["recall"] - lv["recall"] for r, lv in zip(ref, lvs)]
            row = dict(level=key, count=lvs[0].get("count"),
                       detections={m: int(sum(d[m] for d in dec)) for m in METHODS[fam]},
                       recall_drop_points=dict(mean=100 * float(np.mean(drop)), sd=100 * float(np.std(drop, ddof=1)),
                                               min=100 * float(np.min(drop)), max=100 * float(np.max(drop))),
                       ei_flagged=int(sum(lv["ei"]["n_flagged"] for lv in lvs)),
                       ei_flagged_in_fault_set=int(sum(lv["ei"].get("n_flagged_in_fault_set", 0) for lv in lvs)))
            if fam in ("lost_rebuild", "degrade"):
                row["D_mean"] = float(np.mean([lv["D"] for lv in lvs]))
            for key2 in ("witnesses_in_fault_set", "negatives_in_fault_set"):
                if key2 in lvs[0]:
                    v = [lv[key2] for lv in lvs]
                    row[key2] = dict(mean=float(np.mean(v)), trials_with_any=int(sum(x > 0 for x in v)))
            if fam in ("lost", "lost_rebuild", "degrade"):
                row["escalations_mean"] = float(np.mean([lv["wq"]["escalated"] for lv in lvs]))
            rows.append(row)
        fs = {}
        for m in METHODS[fam]:
            rates = [r["detections"][m] for r in rows]
            if fam == "degrade":
                # order by effect: induced drop grows as M falls (the grid is listed by falling M)
                lv = f_star([r["recall_drop_points"]["mean"] for r in rows], rates, n)
                fs[m] = None if lv is None else dict(induced_drop_points=lv,
                                                     M=rows[[r["recall_drop_points"]["mean"] for r in rows].index(lv)]["level"])
            else:
                fs[m] = f_star([float(r["level"]) for r in rows], rates, n)
        fams[fam] = dict(levels=rows, f_star=fs)
    out["faults"] = fams
    # ------------------------- targets, and EI and WQ against the baselines
    verdict = {}
    for fam, (orc, strong, acc) in TARGETS.items():
        if fam not in fams:
            continue
        f = fams[fam]["f_star"][orc]
        val = None if f is None else (f["induced_drop_points"] if fam == "degrade" else f)
        verdict[fam] = dict(oracle=orc, f_star=val, strong=strong, acceptable=acc,
                            result=("none" if val is None else "strong" if val <= strong
                                    else "acceptable" if val <= acc else "weak"))
    out["targets"] = verdict
    vs_base = {}
    for fam in ("desync", "resurrect"):
        if fam not in fams:
            continue
        fs = fams[fam]["f_star"]
        bases = [fs[b] for b in ("B1", "B2")]
        best = min([b for b in bases if b is not None], default=None)
        top = float(GRID[fam][-1])
        vs_base[fam] = dict(EI=fs["EI"], best_baseline=best,
                       ratio=(best / fs["EI"]) if (best is not None and fs["EI"]) else None,
                       ratio_lower_bound=(top / fs["EI"]) if (best is None and fs["EI"]) else None)
        r = vs_base[fam]["ratio"] if vs_base[fam]["ratio"] is not None else vs_base[fam]["ratio_lower_bound"]
        vs_base[fam]["holds"] = bool(r is not None and r >= MIN_RATIO)
    if "lost" in fams:
        fs = fams["lost"]["f_star"]
        vs_base["lost"] = dict(WQ=fs["WQ"], B1=fs["B1"],
                          holds=bool(fs["WQ"] is not None and (fs["B1"] is None or fs["WQ"] <= fs["B1"])))
    out["vs_baselines"] = vs_base
    tt = [t["times"]["total"] for t in trials]
    out["secs_per_trial"] = dict(mean=float(np.mean(tt)), min=float(np.min(tt)), max=float(np.max(tt)))
    out["short_answers"] = int(sum(t["short_answers"] for t in trials))
    out["gt_fallbacks"] = int(sum(lv["gt_fallback"] for t in trials for fam in t["faults"].values() for lv in fam)
                              + sum(h["gt_fallback"] for t in trials for h in t["healthy"].values()))
    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    write_tables(cfg, out)
    print(json.dumps(dict(complete=out["complete"], targets=verdict, vs_baselines=vs_base, false_alarms=fa), indent=1))


def write_tables(cfg, out):
    L = ["# Experiment 4, configuration %s (%s): tables generated from summary.json" % (cfg, out["dataset"]), ""]
    for fam, d in out["faults"].items():
        ms = METHODS[fam]
        L.append("## %s" % fam)
        L.append("")
        L.append("| level | items | " + " | ".join(ms) + " | mean recall drop (points) |")
        L.append("|---|---|" + "---|" * len(ms) + "---|")
        for r in d["levels"]:
            L.append("| %s | %s | " % (r["level"], r["count"] if r["count"] is not None else "-")
                     + " | ".join(str(r["detections"][m]) for m in ms)
                     + " | %.3f |" % r["recall_drop_points"]["mean"])
        L.append("")
        L.append("f*: " + ", ".join("%s = %s" % (m, d["f_star"][m]) for m in ms))
        L.append("")
    with open(os.path.join(RES, cfg, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
