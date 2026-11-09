"""Summarize Experiment 1 from the per-run JSON files (no new measurement).

Writes results/exp1/summary.json. Every number is computed from the run files.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from plimsoll import stats as S  # noqa: E402

RES = os.path.join(ROOT, "results", "exp1")
RUNS = os.path.join(RES, "runs")
ALPHA = 0.05
OP_POINTS = ["op1.pre", "op1.post", "op2.pre", "op2.post", "op3.pre", "op3.post"]


def _load(f):
    return json.load(open(f))


def main_runs(prefix):
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, prefix + "_*.json"))):
        base = os.path.basename(f)
        if any(t in base for t in ("_rep", "_ctl", "_srep")):
            continue
        out.append(_load(f))
    return out


def side_runs(prefix):
    return [_load(f) for f in sorted(glob.glob(os.path.join(RUNS, prefix + "_*_*.json")))
            if any(t in os.path.basename(f) for t in ("_rep", "_ctl", "_srep"))]


def rng3(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                min=float(x.min()), max=float(x.max()))


def oracle_of(key):
    if key[0] in ("EI", "WQ", "SE"):
        return key[0]
    return "WQ" if key[2] == "wq_count" else "SE"


def calibration(runs):
    out = {"seeds": [r["seed"] for r in runs],
           "runs_with_ei_violations": [r["seed"] for r in runs if r["violations"]]}
    for name in sorted(runs[0]["ops"].keys()):
        recs = [r["ops"][name] for r in runs]
        out[name] = dict(changed_answers=rng3([len(x["se_changed"]) for x in recs]),
                         new_persistent=rng3([len(x["wq_new_persistent"]) for x in recs]),
                         persistent_after=rng3([x["wq_count"] for x in recs]),
                         se_D=rng3([x["se_D"] for x in recs]),
                         se_tail=rng3([x["se_tail"] for x in recs]))
    out["self_query_hit_at_ef64"] = rng3([r["points"][p]["wq_hit_theta"] for r in runs for p in OP_POINTS])
    out["persistent_misses_per_2000"] = rng3([r["points"][p]["wq_persistent"] for r in runs for p in OP_POINTS])
    out["recall_at_op_points"] = rng3([r["points"][p]["recall"] for r in runs for p in OP_POINTS])
    out["kills"] = kills(runs)
    out["secs_per_run"] = rng3([r["times"]["total"] for r in runs])
    return out


def kills(runs):
    ks = [k for r in runs for k in r["kills"]]
    return dict(n=len(ks), unknown_outcome=sum(k["outcome"] == "unknown" for k in ks),
                indeterminate_items=sum(k["n_indet"] for k in ks)) if ks else dict(n=0)


def tally(runs, sides):
    t = dict(runs=len(runs), kill_runs=sum(bool(r["kill"]) for r in runs), kills=kills(runs))
    for k in ("ei", "wq_exact", "se_exact", "zero_exact", "stat"):
        t["decisions_" + k] = sum(r["decisions"][k] for r in runs)
        t["raw_" + k] = sum(r["raw"][k] for r in runs)
    per = {"WQ": dict(exact_dec=t["decisions_wq_exact"], exact_raw=t["raw_wq_exact"], stat_dec=0,
                      stat_raw=0, zero_dec=0, zero_raw=0),
           "SE": dict(exact_dec=t["decisions_se_exact"], exact_raw=t["raw_se_exact"], stat_dec=0,
                      stat_raw=0, zero_dec=0, zero_raw=0)}
    for r in runs:
        rej = S.holm(r["tests"], ALPHA) if r["tests"] else {}
        for key in r["tests"]:
            o = "WQ" if key.endswith("wq_count") else "SE"
            per[o]["stat_dec"] += 1
            per[o]["stat_raw"] += int(key in rej)
        for c in r["candidates"]:
            if c["key"][0] == "ZERO":
                per[oracle_of(c["key"])]["zero_raw"] += 1
    # zero-rule decisions are counted per run in decisions['zero_exact'] without the
    # statistic's name; attribute them using the run's ops (wq_count or se_tail)
    for o in per:
        d = per[o]
        dec = d["exact_dec"] + d["stat_dec"]
        raw = d["exact_raw"] + d["stat_raw"]
        d["rate_exact_and_stat"] = raw / dec if dec else 0.0
        d["rate_statistical_only"] = d["stat_raw"] / d["stat_dec"] if d["stat_dec"] else 0.0
    t["per_oracle"] = per
    t["zero_rule"] = dict(decisions=t["decisions_zero_exact"], raw=t["raw_zero_exact"])
    t["ei_raw"] = t["raw_ei"]
    t["ei_rate"] = t["raw_ei"] / t["decisions_ei"] if t["decisions_ei"] else 0.0
    cands = [dict(run=r["seed"], **c) for r in runs for c in r["candidates"]]
    t["candidates"] = len(cands)
    t["unannotated_candidates"] = sum("confirmed" not in c for c in cands)
    t["confirmed"] = sum(bool(c.get("confirmed")) for c in cands)
    t["counted"] = sum(bool(c.get("counted")) for c in cands)
    t["confirmed_but_recurring_in_control"] = sum(bool(c.get("confirmed")) and bool(c.get("control_recur"))
                                                  for c in cands)
    t["candidates_detail"] = cands
    ps = [p for r in runs for p in r["tests"].values()]
    t["tests"] = len(ps)
    t["tests_p_le_0.05"] = int(sum(p <= 0.05 for p in ps))
    t["tests_p_le_0.01"] = int(sum(p <= 0.01 for p in ps))
    two = [(r, r.get("tests_two_envelope", {})) for r in runs]
    if any(tw for _, tw in two):
        rej_runs = 0
        for r, tw in two:
            fam = {k: v for k, v in r["tests"].items() if not k.startswith("k")}
            fam.update(tw)
            rej_runs += int(bool(S.holm(fam, ALPHA)))
        t["two_envelope_secondary"] = dict(
            crash_tests=sum(len(tw) for _, tw in two),
            crash_tests_p_le_0_05=int(sum(p <= 0.05 for _, tw in two for p in tw.values())),
            runs_with_a_holm_rejection=rej_runs)
    t["b1_alarm_runs"] = sum(bool(r.get("b1")) for r in runs)
    t["side_runs"] = dict(n=len(sides), ei_violations=sum(len(x["violations"]) for x in sides),
                          healthy_side_runs_with_ei_violations=[x["tag"] + "_" + str(x["seed"])
                                                                for x in sides if x["violations"]])
    t["secs_per_run"] = rng3([r["times"]["total"] for r in runs])
    return t


def fault_runs():
    out = []
    for r in main_runs("fault"):
        affected = set(i for f in r["fault_log"] for i in f["ids"])
        cands = r["candidates"]
        counted = [c for c in cands if c.get("counted")]
        confirmed = [c for c in cands if c.get("confirmed")]
        by_oracle_counted, by_oracle_confirmed = {}, {}
        for c in counted:
            by_oracle_counted[oracle_of(c["key"])] = by_oracle_counted.get(oracle_of(c["key"]), 0) + 1
        for c in confirmed:
            by_oracle_confirmed[oracle_of(c["key"])] = by_oracle_confirmed.get(oracle_of(c["key"]), 0) + 1
        items = set()
        for c in confirmed:
            k = c["key"]
            if k[0] == "EI" and k[1] in ("existence", "score", "lookup"):
                items.add(k[2])
            elif k[0] == "WQ":
                items.add(k[2])
        out.append(dict(kind=r["fault"]["kind"], frac=r["fault"]["frac"], seed=r["seed"],
                        affected_items=len(affected), candidates=len(cands),
                        unannotated=sum("confirmed" not in c for c in cands),
                        confirmed=len(confirmed), counted=len(counted),
                        confirmed_by_oracle=by_oracle_confirmed, counted_by_oracle=by_oracle_counted,
                        caught_confirmed=len(confirmed) > 0, caught_counted=len(counted) > 0,
                        flagged_items=len(items), flagged_items_in_fault_log=len(items & affected),
                        b1_alarm=bool(r.get("b1")), raw=r["raw"]))
    return out


def main():
    summ = {"calibration": {}, "soundness": {}, "faults": fault_runs()}
    for mode, phases in (("exact", ("calib",)), ("hnsw", ("calib", "killcal"))):
        for ph in phases:
            runs = main_runs("%s_%s" % (ph, mode))
            if runs:
                summ["calibration"]["%s_%s" % (mode, ph)] = calibration(runs)
        env_p = os.path.join(RES, "envelopes_%s.json" % mode)
        if os.path.exists(env_p):
            env = json.load(open(env_p))
            summ["calibration"]["%s_index_preserving" % mode] = env["index_preserving"]
            if "killcal_index_preserving_check" in env:
                summ["calibration"]["hnsw_killcal_index_preserving_check"] = env["killcal_index_preserving_check"]
        runs = main_runs("sound_%s" % mode)
        if runs:
            summ["soundness"][mode] = tally(runs, side_runs("sound_%s" % mode))
    ctl = [x for x in side_runs("fault") if "_ctl" in x["tag"] or x["tag"].startswith("ctl")]
    summ["fault_controls"] = dict(n=len(ctl), ei_violations=sum(len(x["violations"]) for x in ctl))
    # pass rules of Experiment 1
    sound = summ["soundness"]
    calib_ei = sum(len(c.get("runs_with_ei_violations", [])) for k, c in summ["calibration"].items()
                   if isinstance(c, dict) and "seeds" in c)
    confirmed_fp = sum(s["confirmed"] for s in sound.values())
    ei_raw = sum(s["ei_raw"] for s in sound.values()) + calib_ei
    rates = {m: {o: s["per_oracle"][o]["rate_exact_and_stat"] for o in ("WQ", "SE")} for m, s in sound.items()}
    kinds = {}
    for f in summ["faults"]:
        kinds.setdefault(f["kind"], []).append(f)
    complete = (len(kinds) == 3 and all(len(v) == 5 for v in kinds.values())
                and all(f["unannotated"] == 0 for v in kinds.values() for f in v)
                and all(s["unannotated_candidates"] == 0 for s in sound.values())
                and set(sound) == {"exact", "hnsw"} and all(s["runs"] == 30 for s in sound.values()))
    summ["complete"] = complete
    summ["pass"] = {
        "1_confirmed_false_positives_zero": dict(confirmed=confirmed_fp,
                                                 counted=sum(s["counted"] for s in sound.values()),
                                                 pass_=confirmed_fp == 0),
        "2_ei_no_raw_alarm": dict(value=ei_raw, pass_=ei_raw == 0),
        "3_wq_se_raw_rate_at_most_1pct": dict(rates=rates,
                                               pass_=all(v <= 0.01 for m in rates.values() for v in m.values())),
        "4_all_faults_caught_every_run": dict(
            counted={k: "%d/%d" % (sum(f["caught_counted"] for f in v), len(v)) for k, v in kinds.items()},
            confirmed={k: "%d/%d" % (sum(f["caught_confirmed"] for f in v), len(v)) for k, v in kinds.items()},
            pass_=complete and all(f["caught_counted"] for v in kinds.values() for f in v)),
    }
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({"complete": complete, "pass": summ["pass"]}, indent=1))


if __name__ == "__main__":
    main()
