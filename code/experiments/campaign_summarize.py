"""Summarize campaign runs (Experiments 2, 3 and 6) from their JSON files.

Usage: python experiments/campaign_summarize.py <engine> <tag>
Prints and writes results/campaigns/<engine>/<tag>/summary.json:
  - runs, steps, kills by kind, checks, decisions per oracle
  - violations, candidates, replay-confirmed candidates (by key)
  - ablation counts: answers a naive contract would flag (scores taken as exact, strict
    order in every mode, min(k, eligible) results promised), and indeterminate writes by
    their observed outcome (a model without indeterminacy would flag one of the two groups)
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main(engine, tag):
    d = os.path.join(ROOT, "results", "campaigns", engine, tag)
    files = sorted(f for f in glob.glob(os.path.join(d, "run_*.json")) if "_rep" not in os.path.basename(f))
    runs = [json.load(open(f)) for f in files]
    S = dict(engine=engine, tag=tag, runs=len(runs), seeds=[r["seed"] for r in runs])
    act = collections.Counter(h["action"] for r in runs for h in r["history"])
    ops = collections.Counter("%s:%s" % (h["action"], h["op"]) for r in runs for h in r["history"])
    S["steps"] = sum(len(r["history"]) for r in runs)
    S["actions"] = dict(act)
    S["actions_by_op"] = dict(ops)
    S["kills"] = dict(total=act["kill_write"] + act["kill_op"] + act["bg_kill"],
                      in_flight_unknown=sum(1 for r in runs for h in r["history"]
                                            if h["action"] in ("kill_write", "kill_op") and h.get("outcome") == "unknown"))
    S["checks"] = sum(len(r["checks"]) for r in runs)
    dec = collections.Counter()
    for r in runs:
        dec.update(r["decisions"])
    S["decisions"] = dict(dec)
    S["violations"] = sum(len(r["violations"]) for r in runs)
    S["violations_by_rule"] = dict(collections.Counter(v["rule"] for r in runs for v in r["violations"]))
    cands = [dict(seed=r["seed"], **c) for r in runs for c in r["candidates"]]
    S["candidates"] = len(cands)
    S["candidates_confirmed"] = sum(1 for c in cands if c.get("confirmed"))
    S["runs_with_confirmed"] = sorted(set(c["seed"] for c in cands if c.get("confirmed")))
    S["candidate_keys"] = [(c["seed"], c["key"], c.get("replay")) for c in cands][:50]
    S["aborted_runs"] = [(r["seed"], r["aborted"]["step"], r["aborted"]["error"][:160]) for r in runs if r.get("aborted")]
    nc = collections.Counter()
    ind = collections.Counter()
    for r in runs:
        nc.update(r.get("ablation", {}).get("naive_contract", {}))
        for x in r.get("ablation", {}).get("indeterminate", []):
            ind.update({k: v for k, v in x.items() if k != "step"})
    S["ablation_naive_contract"] = dict(nc)
    if nc.get("answers"):
        S["ablation_naive_contract"]["share_flagged"] = nc["any"] / nc["answers"]
    S["ablation_indeterminate"] = dict(ind)
    S["wq_persistent_counts"] = sum(sum(c["wq_persistent"].values()) for r in runs for c in r["checks"])
    S["secs"] = dict(total=sum(r["times"].get("total", 0) for r in runs))
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
