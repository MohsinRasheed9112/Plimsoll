"""Experiment 6 (ablations) from stored run files; no new measurement.

Parts (each writes results/exp6/<part>.json):
  contract      campaigns: answers that a naive contract would flag (scores taken as exact
                float64 distances, strict order in every mode, min(k, eligible) results
                promised), against EI's raw alarms on the same answers
  indeterminacy campaigns: kills that left a write's outcome unknown, and how lookups after
                recovery resolved its items. A model without indeterminacy must assume one
                outcome: "not applied" (only acknowledged writes count) raises a false alarm
                for every item that was applied; "applied" for every item that was not.
  replay        raw candidates against replay-confirmed ones (campaigns, Experiment 1 and
                its real-engine runs)
  calibration   healthy statistics judged without calibration: every operation treated
                as index-preserving (exact modes everywhere) and every persistent miss at
                theta_max treated as an alarm (zero rule everywhere)
Usage: python experiments/exp6_ablations.py [contract] [indeterminacy] [replay] [calibration]
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CAMP = os.path.join(ROOT, "results", "campaigns")
OUT = os.path.join(ROOT, "results", "exp6")

# campaign batches used for each engine (current releases)
# (Qdrant exp3: default full_scan_threshold; exp3h: HNSW path)
BATCHES = {
    "pgvector": ["pgvector/exp3", "pgvector/exp3o", "pgvector/exp3c", "pgvector/exp7_sift", "pgvector/exp7_fmnist"],
    "qdrant": ["qdrant/exp3", "qdrant/exp3h", "qdrant/exp3o", "qdrant/exp3c", "qdrant/exp7_sift", "qdrant/exp7_fmnist"],
    "clickhouse": ["clickhouse/exp3", "clickhouse/exp3o", "clickhouse/exp3c", "clickhouse/exp7_sift",
                   "clickhouse/exp7_fmnist"],
    "chroma": ["chroma/exp3", "chroma/exp3o", "chroma/exp3c", "chroma/exp7_sift", "chroma/exp7_fmnist"],
    "qdrant-1.18.3": ["qdrant-1.18.3/exp2"],
}


def runs(batch):
    files = sorted(f for f in glob.glob(os.path.join(CAMP, batch, "run_*.json"))
                   if "_rep" not in os.path.basename(f))
    return [json.load(open(f)) for f in files]


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print(json.dumps(obj, indent=1)[:4000])


def contract():
    out = {}
    for eng, batches in BATCHES.items():
        nc = collections.Counter()
        ei = collections.Counter()
        n_runs = 0
        for b in batches:
            for r in runs(b):
                n_runs += 1
                nc.update(r["ablation"]["naive_contract"])
                for v in r["violations"]:
                    if v["rule"] in ("existence", "score", "order", "uniqueness", "size", "malformed", "filter"):
                        ei[v["rule"]] += 1
        if not n_runs:
            continue
        a = nc["answers"]
        out[eng] = dict(runs=n_runs, answers=a, naive=dict(nc),
                        naive_share=dict(any=nc["any"] / a, score=nc["score"] / a, order=nc["order"] / a,
                                         size=nc["size"] / a) if a else None,
                        ei_search_violations=dict(ei))
    save("contract", out)


def indeterminacy():
    out = {}
    for eng, batches in BATCHES.items():
        ev = []
        crash_runs = 0
        for b in batches:
            for r in runs(b):
                x = r["ablation"]["indeterminate"]
                if any(h["action"] in ("kill_write", "kill_op", "bg_kill") for h in r["history"]):
                    crash_runs += 1
                for e in x:
                    ev.append(dict(seed=r["seed"], batch=b, **e))
        if not ev and not crash_runs:
            continue
        tot = collections.Counter()
        for e in ev:
            tot.update({k: e[k] for k in ("applied", "not_applied", "other")})
        runs_naive_na = len({(e["batch"], e["seed"]) for e in ev if e["applied"] > 0})
        runs_naive_ap = len({(e["batch"], e["seed"]) for e in ev if e["not_applied"] > 0})
        out[eng] = dict(
            crash_runs=crash_runs, resolution_events=len(ev), items=dict(tot),
            events_with_applied=sum(e["applied"] > 0 for e in ev),
            events_with_not_applied=sum(e["not_applied"] > 0 for e in ev),
            events_with_other=sum(e["other"] > 0 for e in ev),
            # a model without indeterminacy: false alarms per event and per crash run
            naive_not_applied=dict(events_with_false_alarm=sum(e["applied"] > 0 for e in ev),
                                   items_misjudged=tot["applied"], crash_runs_with_false_alarm=runs_naive_na),
            naive_applied=dict(events_with_false_alarm=sum(e["not_applied"] > 0 for e in ev),
                               items_misjudged=tot["not_applied"], crash_runs_with_false_alarm=runs_naive_ap))
    save("indeterminacy", out)


def replay():
    out = {}
    for eng, batches in BATCHES.items():
        c = collections.Counter()
        for b in batches:
            for r in runs(b):
                for k in r["candidates"]:
                    c["raw"] += 1
                    c["confirmed"] += int(bool(k.get("confirmed")))
                    c["raw_" + k["key"][0]] += 1
                    c["confirmed_" + k["key"][0]] += int(bool(k.get("confirmed")))
                    c["replay_skipped"] += int(k.get("replay_skipped") is not None)  # v16
                c["runs"] += 1
                c["runs_with_candidate"] += int(bool(r["candidates"]))
        if c["runs"]:
            out[eng] = dict(c)
    save("replay", out)


def calibration():
    """Exact rules applied to healthy statistics that calibration judges statistically."""
    out = {}
    # reference engine (Experiment 1, hnsw), pgvector and Qdrant (Experiment 1 on real
    # engines): calibration and soundness runs, every operation and kill point
    srcs = {"reference_hnsw": glob.glob(os.path.join(ROOT, "results", "exp1", "runs", "*_hnsw_*.json")),
            "pgvector": glob.glob(os.path.join(ROOT, "results", "exp1r", "pgvector", "runs", "*.json")),
            "qdrant": glob.glob(os.path.join(ROOT, "results", "exp1r", "qdrant", "runs", "*.json"))}
    for name, files in srcs.items():
        rs = [json.load(open(f)) for f in sorted(files)]
        rs = [r for r in rs if not r.get("fault") and r.get("tag", "main") in ("main", None, "")]
        if not rs:
            continue
        per = collections.defaultdict(lambda: collections.Counter())
        for r in rs:
            for op, rec in r["ops"].items():
                kind = rec.get("kind", op)
                key = "crash" if kind == "crash" else kind
                per[key]["decisions"] += 1
                per[key]["se_changed_any"] += int(len(rec["se_changed"]) > 0)
                per[key]["wq_persistent_any"] += int(rec["wq_count"] > 0)
                per[key]["wq_new_persistent_any"] += int(len(rec["wq_new_persistent"]) > 0)
        out[name] = dict(runs=len(rs), by_kind={k: dict(v) for k, v in per.items()})
    # Experiment 4 configuration A: healthy restart (S1) and rebuild (S2) of every trial
    tr = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results", "exp4", "A", "trial", "*.json")))]
    if tr:
        out["exp4_A"] = dict(
            trials=len(tr),
            S1_changed_any=sum(int(t["healthy"]["S1"]["changed"] > 0) for t in tr),
            S2_changed_any=sum(int(t["healthy"]["S2"]["changed"] > 0) for t in tr),
            S2_changed_mean=sum(t["healthy"]["S2"]["changed"] for t in tr) / len(tr),
            S2_persistent_any=sum(int(t["healthy"]["S2"]["wq"]["persistent"] > 0) for t in tr))
    save("calibration", out)


if __name__ == "__main__":
    args = sys.argv[1:] or ["contract", "indeterminacy", "replay", "calibration"]
    for a in args:
        globals()[a]()
