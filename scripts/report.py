"""Collect every number that Sec. V reports from the result files.

Usage (from the repository root): python scripts/report.py
Writes results/report.json and prints one line per result group. Missing result files
are reported as None. Candidates whose run matches the signature of a bug of Sec. V-C
are classified here; others can be attributed by manual triage in the optional file
results/triage_attributions.json ({"<engine>/<tag>/<seed>": [{"keys": [...], "bug": name}]}).
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

# campaign batches per engine (Tables III and VI, Experiments 2, 3 and 7)
BATCHES = (("qdrant-1.18.3", ["exp2"]),
           ("clickhouse-26.3.45.3", ["exp2"]),
           ("pgvector", ["exp3", "exp3o", "exp3c", "exp7_sift", "exp7_fmnist"]),
           ("qdrant", ["exp3", "exp3h", "exp3o", "exp3c", "exp7_sift", "exp7_fmnist"]),
           ("clickhouse", ["exp3", "exp3o", "exp3c", "exp7_sift", "exp7_fmnist"]),
           ("chroma", ["exp3", "exp3o", "exp3c", "exp7_sift", "exp7_fmnist"]))


def J(*p):
    path = os.path.join(R, *p)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def campaign_runs(engine, tag):
    files = sorted(f for f in glob.glob(os.path.join(R, "campaigns", engine, tag, "run_*.json"))
                   if "_rep" not in os.path.basename(f))
    runs = []
    for path in files:
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def known_signature(engine, r, c):
    """Classify a candidate by the signature of a bug of Sec. V-C (same rules as the
    campaign driver): '117286' for an EI size candidate on ClickHouse 26.9.13.15 at or
    after a DELETE; 'lost_update' for EI candidates on ClickHouse from the first step at
    which an acknowledged concurrent lightweight UPDATE was not visible after a mutation;
    'misroute' for a WQ zero-rule candidate on Qdrant 1.19.2 with only filtered misses at
    a step without a payload index."""
    if engine == "clickhouse" and tuple(c["key"][:2]) == ("EI", "size"):
        if any(h.get("op") == "delete" and h["step"] <= c["key"][2] for h in r["history"]):
            return "117286"
    if engine == "clickhouse" and c["key"][0] == "EI":
        s0 = None
        for v in r["violations"]:
            w = v.get("where", "")
            if (v["rule"] == "lookup" and w.endswith(":concurrent_lookup")
                    and w.split(":")[1] in ("attr_index", "materialize", "optimize")
                    and (v.get("got") == "other vector" or str(v.get("expected", "")).startswith("attr"))):
                s0 = v["step"] if s0 is None else min(s0, v["step"])
        return "lost_update" if (s0 is not None and c["key"][2] >= s0) else None
    if engine == "qdrant" and tuple(c["key"][:2]) == ("WQ", "wq_persistent_zero"):
        s = c["key"][2]
        vs = [v for v in r["violations"] if v["rule"] == "wq_persistent_zero" and v["step"] == s]
        if not vs or not all(v["where"].endswith(":filtered") for v in vs):
            return None
        last = None
        for h in r["history"]:
            if h.get("op") in ("attr_index", "attr_index_drop") and h["step"] <= s:
                last = h
        # a killed index drop whose outcome was acknowledged also leaves no index
        ok = last is None or (last["op"] == "attr_index_drop" and last.get("err") is None
                              and (last["action"] == "op" or last.get("outcome") == "ack"))
        return "misroute" if ok else None
    return None


TRIAGE = J("triage_attributions.json") or {}


def triaged(engine, tag, r, c):
    for t in TRIAGE.get("%s/%s/%d" % (engine, tag, r["seed"]), []):
        if list(c["key"]) in [list(k) for k in t["keys"]]:
            return t["bug"]
    return None


def _log_secs(engine, tag, seed):
    """Duration of a run that ended on an engine error, from the driver's log line
    ("<engine> seed <s> main: <secs>s"); such runs have no total time in their file."""
    for lf in glob.glob(os.path.join(R, "campaigns", "%s_%s*.log" % (engine, tag))):
        with open(lf, errors="replace") as f:
            for line in f:
                m = re.search(r" seed %d main: (\d+)s," % seed, line)
                if m:
                    return float(m.group(1))
    return None


def campaign_facts(engine, tag):
    rs = campaign_runs(engine, tag)
    if not rs:
        return None
    out = dict(runs=len(rs), seeds=[r["seed"] for r in rs])
    durs = [r["times"].get("total") or _log_secs(engine, tag, r["seed"]) for r in rs]
    out["runs_without_time"] = [r["seed"] for r, d in zip(rs, durs) if d is None]
    out["secs"] = round(sum(d for d in durs if d), 1)
    out["hours"] = round(out["secs"] / 3600, 2)
    out["aborted"] = [r["seed"] for r in rs if r.get("aborted")]
    dec = Counter()
    for r in rs:
        dec.update(r["decisions"])
    out["decisions"] = dict(dec)
    act = Counter(h["action"] for r in rs for h in r["history"])
    out["kills"] = act["kill_write"] + act["kill_op"] + act["bg_kill"]
    out["maintenance_ops"] = act["op"] + act["kill_op"]
    out["writes"] = act["write"] + act["kill_write"] + act["bg_kill"]
    out["violations"] = sum(len(r["violations"]) for r in rs)
    out["violations_by_rule"] = dict(Counter(v["rule"] for r in rs for v in r["violations"]))
    cands = [(r["seed"], tuple(c["key"]), c.get("replay"), c.get("confirmed")) for r in rs for c in r["candidates"]]
    out["candidates"] = len(cands)
    out["candidates_confirmed"] = sum(1 for c in cands if c[3])
    out["candidates_replay_skipped"] = sum(1 for r in rs for c in r["candidates"] if c.get("replay_skipped"))
    cls = Counter()
    for r in rs:
        for c in r["candidates"]:
            k = known_signature(engine, r, c) or triaged(engine, tag, r, c)
            if k:
                cls["known_" + k] += 1
            elif c["key"][0] == "ERR":
                cls["engine_error"] += 1
                cls["engine_error_confirmed"] += int(bool(c.get("confirmed")))
            else:
                cls["other"] += 1
                cls["other_confirmed"] += int(bool(c.get("confirmed")))
    out["candidate_classes"] = dict(cls)
    out["runs_ended_by_error"] = sum(1 for r in rs if r.get("aborted"))
    out["confirmed_rules"] = dict(Counter("%s:%s" % (c[1][0], c[1][1]) for c in cands if c[3]))
    out["check_secs"] = round(sum(c.get("secs", 0) for r in rs for c in r["checks"]), 1)
    out["check_share"] = (sum(c.get("secs", 0) for r in rs if r["times"].get("total") for c in r["checks"])
                          / max(1e-9, sum(r["times"]["total"] for r in rs if r["times"].get("total"))))
    out["checks"] = sum(len(r["checks"]) for r in rs)
    out["runs_with_candidates"] = sorted({c[0] for c in cands})
    out["candidate_rules"] = dict(Counter("%s:%s" % (c[1][0], c[1][1]) for c in cands))
    # self-query hit rate at ef = 64 and the operating point of the first check
    hits = Counter()
    for r in rs:
        for c in r["checks"]:
            for mode, (h, n) in (c.get("wq_first_hits") or {}).items():
                hits[mode + "_h"] += h
                hits[mode + "_n"] += n
    if hits:
        out["first_hit_rate"] = {m: hits[m + "_h"] / hits[m + "_n"] for m in ("plain", "filtered") if hits[m + "_n"]}
        rates = [h / n for r in rs for c in r["checks"] for mode, (h, n) in (c.get("wq_first_hits") or {}).items()
                 if mode == "plain" and n]
        out["first_hit_rate_min_check"] = min(rates) if rates else None
    rec = [c["recall_at_k"] for r in rs for c in r["checks"] if c.get("recall_at_k") is not None]
    if rec:
        out["init_recall"] = dict(min=min(rec), max=max(rec), n=len(rec))
    out["persistent_misses"] = sum(sum(c["wq_persistent"].values()) for r in rs for c in r["checks"])
    out["witness_queries"] = sum(2 * r["params"]["CHECK_W"] for r in rs for c in r["checks"])
    out["compare_skipped"] = sum(len(r.get("compare_skipped", [])) for r in rs)
    ind = Counter()
    for r in rs:
        for x in r["ablation"]["indeterminate"]:
            ind.update({k: v for k, v in x.items() if k != "step"})
    out["indeterminate"] = dict(ind)
    out["concurrent_batches"] = sum(sum(x["batches"] for x in r.get("concurrent", [])) for r in rs)
    return out


def replay_outcomes():
    """Replay outcomes per candidate class over the campaigns on current releases and
    Experiment 7 (Sec. V-E4)."""
    out = defaultdict(Counter)
    for eng, tags in BATCHES:
        if "-" in eng:          # older releases (Experiment 2)
            continue
        for tag in tags:
            for r in campaign_runs(eng, tag):
                for c in r["candidates"]:
                    k = known_signature(eng, r, c) or triaged(eng, tag, r, c)
                    cls = k or ("ERR" if c["key"][0] == "ERR" else
                                ("qdrant_fmnist_other" if (eng == "qdrant" and tag == "exp7_fmnist") else "other"))
                    o = out["%s:%s" % (eng, cls)]
                    o["raw"] += 1
                    if c.get("replay") is None:
                        o["not_replayed"] += 1
                    else:
                        o["replayed"] += 1
                        o["confirmed"] += int(bool(c.get("confirmed")))
    return {k: dict(v) for k, v in sorted(out.items())}


def _part(rel, *keys):
    """A nested part of a result file, or None if the file is missing."""
    d = J(*rel)
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def main():
    F = {}
    F["exp1"] = J("exp1", "summary.json")
    for e in ("pgvector", "qdrant"):
        F["exp1r_" + e] = J("exp1r", e, "summary.json")
    F["exp4_A"] = J("exp4", "A", "summary.json")
    F["exp4_B"] = J("exp4", "B", "summary.json")
    F["repro_qdrant"] = {v: _part(("repro", "qdrant_lost_points", v + ".json"), "summary")
                         for v in ("1.18.3", "1.19.0", "1.19.2")}
    F["repro_clickhouse"] = {}
    for v in ("26.9.13.15", "26.3.45.3"):
        d = J("repro", "clickhouse_117286", v + ".json")
        F["repro_clickhouse"][v] = None if d is None else {
            ph: {k: x for k, x in d[ph].items() if not isinstance(x, list)}
            for ph in ("before_delete", "after_delete")}
    F["chroma_ef_probe"] = J("calib", "chroma", "ef_probe.json")
    cal = J("calib", "chroma", "calib.json")
    if cal:
        F["chroma_calib"] = dict(preserving=cal["preserving"], zero_rule=cal["zero_rule"],
                                 trials=[{k: t[k] for k in ("seed", "persistent", "restart_changed", "vacuum_changed")}
                                         for t in cal["trials"]])
    camps = {}
    for eng, tags in BATCHES:
        for t in tags:
            f = campaign_facts(eng, t)
            if f:
                camps["%s/%s" % (eng, t)] = f
    F["campaigns"] = camps
    F["replay"] = replay_outcomes()
    F["exp5"] = J("exp5", "summary.json")
    F["exp7"] = J("exp7", "summary.json")
    for part in ("contract", "indeterminacy", "replay", "calibration"):
        F["exp6_" + part] = J("exp6", part + ".json")
    for part in ("oracles", "ground_truth", "capacity", "ground_truth_scale"):
        F["exp8_" + part] = J("exp8", part + ".json")
    os.makedirs(R, exist_ok=True)
    with open(os.path.join(R, "report.json"), "w") as f:
        json.dump(F, f, indent=1, default=str)
    for k, v in F.items():
        print("==", k, json.dumps(v, default=str)[:1500] if v is not None else None)


if __name__ == "__main__":
    main()
