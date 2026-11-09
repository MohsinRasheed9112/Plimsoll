"""Summarize Experiment 7 (Sec. V-F1, Table VI) from the campaign run files.

Usage: python experiments/exp7_summarize.py
Writes results/exp7/summary.json. Per (system, dataset): decisions and raw alarms per
oracle, the self-query hit rate at ef = 64 (unfiltered and filtered), persistent misses
at theta_max, the init check's recall@10, and candidates with their replays. Raw alarms
that have the signature of a known engine bug (Sec. V-C) are listed by rule and kept
apart; triage itself is not computed here.
"""
import glob
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EI_RULES = ("existence", "score", "order", "uniqueness", "size", "malformed", "filter", "lookup",
            "exact_size", "exact_rank")


def summarize(engine, tag):
    files = sorted(f for f in glob.glob(os.path.join(ROOT, "results", "campaigns", engine, tag, "run_*.json"))
                   if "_rep" not in os.path.basename(f))
    rs = [json.load(open(f)) for f in files]
    if not rs:
        return None
    dec = Counter()
    for r in rs:
        dec.update(r["decisions"])
    viol = Counter(v["rule"] for r in rs for v in r["violations"])
    ei_raw = sum(c for rule, c in viol.items() if rule in EI_RULES)
    # ClickHouse 26.9.13.15: an EI size alarm at or after a DELETE has the signature of #117286
    # (Sec. V-C4); such alarms are counted apart from false-alarm candidates
    def sig117286(r, v):
        return (engine == "clickhouse" and v["rule"] == "size"
                and any(h.get("op") == "delete" and h["step"] <= v["step"] for h in r["history"]))
    ei_known = sum(1 for r in rs for v in r["violations"] if v["rule"] in EI_RULES and sig117286(r, v))

    # Qdrant 1.19.2: a zero-rule persistent miss in a filtered self-query, at a step with no
    # payload index on the attribute, has the signature of the planner misroute (Sec. V-C1)
    def sig_misroute(r, v):
        if not engine.startswith("qdrant") or v["rule"] != "wq_persistent_zero" or not v["where"].endswith(":filtered"):
            return False
        last = None
        for h in r["history"]:
            if h.get("op") in ("attr_index", "attr_index_drop") and h["step"] <= v["step"]:
                last = h
        return last is None or (last["op"] == "attr_index_drop" and last.get("err") is None
                                and (last["action"] == "op" or last.get("outcome") == "ack"))
    wq_known = sum(v.get("count", 1) for r in rs for v in r["violations"] if sig_misroute(r, v))
    # a WQ or SE violation record covers several witnesses or answers: count them
    wq_raw = sum(v.get("count", 1) for r in rs for v in r["violations"]
                 if v["rule"] in ("wq_new_persistent", "wq_persistent_zero"))
    se_raw = sum(v.get("count", 1) for r in rs for v in r["violations"] if v["rule"] == "se_changed")
    hits = Counter()
    for r in rs:
        for c in r["checks"]:
            for mode, (h, n) in (c.get("wq_first_hits") or {}).items():
                hits[mode + "_h"] += h
                hits[mode + "_n"] += n
    rec = [c["recall_at_k"] for r in rs for c in r["checks"] if c.get("recall_at_k") is not None]
    rec_u = [c["recall_at_k_by_filter"]["none"] for r in rs for c in r["checks"]
             if (c.get("recall_at_k_by_filter") or {}).get("none") is not None]
    pers = sum(sum(c["wq_persistent"].values()) for r in rs for c in r["checks"])
    wit = sum(2 * len_w for r in rs for len_w in [r["params"]["CHECK_W"]] for _ in r["checks"])
    cands = [(r["seed"], tuple(c["key"]), c.get("replay"), c.get("confirmed")) for r in rs for c in r["candidates"]]
    ei_dec = dec["ei_answers"] + dec["lookups"]
    # WQ decisions: exact-mode comparisons (counted by the run) plus, where the zero rule
    # applies (Qdrant; Chroma's calibration rejected it), one per witness and mode per check
    zero = engine.startswith("qdrant")
    wq_dec = dec["wq_exact"] + (sum(2 * r["params"]["CHECK_W"] * len(r["checks"]) for r in rs) if zero else 0)
    out = dict(engine=engine, tag=tag, runs=len(rs), aborted=[r["seed"] for r in rs if r.get("aborted")],
               decisions=dict(dec), violations_by_rule=dict(viol),
               ei=dict(decisions=ei_dec, raw=ei_raw, rate=ei_raw / ei_dec if ei_dec else None,
                       known_117286=ei_known, other=ei_raw - ei_known,
                       other_rate=(ei_raw - ei_known) / ei_dec if ei_dec else None),
               wq=dict(decisions=wq_dec, raw=wq_raw, rate=wq_raw / wq_dec if wq_dec else None, zero_rule=zero,
                       known_F6=wq_known, other=wq_raw - wq_known,
                       other_rate=(wq_raw - wq_known) / wq_dec if wq_dec else None),
               se=dict(decisions=dec["se_exact"], raw=se_raw, rate=se_raw / dec["se_exact"] if dec["se_exact"] else None),
               first_hit_rate={m: hits[m + "_h"] / hits[m + "_n"] for m in ("plain", "filtered") if hits[m + "_n"]},
               persistent_miss_rate=pers / wit if wit else None,
               init_recall=dict(min=min(rec), max=max(rec), mean=sum(rec) / len(rec)) if rec else None,
               init_recall_unfiltered=dict(min=min(rec_u), max=max(rec_u), mean=sum(rec_u) / len(rec_u),
                                           runs=len(rec_u)) if rec_u else None,
               candidates=len(cands), confirmed=sum(1 for c in cands if c[3]),
               candidate_keys=[[c[0], list(c[1]), c[2]] for c in cands][:40])
    return out


def main():
    res = {}
    for engine in ("pgvector", "qdrant", "clickhouse", "chroma"):
        for tag in ("exp7_sift", "exp7_fmnist"):
            s = summarize(engine, tag)
            if s:
                res["%s/%s" % (engine, tag)] = s
    os.makedirs(os.path.join(ROOT, "results", "exp7"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "exp7", "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(k, "runs", v["runs"], "EI", v["ei"], "WQ", v["wq"], "SE", v["se"], "hits", v["first_hit_rate"],
              "recall", v["init_recall"], "cands", v["candidates"], v["confirmed"])


if __name__ == "__main__":
    main()
