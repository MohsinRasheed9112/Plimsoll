"""Print the Experiment 1 numbers (Table I, Sec. V-B) from results/exp1/summary.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
s = json.load(open(os.path.join(ROOT, "results", "exp1", "summary.json")))

print("complete:", s["complete"])
for k, v in s["pass"].items():
    print("PASS RULE", k, json.dumps(v))
c = s["calibration"]
for name in ("exact_calib", "hnsw_calib", "hnsw_killcal"):
    x = c.get(name)
    if not x:
        continue
    print("\n==", name, "seeds", len(x["seeds"]), "EI-violating runs", x["runs_with_ei_violations"])
    for op in sorted(k for k in x if k.startswith(("op", "k")) and isinstance(x[k], dict) and "se_D" in x[k]):
        o = x[op]
        print("  %s changed=%s newpersist=%s persist_after=%s D=%s tail=%s" % (
            op, (o["changed_answers"]["min"], o["changed_answers"]["max"]),
            (o["new_persistent"]["min"], o["new_persistent"]["max"]),
            (o["persistent_after"]["min"], o["persistent_after"]["max"]),
            (round(o["se_D"]["mean"], 5), round(o["se_D"]["sd"], 5), o["se_D"]["min"], o["se_D"]["max"]),
            (o["se_tail"]["min"], o["se_tail"]["max"])))
    h = x["self_query_hit_at_ef64"]
    print("  self-query hit@ef64 min/mean/max %.4f %.4f %.4f" % (h["min"], h["mean"], h["max"]))
    p = x["persistent_misses_per_2000"]
    print("  persistent misses per 2000 at op points min/mean/max", p["min"], round(p["mean"], 3), p["max"])
    r = x["recall_at_op_points"]
    print("  recall@10 at op points min/mean/max %.4f %.4f %.4f" % (r["min"], r["mean"], r["max"]))
    print("  kills", x["kills"], "secs/run", round(x["secs_per_run"]["mean"], 1))
print("\nindex-preserving exact", c.get("exact_index_preserving"), "hnsw", c.get("hnsw_index_preserving"),
      "killcal check", c.get("hnsw_killcal_index_preserving_check"))
for m, t in s["soundness"].items():
    print("\n== soundness", m, "runs", t["runs"], "kill runs", t["kill_runs"], "kills", t["kills"])
    print("  EI decisions %d raw %d" % (t["decisions_ei"], t["raw_ei"]))
    for o, d in t["per_oracle"].items():
        print("  %s exact dec %d raw %d | stat dec %d raw %d | rate %.6f stat-only %.4f" % (
            o, d["exact_dec"], d["exact_raw"], d["stat_dec"], d["stat_raw"], d["rate_exact_and_stat"],
            d["rate_statistical_only"]))
    print("  zero rule", t["zero_rule"], "tests", t["tests"], "p<=.05", t["tests_p_le_0.05"],
          "p<=.01", t["tests_p_le_0.01"])
    print("  candidates", t["candidates"], "confirmed", t["confirmed"], "counted", t["counted"],
          "confirmed+control-recur", t["confirmed_but_recurring_in_control"])
    for cd in t["candidates_detail"]:
        print("   cand", cd)
    print("  two-envelope secondary", t.get("two_envelope_secondary"))
    print("  B1 alarm runs", t["b1_alarm_runs"], "side runs", t["side_runs"])
print("\n== faults")
for f in s["faults"]:
    print("  %-13s seed %d affected %4d | cand %4d confirmed %4d counted %4d by-oracle %s | flagged items %d (in log %d) | B1 %s" % (
        f["kind"], f["seed"], f["affected_items"], f["candidates"], f["confirmed"], f["counted"],
        f["counted_by_oracle"], f["flagged_items"], f["flagged_items_in_fault_log"], f["b1_alarm"]))
print("fault controls", s["fault_controls"])
