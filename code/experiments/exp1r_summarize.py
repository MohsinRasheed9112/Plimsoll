"""Summarize Experiment 1 on a real engine (Sec. V-B2) from its run files.

Usage: python experiments/exp1r_summarize.py <engine>
Writes results/exp1r/<engine>/summary.json. Reuses the tallies of exp1_summarize.py.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp1_summarize as E  # noqa: E402
from exp1r import ENGINES  # noqa: E402


def main(engine):
    res = os.path.join(E.ROOT, "results", "exp1r", engine)
    E.RUNS = os.path.join(res, "runs")
    seeds = ENGINES[engine]["seeds"]
    out = {"engine": engine, "calibration": {}, "soundness": {}}
    for ph in ("calib", "killcal"):
        runs = E.main_runs(ph)
        if runs:
            out["calibration"][ph] = E.calibration(runs)
            out["calibration"][ph]["complete"] = sorted(r["seed"] for r in runs) == list(seeds[ph])
    env = json.load(open(os.path.join(res, "envelopes.json")))
    out["calibration"]["index_preserving"] = env["index_preserving"]
    out["calibration"]["killcal_index_preserving_check"] = env.get("killcal_index_preserving_check")
    runs = E.main_runs("sound")
    out["soundness"] = E.tally(runs, E.side_runs("sound"))
    out["soundness"]["complete"] = sorted(r["seed"] for r in runs) == list(seeds["sound"])
    t = out["soundness"]
    out["pass"] = {
        "1_confirmed_false_positives_zero": dict(confirmed=t["confirmed"], counted=t["counted"],
                                                pass_=t["confirmed"] == 0),
        "2_ei_raw": dict(soundness=t["raw_ei"], calibration_runs_with_ei=len(
            [s for ph in ("calib", "killcal") for s in out["calibration"].get(ph, {}).get("runs_with_ei_violations", [])])),
        "3_wq_se_raw_rate_at_most_1pct": dict(rates={o: t["per_oracle"][o]["rate_exact_and_stat"] for o in ("WQ", "SE")},
                                               pass_=all(t["per_oracle"][o]["rate_exact_and_stat"] <= 0.01
                                                         for o in ("WQ", "SE"))),
    }
    # self-query hit rate at the operating point (ef = 64) and recall there
    c = out["calibration"]["calib"]
    out["operating_point"] = dict(self_query_hit_at_ef64=c["self_query_hit_at_ef64"],
                                  recall_at_op_points=c["recall_at_op_points"])
    out["restart_preserving"] = dict(
        clean_restart_identical_in_all_calibration_runs=env["index_preserving"].get("restart"))
    with open(os.path.join(res, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("pass", "operating_point", "restart_preserving")}, indent=1))
    print("calibration complete:", {ph: out["calibration"][ph]["complete"] for ph in ("calib", "killcal")
                                    if ph in out["calibration"]}, "soundness complete:", t["complete"])


if __name__ == "__main__":
    main(sys.argv[1])
