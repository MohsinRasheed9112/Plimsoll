"""One Plimsoll run of Experiment 1 (paper Algorithm 1 and Sec. V-B).

The run records every operation and its outcome, keeps the logical model, measures
before and after each maintenance operation and each crash, and applies the oracles.
With calibration envelopes it also decides exact-mode and statistical candidates.
The engine's fault log is copied into the result for evaluation only; no oracle reads it.
"""
import json
import os
import shutil
import threading
import time

import numpy as np

from . import oracles as O
from . import stats as S
from .contracts import CONTRACTS
from .model import Model
from .refengine.client import EngineError, EngineUnavailable, RefEngine

P = dict(N_INITIAL=90_000, BATCH_INITIAL=1_000, SEG_UPDATES=3_000, SEG_DELETES=1_000,
         SEG_INSERTS=1_000, BATCH=100, K=10, EF=64, N_WITNESS=2_000, N_NEG_MAX=2_000,
         N_LOOKUP_LIVE=500, N_LOOKUP_DEL=500, KILL_DELAY_MAX=0.020, NOISE_SD=1.0,
         TAIL_DROP=0.5, ALPHA_FW=0.05, N_BOOT=2_000, M=16, EFC=100, THREADS=2,
         CAP=100_000, DIM=128)

OPS = [("op1", "flush"), ("op2", "restart"), ("op3", "rebuild")]
STATS = ("wq_count", "se_D", "se_tail")


def _rebuild_seed(build_seed):
    return build_seed + 1_000_003


class Run:
    def __init__(self, mode, seed, data, workdir, kill=False, fault=None, build_seed=None,
                 envelopes=None, log_path=None, tag="main", engine_factory=None, params=None,
                 contract=None):
        self.p = dict(P)
        self.p.update(params or {})
        self.mode = mode
        self.seed = int(seed)
        self.base, self.queries = data
        self.kill = kill
        self.fault = fault
        self.build_seed = int(build_seed if build_seed is not None else seed)
        self.env = envelopes
        self.contract = contract if contract is not None else CONTRACTS[mode]
        self.rng = np.random.default_rng(self.seed)                 # workload stream
        self.wrng = np.random.default_rng([self.seed, 1])           # witness sampling
        self.trng = np.random.default_rng([self.seed, 2])           # bootstrap
        self.model = Model(self.p["CAP"], self.p["DIM"])
        self.dir = os.path.join(workdir, "%s_%d_%s" % (mode, self.seed, tag))
        shutil.rmtree(self.dir, ignore_errors=True)
        if engine_factory is None:
            self.eng = RefEngine(self.dir, mode, threads=self.p["THREADS"], M=self.p["M"],
                                 ef_construction=self.p["EFC"], cap=self.p["CAP"], dim=self.p["DIM"],
                                 log_path=log_path)
        else:
            self.eng = engine_factory(self.dir, self.seed)
        self.res = dict(mode=mode, seed=self.seed, kill=kill, fault=fault,
                        build_seed=self.build_seed, tag=tag, ops={}, points={},
                        violations=[], kills=[], decisions=dict(ei=0, wq_exact=0, se_exact=0,
                                                                zero_exact=0, stat=0),
                        raw=dict(ei=0, wq_exact=0, se_exact=0, zero_exact=0, stat=0),
                        candidates=[], tests={}, times={})

    # ------------------------------------------------------------- workload
    def _write(self, op, ids, vecs=None):
        self.eng.write(op, ids, vecs)
        self.model.ack(op, ids, vecs)

    def _make_batch(self, kind, pool):
        live = self.model.definite_live_ids()
        B = self.p["BATCH"]
        if kind == "u":
            ids = np.sort(self.rng.choice(live, size=B, replace=False))
            src = self.rng.integers(0, len(self.base), size=B)
            vecs = (self.base[src] + self.rng.normal(0.0, self.p["NOISE_SD"], size=(B, self.p["DIM"]))
                    ).astype(np.float32)
            return "update", ids, vecs
        if kind == "d":
            return "delete", np.sort(self.rng.choice(live, size=B, replace=False)), None
        ids = np.array([pool.pop() for _ in range(B)], dtype=np.int64)
        return "insert", ids, self.base[ids].copy()

    # ----------------------------------------------------------- measuring
    def _search(self, Q, ef, where):
        ids, ds = self.eng.search(Q, self.p["K"], ef)
        v, nbad = O.ei_check_search(self.model, self.contract, Q, ids, ds, self.p["K"], where)
        self.res["violations"] += v
        self.res["decisions"]["ei"] += len(ids)
        self.res["raw"]["ei"] += nbad
        return ids, ds

    def _measure(self, point, witnesses=None):
        if hasattr(self.eng, "quiesce"):
            self.eng.quiesce()                  # wait until background index work is done
        t0 = time.time()
        m = self.model
        out = {}
        Q = self.queries
        out["se"] = self._search(Q, self.p["EF"], point + ":se")
        gt = O.ground_truth(m, self.contract, Q, self.p["K"], rng=self.wrng)
        out["gt_checked"] = gt["checked"]
        out["recall"] = O.recall_per_query(m, self.contract, Q, out["se"][0], gt, self.p["K"])
        out["hits"] = int(np.rint(out["recall"] * self.p["K"]).sum())  # integer: D is exact
        # witnesses
        if witnesses is None:
            live = m.definite_live_ids()
            witnesses = np.sort(self.wrng.choice(live, size=min(self.p["N_WITNESS"], len(live)),
                                                 replace=False))
        Vw = m.vec[witnesses]
        ids, _ = self._search(Vw, self.p["EF"], point + ":wq")
        first = np.array([r[0] if r else -1 for r in ids])
        hit = O.wq_hits(m, self.contract, witnesses, first)
        persistent = ~hit
        ef_max = self.contract["theta_max_ef"]
        miss = np.flatnonzero(~hit)
        if ef_max is not None and len(miss):
            ids2, _ = self._search(Vw[miss], ef_max, point + ":wq_esc")
            first2 = np.array([r[0] if r else -1 for r in ids2])
            hit2 = O.wq_hits(m, self.contract, witnesses[miss], first2)
            persistent[miss] = ~hit2
        out["witnesses"] = witnesses
        out["hit_theta"] = hit
        out["persistent"] = persistent
        # negative witnesses
        dead = m.deleted_ids()
        if len(dead):
            neg = np.sort(self.wrng.choice(dead, size=min(self.p["N_NEG_MAX"], len(dead)), replace=False))
            self._search(m.vec[neg], self.p["EF"], point + ":neg")
            out["n_neg"] = len(neg)
        else:
            out["n_neg"] = 0
        # lookups
        live = m.definite_live_ids()
        lk = list(self.wrng.choice(live, size=min(self.p["N_LOOKUP_LIVE"], len(live)), replace=False))
        if len(dead):
            lk += list(self.wrng.choice(dead, size=min(self.p["N_LOOKUP_DEL"], len(dead)), replace=False))
        lv, vv = self.eng.lookup(lk)
        v = O.ei_check_lookup(m, lk, lv, vv, point + ":lookup")
        self.res["violations"] += v
        self.res["decisions"]["ei"] += len(lk)
        self.res["raw"]["ei"] += len(v)
        self.res["points"][point] = dict(recall=float(out["recall"].mean()),
                                         wq_hit_theta=float(hit.mean()),
                                         wq_persistent=int(persistent.sum()),
                                         n_witness=int(len(witnesses)), n_neg=out["n_neg"],
                                         gt_checked=gt["checked"], secs=round(time.time() - t0, 2))
        return out

    # --------------------------------------------------------- comparisons
    def _compare(self, name, kind, before, after, same_witness):
        Q = self.queries
        rec = {}
        # restrict witness comparison to witnesses unchanged across the operation
        wb, wa = before["witnesses"], after["witnesses"]
        assert np.array_equal(wb, wa)
        keep = same_witness
        pb, pa = before["persistent"][keep], after["persistent"][keep]
        rec["wq_n"] = int(keep.sum())
        rec["wq_count"] = int(pa.sum())
        rec["wq_new_persistent"] = [int(x) for x in wb[keep][(~pb) & pa]]
        changed = O.answers_identical(self.model, self.contract, Q, before["se"], after["se"])
        rec["se_changed"] = [int(x) for x in np.flatnonzero(changed)]
        drop = before["recall"] - after["recall"]
        # D from integer hit counts: exactly 0 when no hit changes
        rec["se_D"] = (before["hits"] - after["hits"]) / float(self.p["K"] * len(Q))
        rec["se_tail"] = int(np.count_nonzero(drop > self.p["TAIL_DROP"]))
        rec["recall_before"] = float(before["recall"].mean())
        rec["recall_after"] = float(after["recall"].mean())
        rec["kind"] = kind
        self.res["ops"][name] = rec
        if self.env is None:
            return
        env = self.env
        exact = (kind != "crash") and env["index_preserving"][kind]
        if exact:
            self.res["decisions"]["wq_exact"] += rec["wq_n"]
            self.res["decisions"]["se_exact"] += len(Q)
            self.res["raw"]["wq_exact"] += len(rec["wq_new_persistent"])
            self.res["raw"]["se_exact"] += len(rec["se_changed"])
            for w in rec["wq_new_persistent"]:
                self.res["candidates"].append(dict(key=["WQ", name, w], mode="exact"))
            for qi in rec["se_changed"]:
                self.res["candidates"].append(dict(key=["SE", name, qi], mode="exact"))
            return
        two = [env["stats"]["restart"], env["stats"]["rebuild"]]
        if kind != "crash":
            primary = [env["stats"][kind]]
        elif "stats_crash" in env:
            # crash envelopes from calibration runs with kills
            primary = [env["stats_crash"][name]]
        else:
            primary = two
        for stat in STATS:
            obs = rec[stat]
            calib = [e[stat] for e in primary]
            if kind == "crash" and "stats_crash" in env:
                # secondary analysis: judge the crash against both the restart and the rebuild envelope
                self.res.setdefault("tests_two_envelope", {})["%s|%s" % (name, stat)] = \
                    self._pvalue(stat, obs, [e[stat] for e in two], rec, len(Q))
            if stat in ("wq_count", "se_tail") and all(max(c) == 0 for c in calib):
                self.res["decisions"]["zero_exact"] += 1
                if obs > 0:
                    self.res["raw"]["zero_exact"] += 1
                    # across a crash a control without the kill has no comparison, so the
                    # candidate is replayed with fresh seeds
                    mode = "stat_zero" if kind == "crash" else "exact"
                    self.res["candidates"].append(dict(key=["ZERO", name, stat], mode=mode))
                continue
            self.res["tests"]["%s|%s" % (name, stat)] = self._pvalue(stat, obs, calib, rec, len(Q))
            self.res["decisions"]["stat"] += 1

    def _pvalue(self, stat, obs, calib, rec, nq):
        """Largest p-value over the given envelopes."""
        ps = []
        for c in calib:
            if stat == "se_D":
                ps.append(S.mean_pvalue(obs, c, floor=1.0 / (self.p["K"] * nq)))
            else:
                n = rec["wq_n"] if stat == "wq_count" else nq
                if max(c) == 0:
                    ps.append(0.0 if obs > 0 else 1.0)
                else:
                    ps.append(S.count_pvalue(obs, c, n, self.p["N_BOOT"], self.trng)[0])
        return float(max(ps))

    # ----------------------------------------------------------------- run
    def _segment_op(self, s):
        name, kind = OPS[s]
        before = self._measure(name + ".pre")
        t0 = time.time()
        if kind == "flush":
            self.eng.flush()
        elif kind == "restart":
            lf = []
            if self.fault and self.fault["kind"] in ("lost_entries", "resurrect"):
                lf = [dict(kind=self.fault["kind"], frac=self.fault["frac"], seed=self.seed)]
            self.eng.clean_restart(load_faults=lf)
        else:
            self.eng.rebuild(_rebuild_seed(self.build_seed))
        self.res["times"][name] = round(time.time() - t0, 2)
        after = self._measure(name + ".post", witnesses=before["witnesses"])
        keep = np.ones(len(before["witnesses"]), dtype=bool)
        self._compare(name, kind, before, after, keep)

    def _crash(self, s, op, ids, vecs, delay):
        name = "k%d" % (s + 1)
        before = self._measure(name + ".pre")
        timer = threading.Timer(delay, self.eng.signal_kill)
        timer.start()
        try:
            self.eng.write(op, ids, vecs)
            self.model.ack(op, ids, vecs)
            outcome = "ack"
        except (EngineUnavailable, EngineError):
            self.model.unknown(op, ids, vecs)
            outcome = "unknown"
        timer.join()
        self.eng.kill()
        self.eng.recover()
        indet = sorted(self.model.indet.keys())
        viol = []
        if indet:
            lv, vv = self.eng.lookup(indet)
            self.res["decisions"]["ei"] += len(indet)
            for i, l, v in zip(indet, lv, vv):
                if not self.model.resolve(i, v if l else None):
                    viol.append(dict(rule="lookup", where=name + ":resolve", id=int(i)))
        self.res["violations"] += viol
        self.res["raw"]["ei"] += len(viol)
        self.res["kills"].append(dict(op=name, batch_op=op, outcome=outcome, n_indet=len(indet)))
        # witnesses whose item the in-flight batch touched may have changed: exclude them
        touched = set(int(i) for i in ids)
        keep = np.array([int(w) not in touched for w in before["witnesses"]], dtype=bool)
        after = self._measure(name + ".post", witnesses=before["witnesses"])
        self._compare(name, "crash", before, after, keep)

    def _no_crash(self, s, op, ids, vecs):
        """Control for a kill run: same draws and measurements, the batch without a kill."""
        name = "k%d" % (s + 1)
        before = self._measure(name + ".pre")
        self._write(op, ids, vecs)
        self._measure(name + ".post", witnesses=before["witnesses"])

    def execute(self):
        try:
            return self._execute()
        finally:
            try:
                self.eng.kill()
            except Exception:
                pass

    def _execute(self):
        t_start = time.time()
        self.eng.start(build_seed=self.build_seed)
        perm = self.rng.permutation(len(self.base))
        initial = perm[:self.p["N_INITIAL"]]
        pool = list(perm[self.p["N_INITIAL"]:][::-1])
        t0 = time.time()
        for b in range(0, len(initial), self.p["BATCH_INITIAL"]):
            ids = np.sort(initial[b:b + self.p["BATCH_INITIAL"]])
            self._write("insert", ids, self.base[ids])
        if hasattr(self.eng, "after_initial_load"):
            self.eng.after_initial_load()       # e.g. CREATE INDEX on a SQL engine
        self.res["times"]["load"] = round(time.time() - t0, 2)
        for s in range(3):
            if self.fault and self.fault["kind"] == "desync" and s == 1:
                count = int(round(self.fault["frac"] * len(self.model.definite_live_ids())))
                self.eng.arm_desync(count)
            kinds = (["u"] * (self.p["SEG_UPDATES"] // self.p["BATCH"]) + ["d"] * (self.p["SEG_DELETES"] // self.p["BATCH"])
                     + ["i"] * (self.p["SEG_INSERTS"] // self.p["BATCH"]))
            order = self.rng.permutation(len(kinds))
            kill_at = int(self.rng.integers(len(kinds))) if self.kill else -1  # kill or "draw_only"
            delay = float(self.rng.uniform(0.0, self.p["KILL_DELAY_MAX"]))
            for bi, j in enumerate(order):
                op, ids, vecs = self._make_batch(kinds[j], pool)
                if bi == kill_at and self.kill == "draw_only":
                    self._no_crash(s, op, ids, vecs)
                elif bi == kill_at:
                    self._crash(s, op, ids, vecs, delay)
                else:
                    self._write(op, ids, vecs)
            self._segment_op(s)
        self.eng.stop()
        fl = os.path.join(self.dir, "faults.jsonl")
        self.res["fault_log"] = [json.loads(l) for l in open(fl)] if os.path.exists(fl) else []
        self.res["times"]["total"] = round(time.time() - t_start, 2)
        shutil.rmtree(self.dir, ignore_errors=True)
        self._finish()
        return self.res

    def _finish(self):
        """Holm over the run's statistical tests; candidates from EI violations."""
        if self.env is not None and self.res["tests"]:
            rej = S.holm(self.res["tests"], self.p["ALPHA_FW"])
            self.res["raw"]["stat"] = len(rej)
            for key, level in rej.items():
                op, stat = key.split("|")
                self.res["candidates"].append(dict(key=["STAT", op, stat], mode="stat",
                                                   level=level))
        seen = set()
        for v in self.res["violations"]:
            if "id" in v:
                key = ("EI", v["rule"], v["id"])
            else:
                key = ("EI", v["rule"], v["where"], v.get("query", -1))
            if key not in seen:
                seen.add(key)
                self.res["candidates"].append(dict(key=list(key), mode="ei"))


def run_once(**kw):
    return Run(**kw).execute()
