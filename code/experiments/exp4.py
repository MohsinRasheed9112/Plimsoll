"""Experiment 4: sensitivity, the smallest detectable fault (Sec. V-D, Table IV).

Usage (from code/):
  python experiments/exp4.py <cfg> calib                 calibration trials
  python experiments/exp4.py <cfg> envelopes             envelopes from calibration
  python experiments/exp4.py <cfg> trials [lo hi]        trials with seeds in [lo, hi)
  python experiments/exp4.py <cfg> smoke <seed> <phase>  one trial with a smoke seed

<cfg> is A (SIFT 100K) or B (SIFT 1M). Each trial writes one JSON file under
results/exp4/<cfg>/<phase>/; an existing file is not recomputed. The index is
hnswlib, driven directly with the reference engine's parameters. The oracles read
only answers and the logical model. Fault sets are recorded for evaluation; no
oracle reads them.
"""
import json
import os
import platform
import sys
import time
from fractions import Fraction

import hnswlib
import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from plimsoll import gt as G  # noqa: E402
from plimsoll import oracles as O  # noqa: E402
from plimsoll import stats as S  # noqa: E402
from plimsoll.contracts import CONTRACTS  # noqa: E402
from plimsoll.model import DELETED, LIVE, Model  # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "exp4")

P = dict(K=10, EF=64, M=16, EFC=100, THREADS=2, N_DEL=5_000, N_W=2_000, N_NEG=2_000,
         NQ=1_000, NOISE_SD=1.0, LEVEL=0.001, TAIL_DROP=0.5, N_BOOT_CSTAR=20_000, DIM=128)

CFGS = {
    "A": dict(name="sift100k", n=100_000, calib=range(4000, 4020), trials=range(4100, 4200),
              rebuild=True, degrade=True),
    "B": dict(name="sift1m", n=1_000_000, calib=range(5000, 5020), trials=range(5100, 5200),
              rebuild=True, degrade=True),
}

# fault grids in percent (exact decimal strings; counts use exact rational arithmetic)
GRID = dict(
    desync=["0.002", "0.005", "0.01", "0.02", "0.03", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"],
    lost=["0.01", "0.02", "0.05", "0.1", "0.15", "0.2", "0.3", "0.5", "1", "2", "5"],
    resurrect=["0.02", "0.05", "0.1", "0.15", "0.2", "0.3", "0.5", "1", "2", "5", "10", "20", "50"],
    degrade_M=[14, 13, 12, 11, 10, 8, 6],
)


def count_of(pct, base):
    """round(pct% of base), halves up, in exact arithmetic."""
    x = Fraction(pct) / 100 * base + Fraction(1, 2)
    return int(x.numerator // x.denominator)


# ------------------------------------------------------------------ data
_DATA = {}


def load(cfg):
    if cfg in _DATA:
        return _DATA[cfg]
    n = CFGS[cfg]["n"]
    if cfg == "A":
        X = np.load(os.path.join(DATA, "sift100k_base.npy"))
        Q = np.load(os.path.join(DATA, "sift100k_queries.npy"))[:P["NQ"]]
    else:
        X = np.empty((n, P["DIM"]), np.float32)
        for i in range(4):
            X[i * 250_000:(i + 1) * 250_000] = np.load(os.path.join(DATA, "sift1m_base_part%d.npy" % i))
        Q = np.load(os.path.join(DATA, "sift1m_queries.npy"))[:P["NQ"]]
    assert X.shape == (n, P["DIM"]) and X.dtype == np.float32 and Q.shape == (P["NQ"], P["DIM"])
    _DATA[cfg] = (np.ascontiguousarray(X), np.ascontiguousarray(Q))
    return _DATA[cfg]


# ----------------------------------------------------------------- index
class Index:
    """hnswlib with the reference engine's parameters; answers as lists of arrays."""

    def __init__(self, n, short):
        self.n = n
        self.h = None
        self.short = short  # shared counter: answers with fewer than k results

    def build(self, vecs, ids, M, seed):
        h = hnswlib.Index(space="l2", dim=P["DIM"])
        h.init_index(max_elements=self.n, M=M, ef_construction=P["EFC"], random_seed=int(seed))
        h.set_num_threads(P["THREADS"])
        h.add_items(vecs, np.asarray(ids, dtype=np.int64), num_threads=P["THREADS"])
        self.h = h
        return self

    def save(self, path):
        self.h.save_index(path)

    def load(self, path):
        h = hnswlib.Index(space="l2", dim=P["DIM"])
        h.load_index(path, max_elements=self.n)
        h.set_num_threads(P["THREADS"])
        self.h = h
        return self

    def search(self, V, ef):
        self.h.set_ef(int(ef))
        try:
            lab, dis = self.h.knn_query(V, k=P["K"], num_threads=P["THREADS"])
            return [r.astype(np.int64) for r in lab], [r for r in dis]
        except RuntimeError:
            ids, ds = [], []
            for v in V:
                got = False
                for kk in range(P["K"], 0, -1):
                    try:
                        lab, dis = self.h.knn_query(v[None, :], k=kk, num_threads=1)
                        ids.append(lab[0].astype(np.int64))
                        ds.append(dis[0])
                        got = True
                        break
                    except RuntimeError:
                        continue
                if not got:
                    ids.append(np.zeros(0, np.int64))
                    ds.append(np.zeros(0, np.float32))
                if len(ids[-1]) < P["K"]:
                    self.short[0] += 1
            return ids, ds


class RowsView:
    """The engine's row store as seen by the exact path of baseline B2."""

    def __init__(self, status, vec):
        self.status = status
        self.vec = vec


# ------------------------------------------------------------ measuring
class Trial:
    def __init__(self, cfg, seed, phase, env=None, c_star=None):
        self.cfg, self.seed, self.phase = cfg, int(seed), phase
        self.X, self.Q = load(cfg)
        self.N = len(self.X)
        self.contract = json.loads(json.dumps(CONTRACTS["hnsw"]))
        self.contract["theta_max_ef"] = self.N  # default of Sec. III-D: collection capacity
        self.c_star = c_star if c_star is not None else (env or {}).get("c_star")
        self.res = dict(cfg=cfg, dataset=CFGS[cfg]["name"], seed=self.seed, phase=phase,
                        params=P, times={}, healthy={}, faults={}, ei_totals={})
        self.ei_total = dict(decisions=0, bad=0)
        self.fault = phase in ("trial", "smoke_trial")
        self.short = [0]

    # EI over one batch of answers; returns (bad answers, violations)
    def _ei(self, model, V, ids, ds, where, tally):
        viols, nbad = O.ei_check_search(model, self.contract, V, ids, ds, P["K"], where)
        tally["decisions"] += len(ids)
        tally["bad"] += nbad
        for v in viols:
            tally["by_rule"][v["rule"]] = tally["by_rule"].get(v["rule"], 0) + 1
            if "id" in v:
                tally["ids"].add(int(v["id"]))
        tally["by_source"][where.split(":")[-1]] = tally["by_source"].get(where.split(":")[-1], 0) + nbad
        if len(tally["examples"]) < 3:
            tally["examples"] += viols[:3 - len(tally["examples"])]
        return nbad

    def measure(self, name, idx, model, W, NW, gt, esc, rows=None, gt_rows=None, ref=None,
                fault_set=None):
        """Search the 1,000 queries, the witnesses and the negative witnesses; apply EI
        to every answer; compute recall and the B2 overlap; escalate witness misses as
        the decision needs:
          esc == "all":            escalate every miss (exact persistent set)
          esc == "none":           no escalation (WQ not evaluated in this state)
          esc == ("new", base):    stop at the first new persistent miss (exact mode)
          esc == ("count", c):     stop once c persistent misses are found (count test)
        """
        t0 = time.time()
        Q = self.Q
        tally = dict(decisions=0, bad=0, by_rule={}, by_source={}, ids=set(), examples=[])
        se_ids, se_ds = idx.search(Q, P["EF"])
        self._ei(model, Q, se_ids, se_ds, name + ":se", tally)
        recall = O.recall_per_query(model, self.contract, Q, se_ids, gt, P["K"])
        hits = int(np.rint(recall * P["K"]).sum())
        if rows is not None:
            ov = O.recall_per_query(rows, self.contract, Q, se_ids, gt_rows, P["K"])
            b2_hits = int(np.rint(ov * P["K"]).sum())
        else:
            b2_hits = hits  # the engine's rows equal the logical state
        Vw = model.vec[W]
        w_ids, w_ds = idx.search(Vw, P["EF"])
        self._ei(model, Vw, w_ids, w_ds, name + ":wq", tally)
        first = np.array([r[0] if len(r) else -1 for r in w_ids])
        hit = O.wq_hits(model, self.contract, W, first)
        miss = np.flatnonzero(~hit)
        wq = dict(hit_rate=float(hit.mean()), misses=int(len(miss)))
        if esc == "all":
            persistent = np.zeros(len(W), dtype=bool)
            if len(miss):
                e_ids, e_ds = idx.search(Vw[miss], self.contract["theta_max_ef"])
                self._ei(model, Vw[miss], e_ids, e_ds, name + ":esc", tally)
                f2 = np.array([r[0] if len(r) else -1 for r in e_ids])
                persistent[miss] = ~O.wq_hits(model, self.contract, W[miss], f2)
            wq.update(escalated=int(len(miss)), persistent=int(persistent.sum()),
                      persistent_ids=[int(x) for x in W[persistent]])
            wq["_persistent"] = persistent
        elif esc != "none":
            kind, arg = esc
            if kind == "new":
                todo = [j for j in miss if not arg[j]]
                need = 1
            else:
                todo = list(miss)
                need = int(arg)
            found, n_esc = 0, 0
            for j in todo:
                if found >= need:
                    break
                e_ids, e_ds = idx.search(Vw[j:j + 1], self.contract["theta_max_ef"])
                self._ei(model, Vw[j:j + 1], e_ids, e_ds, name + ":esc", tally)
                n_esc += 1
                f2 = np.array([r[0] if len(r) else -1 for r in e_ids])
                if not O.wq_hits(model, self.contract, W[j:j + 1], f2)[0]:
                    found += 1
            wq.update(mode=kind, need=need, candidates=len(todo), escalated=n_esc,
                      persistent_found=found, alarm=bool(found >= need))
        n_ids, n_ds = idx.search(model.vec[NW], P["EF"])
        self._ei(model, model.vec[NW], n_ids, n_ds, name + ":neg", tally)
        out = dict(recall=float(recall.mean()), hits=hits, b2_hits=b2_hits,
                   gt_checked=int(gt.get("checked", 0)), gt_fallback=int(gt.get("n_fallback", 0)))
        if ref is not None:
            out["D"] = (ref["hits"] - hits) / float(P["K"] * len(Q))
            out["tail"] = int(np.count_nonzero(ref["_recall"] - recall > P["TAIL_DROP"]))
            out["changed"] = int(O.answers_identical(model, self.contract, Q, ref["_se"],
                                                     (se_ids, se_ds)).sum())
        flagged = np.array(sorted(tally["ids"]), dtype=np.int64)
        out["ei"] = dict(decisions=tally["decisions"], bad=tally["bad"], by_rule=tally["by_rule"],
                         by_source=tally["by_source"], n_flagged=int(len(flagged)),
                         examples=_jsonable(tally["examples"]))
        if len(flagged) <= 50:
            out["ei"]["flagged_ids"] = [int(x) for x in flagged]
        if fault_set is not None:
            # evaluation only: how many flagged items the fault actually touched
            out["ei"]["n_flagged_in_fault_set"] = int(np.isin(flagged, fault_set).sum())
        out["wq"] = {k: v for k, v in wq.items() if not k.startswith("_")}
        out["secs"] = round(time.time() - t0, 2)
        self.ei_total["decisions"] += tally["decisions"]
        self.ei_total["bad"] += tally["bad"]
        # in-memory extras for comparisons within the trial
        out["_recall"] = recall
        out["_se"] = (se_ids, se_ds)
        if "_persistent" in wq:
            out["_persistent"] = wq["_persistent"]
        return out

    # --------------------------------------------------------------- trial
    def run(self):
        t_all = time.time()
        X, Q, N = self.X, self.Q, self.N
        s = self.seed
        rng = np.random.default_rng([s, 0])    # workload
        wrng = np.random.default_rng([s, 1])   # witnesses and ground-truth checks
        frng = np.random.default_rng([s, 2])   # fault sets
        fault = self.fault
        order = rng.permutation(N)
        deleted = np.sort(rng.choice(N, P["N_DEL"], replace=False))
        model = Model(N, P["DIM"])
        model.status[:] = LIVE          # same as acknowledging the inserts of all items
        model.vec[:] = X
        model.status[deleted] = DELETED  # acknowledged deletes; vec keeps the last vector
        live = model.definite_live_ids()
        n_live = len(live)
        W = np.sort(wrng.choice(live, P["N_W"], replace=False))
        NW = np.sort(wrng.choice(deleted, P["N_NEG"], replace=False))
        if fault:
            draws = self._draw_faults(frng, live, deleted, n_live)
        t0 = time.time()
        idx = Index(N, self.short).build(X[order], order, P["M"], s)
        for i in deleted:
            idx.h.mark_deleted(int(i))
        self.res["times"]["build"] = round(time.time() - t0, 2)
        t0 = time.time()
        gt0 = G.compute(Q, model.vec, live, P["K"], self.contract, rng=wrng)
        self.res["times"]["gt0"] = round(time.time() - t0, 2)
        H0 = self.measure("S0", idx, model, W, NW, gt0, "all")
        self.res["healthy"]["S0"] = _strip(H0)
        snap = os.path.join(RES, self.cfg, "tmp_%s_%d.bin" % (self.phase, s))
        idx.save(snap)
        if fault:
            t0 = time.time()
            self.res["faults"]["desync"] = self._desync(idx, model, W, NW, gt0, draws)
            self.res["times"]["desync"] = round(time.time() - t0, 2)
            t0 = time.time()
            self.res["faults"]["resurrect"] = self._resurrect(idx, model, W, NW, gt0, draws)
            self.res["times"]["resurrect"] = round(time.time() - t0, 2)
        del idx
        # clean restart: load the snapshot (index-preserving, as calibration checks)
        t0 = time.time()
        idx = Index(N, self.short).load(snap)
        os.remove(snap)
        H1 = self.measure("S1", idx, model, W, NW, gt0, "all", ref=H0)
        H1["new_persistent"] = int(np.count_nonzero(H1["_persistent"] & ~H0["_persistent"]))
        self.res["healthy"]["S1"] = _strip(H1)
        if fault:
            self.res["faults"]["lost"] = self._lost(idx, model, W, NW, gt0, H0, draws["lost"],
                                                    ("new", H0["_persistent"]))
        self.res["times"]["restart_and_lost"] = round(time.time() - t0, 2)
        del idx
        if CFGS[self.cfg]["rebuild"]:
            t0 = time.time()
            idx = Index(N, self.short).build(X[live], live, P["M"], s + 1_000_003)
            H2 = self.measure("S2", idx, model, W, NW, gt0, "all", ref=H0)
            self.res["healthy"]["S2"] = _strip(H2)
            if fault:
                self.res["faults"]["lost_rebuild"] = self._lost(
                    idx, model, W, NW, gt0, H0, draws["lost_rebuild"], ("count", self.c_star))
            self.res["times"]["rebuild_and_lost"] = round(time.time() - t0, 2)
            del idx
        if CFGS[self.cfg]["degrade"] and fault:
            t0 = time.time()
            out = []
            for m in GRID["degrade_M"]:
                idx = Index(N, self.short).build(X[live], live, m, s + 2_000_003 + m)
                r = self.measure("M%d" % m, idx, model, W, NW, gt0, ("count", self.c_star), ref=H0)
                r["M"] = m
                out.append(_strip(r))
                del idx
            self.res["faults"]["degrade"] = out
            self.res["times"]["degrade"] = round(time.time() - t0, 2)
        self.res["n_live"] = int(n_live)
        self.res["ei_totals"] = self.ei_total
        self.res["short_answers"] = self.short[0]
        self.res["times"]["total"] = round(time.time() - t_all, 2)
        return self.res

    def _draw_faults(self, frng, live, deleted, n_live):
        """All fault sets and new vectors, drawn before any measurement. Levels are
        nested: the set of a level is a prefix of one random permutation."""
        d_max = count_of(GRID["desync"][-1], n_live)
        dperm = frng.permutation(live)[:d_max]
        src = frng.integers(0, self.N, size=d_max)
        newv = (self.X[src] + frng.normal(0.0, P["NOISE_SD"], size=(d_max, P["DIM"]))).astype(np.float32)
        r_max = count_of(GRID["resurrect"][-1], P["N_DEL"])
        rperm = frng.permutation(deleted)[:r_max]
        l_max = count_of(GRID["lost"][-1], n_live)
        lperm = frng.permutation(live)[:l_max]
        lperm2 = frng.permutation(live)[:l_max]
        return dict(desync=(dperm, newv), resurrect=rperm, lost=lperm, lost_rebuild=lperm2)

    def _desync(self, idx, model, W, NW, gt0, draws):
        dperm, newv = draws["desync"]
        n_live = len(gt0["live"])
        out = []
        for pct in GRID["desync"]:
            c = count_of(pct, n_live)
            D = dperm[:c]
            model.vec[D] = newv[:c]  # rows and model hold the new vectors; the index does not
            gt = G.update(gt0, self.Q, model.vec, removed=D, added=D, k=P["K"],
                          contract=self.contract, rng=np.random.default_rng([self.seed, 3, c]))
            r = self.measure("desync", idx, model, W, NW, gt, "none", fault_set=D)
            r.update(pct=pct, count=c, witnesses_in_fault_set=int(np.isin(W, D).sum()))
            out.append(_strip(r))
        model.vec[dperm] = self.X[dperm]  # restore
        return out

    def _resurrect(self, idx, model, W, NW, gt0, draws):
        rperm = draws["resurrect"]
        rows_status = model.status.copy()
        out = []
        prev = 0
        for pct in GRID["resurrect"]:
            c = count_of(pct, P["N_DEL"])
            for i in rperm[prev:c]:
                idx.h.unmark_deleted(int(i))   # undeleted in the index
                rows_status[i] = LIVE          # and in the rows
            prev = c
            R = rperm[:c]
            gt_rows = G.update(gt0, self.Q, model.vec, removed=[], added=R, k=P["K"],
                               contract=self.contract, rng=np.random.default_rng([self.seed, 4, c]))
            r = self.measure("resurrect", idx, model, W, NW, gt0, "none",
                             rows=RowsView(rows_status, model.vec), gt_rows=gt_rows, fault_set=R)
            r.update(pct=pct, count=c, negatives_in_fault_set=int(np.isin(NW, R).sum()))
            out.append(_strip(r))
        return out

    def _lost(self, idx, model, W, NW, gt0, H0, lperm, esc):
        n_live = len(gt0["live"])
        out = []
        prev = 0
        for pct in GRID["lost"]:
            c = count_of(pct, n_live)
            for i in lperm[prev:c]:
                idx.h.mark_deleted(int(i))     # lost from the index only; rows stay live
            prev = c
            r = self.measure("lost", idx, model, W, NW, gt0, esc, ref=H0, fault_set=lperm[:c])
            r.update(pct=pct, count=c, witnesses_in_fault_set=int(np.isin(W, lperm[:c]).sum()))
            out.append(_strip(r))
        return out


def _strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ------------------------------------------------------------- drivers
def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                hnswlib=getattr(hnswlib, "__version__", "0.8.0 (pip metadata)"),
                machine=platform.machine(), processor=platform.processor(),
                cpus=os.cpu_count(), node=platform.node(), platform=platform.platform())


def run_seed(cfg, seed, phase, env=None, c_star=None, force=False):
    d = os.path.join(RES, cfg, phase)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "%d.json" % seed)
    if os.path.exists(path) and not force:
        return None
    res = Trial(cfg, seed, phase, env=env, c_star=c_star).run()
    res["environment"] = environment()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)
    return res


def envelopes(cfg):
    d = os.path.join(RES, cfg, "calib")
    runs = [json.load(open(os.path.join(d, "%d.json" % s))) for s in CFGS[cfg]["calib"]]
    env = dict(seeds=[r["seed"] for r in runs])
    for st in ("S0", "S1", "S2"):
        if st not in runs[0]["healthy"]:
            continue
        h = [r["healthy"][st] for r in runs]
        env[st] = dict(recall=[x["recall"] for x in h], b2=[x["b2_hits"] / 1e4 for x in h],
                       persistent=[x["wq"]["persistent"] for x in h],
                       ei_bad=[x["ei"]["bad"] for x in h])
        if st != "S0":
            env[st].update(D=[x["D"] for x in h], tail=[x["tail"] for x in h],
                           changed=[x["changed"] for x in h])
        if st == "S1":
            env[st]["new_persistent"] = [x["new_persistent"] for x in h]
    env["index_preserving_restart"] = bool(all(c == 0 for c in env["S1"]["changed"])
                                           and all(c == 0 for c in env["S1"]["new_persistent"]))
    env["ei_bad_total"] = int(sum(r["ei_totals"]["bad"] for r in runs))
    env["ei_decisions_total"] = int(sum(r["ei_totals"]["decisions"] for r in runs))
    if "S2" in env:
        env["c_star"], env["c_star_rule"] = c_star_from(env["S2"]["persistent"], P["N_W"])
    else:
        env["c_star"], env["c_star_rule"] = None, None
    with open(os.path.join(RES, cfg, "envelopes.json"), "w") as f:
        json.dump(env, f, indent=1)
    return env


def c_star_from(counts, n):
    """Smallest persistent-miss count that the WQ count test rejects at the level.
    All-zero calibration counts: the zero rule (any persistent miss)."""
    if max(counts) == 0:
        return 1, "zero"
    for c in range(1, n + 1):
        p, _, _ = S.count_pvalue(c, counts, n, P["N_BOOT_CSTAR"], np.random.default_rng([c, 77]))
        if p <= P["LEVEL"]:
            return c, "betabinom p=%.6f" % p
    return None, "none"


def main(argv):
    cfg, what = argv[1], argv[2]
    os.makedirs(os.path.join(RES, cfg), exist_ok=True)
    with open(os.path.join(RES, "environment.txt"), "a") as f:
        f.write(json.dumps(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), cfg=cfg,
                                what=what, **environment())) + "\n")
    if what == "calib":
        lo = int(argv[3]) if len(argv) > 3 else CFGS[cfg]["calib"].start
        hi = int(argv[4]) if len(argv) > 4 else CFGS[cfg]["calib"].stop
        for s in range(lo, hi):
            r = run_seed(cfg, s, "calib")
            if r:
                print(cfg, "calib", s, r["times"], flush=True)
    elif what == "envelopes":
        env = envelopes(cfg)
        print(json.dumps({k: v for k, v in env.items() if k not in ("S0", "S1", "S2")}, indent=1))
    elif what == "trials":
        env = json.load(open(os.path.join(RES, cfg, "envelopes.json")))
        if not env["index_preserving_restart"]:
            raise SystemExit("restart was not index-preserving in calibration; the trials require it")
        lo = int(argv[3]) if len(argv) > 3 else CFGS[cfg]["trials"].start
        hi = int(argv[4]) if len(argv) > 4 else CFGS[cfg]["trials"].stop
        for s in range(lo, hi):
            r = run_seed(cfg, s, "trial", env=env)
            if r:
                print(cfg, "trial", s, r["times"], flush=True)
    elif what == "smoke":
        s, phase = int(argv[3]), argv[4]
        assert s >= 9000
        r = run_seed(cfg, s, "smoke_" + phase, c_star=1, force=True)
        print(json.dumps(r["times"]), "EI", r["ei_totals"])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
