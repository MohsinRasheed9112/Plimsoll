"""One campaign run (paper Algorithm 2, Experiments 2, 3 and 7): a seeded random
scenario of writes, maintenance operations and process kills on one engine, judged by
EI and by the exact modes of WQ and SE (Secs. III-C to III-E and III-G).

The scenario, its batches and its kill points come from the seed; kill timing and the
engine's internal scheduling do not, so a replay repeats the scenario but not
necessarily the outcome. A candidate is identified by (oracle, rule, step), and it
counts as reproduced when a replay raises the same key.

Checks after each maintenance operation, each kill, and every third write step:
  EI      every answer: existence, filter, score, order (except documented relaxed
          ordering), uniqueness; lookups of recently written items and of a sample
  WQ      self-queries of sampled definite items, unfiltered and with a filter that the
          item satisfies; misses are escalated to theta_max
  SE      the same queries before and after an index-preserving operation
Right after every restart or recovery, before any other read, all items written since
the previous check are looked up (durability and visibility of acknowledged writes).
"""
import json
import os
import shutil
import threading
import time

import numpy as np

from . import oracles as O
from .amodel import AModel, ei_check_filter
from .tolerance import distances_and_eps
from .refengine.client import EngineError, EngineUnavailable

CP = dict(N_POOL=20_000, N_INITIAL=15_000, BATCH_INITIAL=1_000, DIM=128, K=10, EF=64, N_ATTR=100,
          STEPS=24, BATCH_MIN=50, BATCH_MAX=200, KILL_BATCH_MIN=500, KILL_BATCH_MAX=3_000,
          P_WRITE=0.6, P_KILL=0.35, NOISE_SD=1.0,
          CHECK_Q=200, CHECK_W=300, CHECK_NEG=200, CHECK_NEGF=200, CHECK_LOOKUP=300,
          KILL_DELAY_WRITE=0.1, BG_KILL_MAX=2.0, CONCURRENT_BATCHES=0, CONCURRENT_SIZE=100, OP_CONCURRENT=0,
          KILL_DELAY_OP=dict(flush=0.5, restart=1.0, rebuild=30.0, vacuum=5.0, attr_index=2.0,
                             reindex_concurrently=30.0, optimize=5.0, snapshot_restore=5.0,
                             attr_index_drop=1.0))
FILTERS = [None, ("eq", None), ("lt", 10), ("lt", 50)]  # ("eq", None): a random value, 1%


class CampaignRun:
    def __init__(self, engine_factory, contract, seed, data, workdir, tag="main", params=None):
        self.p = dict(CP)
        self.p.update(params or {})
        self.seed = int(seed)
        self.base, self.queries = data
        self.contract = contract
        self.rng = np.random.default_rng([self.seed, 10])     # scenario
        self.crng = np.random.default_rng([self.seed, 11])    # checks
        self.dir = os.path.join(workdir, "camp_%d_%s" % (self.seed, tag))
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.eng = engine_factory(self.dir, self.seed)
        self.m = AModel(self.p["N_POOL"], self.p["DIM"])
        self.recent = set()      # written since the last check
        self.unsettled = set()   # written since the last restart or recovery
        self.step = -1
        self.res = dict(seed=self.seed, tag=tag, engine=getattr(self.eng, "name", "?"), params=self.p,
                        code=_code_hashes(),
                        history=[], checks=[], violations=[], candidates=[], times={},
                        decisions=dict(ei_answers=0, lookups=0, wq_exact=0, se_exact=0),
                        errors=[],
                        ablation=dict(indeterminate=[],
                                      naive_contract=dict(answers=0, any=0, score=0, order=0, size=0)))

    # ------------------------------------------------------------ helpers
    def _attrs(self, n):
        return self.rng.integers(0, self.p["N_ATTR"], size=n)

    def _newvecs(self, n):
        src = self.rng.integers(0, len(self.base), size=n)
        return (self.base[src] + self.rng.normal(0.0, self.p["NOISE_SD"], size=(n, self.p["DIM"]))).astype(np.float32)

    def _viol(self, vs):
        for v in vs:
            v["step"] = self.step
        self.res["violations"] += vs

    def _record(self, **kw):
        kw["step"] = self.step
        self.res["history"].append(kw)

    # -------------------------------------------------------------- writes
    def _make_write(self, kind, size):
        live = self.m.definite_live_ids()
        if kind == "insert":
            free = np.flatnonzero(self.m.status == 0)
            if len(free) == 0:
                kind = "update"
            else:
                ids = np.sort(self.rng.choice(free, size=min(size, len(free)), replace=False))
                return kind, ids, self.base[ids].copy(), self._attrs(len(ids))
        ids = np.sort(self.rng.choice(live, size=min(size, len(live)), replace=False))
        if kind == "update":
            return kind, ids, self._newvecs(len(ids)), None
        if kind == "set_attr":
            return kind, ids, None, self._attrs(len(ids))
        return "delete", ids, None, None

    def _write_kind(self):
        return str(self.rng.choice(["insert", "update", "set_attr", "delete"], p=[0.3, 0.3, 0.2, 0.2]))

    def _do_write(self, op, ids, vecs, attrs):
        try:
            self.eng.write(op, ids, vecs, attrs)
        except EngineError:
            # the engine refused a write on items the model holds: look them up first
            lv, vv = self.eng.lookup([int(i) for i in ids])
            bad = []
            for i, l, v in zip(ids, lv, vv):
                bad += self._lookup_rule(int(i), l, v, "write_error:%s" % op)
            self._viol(bad)
            self._record(action="write_error", op=op, n=len(ids), missing=len(bad))
            raise
        self.m.ack(op, ids, vecs, attrs)
        self._touch(ids)

    def _touch(self, ids):
        t = set(int(i) for i in ids)
        self.recent |= t
        self.unsettled |= t

    # --------------------------------------------------------------- kills
    def _killed(self, fn, delay):
        """Run fn while a timer sends SIGKILL after delay seconds. Returns
        ('ack', None) if fn returned before the kill took effect, else ('unknown', exc)."""
        timer = threading.Timer(delay, self.eng.signal_kill)
        timer.start()
        try:
            fn()
            outcome, err = "ack", None
        except Exception as e:  # any failure while the kill races the operation
            outcome, err = "unknown", repr(e)[:300]
        timer.join()
        return outcome, err

    def _recover(self):
        """Kill for certain, restart, and resolve every indeterminate item by lookup
        before any other read; also look up every recently written item."""
        self.eng.kill()
        t0 = time.time()
        self.eng.recover()
        t_rec = time.time() - t0
        self._post_restart_lookups("recover")
        return t_rec

    def _post_restart_lookups(self, where):
        indet = sorted(self.m.indet.keys())
        recent = sorted(self.unsettled - set(indet))
        self.unsettled = set()
        bad = []
        # acknowledged writes: durable and visible as soon as the engine serves reads
        if recent:
            lv, vv = self.eng.lookup(recent)
            self.res["decisions"]["lookups"] += len(recent)
            for i, l, v in zip(recent, lv, vv):
                bad += self._lookup_rule(i, l, v, where + ":recent")
        # unacknowledged writes: the engine may still apply one after recovery (ClickHouse
        # resumes a stored mutation), so their items are resolved only once the engine
        # reports no pending work
        if indet:
            if hasattr(self.eng, "quiesce"):
                self.eng.quiesce()
            lv, vv = self.eng.lookup(indet)
            self.res["decisions"]["lookups"] += len(indet)
            res = dict(applied=0, not_applied=0, other=0)
            from .amodel import _same
            for i, l, v in zip(indet, lv, vv):
                outs = self.m.indet[i]
                obs = v if l else None
                res["not_applied" if _same(obs, outs[0]) else "applied" if any(_same(obs, o) for o in outs[1:])
                    else "other"] += 1
                if not self.m.resolve(i, obs):
                    bad.append(dict(rule="lookup", where=where + ":resolve", id=int(i)))
            self.res["ablation"]["indeterminate"].append(dict(step=self.step, **res))
        self._viol(bad)

    def _lookup_rule(self, i, live, val, where):
        s = self.m.status[i]
        if s == 1:
            if not live:
                return [dict(rule="lookup", where=where, id=int(i), expected="live", got="absent")]
            if not np.array_equal(val[0], self.m.vec[i]):
                return [dict(rule="lookup", where=where, id=int(i), expected="live", got="other vector")]
            if int(val[1]) != int(self.m.attr[i]):
                return [dict(rule="lookup", where=where, id=int(i), expected="attr %d" % self.m.attr[i],
                             got="attr %d" % int(val[1]))]
        elif s in (0, 2) and live:
            return [dict(rule="lookup", where=where, id=int(i), expected="absent", got="live")]
        return []

    # --------------------------------------------------------------- checks
    def _plan_check(self):
        """Queries, filters and witnesses for one check round (fixed before searching)."""
        c = self.crng
        live = self.m.definite_live_ids()
        dead = self.m.deleted_ids()
        qi = c.choice(len(self.queries), size=self.p["CHECK_Q"], replace=False)
        filts = []
        for _ in qi:
            f = FILTERS[int(c.integers(len(FILTERS)))]
            if f is not None and f[1] is None:
                f = ("eq", int(c.integers(self.p["N_ATTR"])))
            filts.append(f)
        iters = [self.eng.ITERATIVE[int(c.integers(len(self.eng.ITERATIVE)))] for _ in qi]
        W = np.sort(c.choice(live, size=min(self.p["CHECK_W"], len(live)), replace=False))
        NW = np.sort(c.choice(dead, size=min(self.p["CHECK_NEG"], len(dead)), replace=False)) if len(dead) else np.zeros(0, np.int64)
        NF = np.sort(c.choice(live, size=min(self.p["CHECK_NEGF"], len(live)), replace=False))
        nf_val = [(int(self.m.attr[i]) + 1 + int(c.integers(self.p["N_ATTR"] - 1))) % self.p["N_ATTR"] for i in NF]
        LK = np.sort(c.choice(self.p["N_POOL"], size=self.p["CHECK_LOOKUP"], replace=False))
        return dict(qi=qi, filts=filts, iters=iters, W=W, NW=NW, NF=NF, nf_val=nf_val, LK=LK)

    def _search(self, V, ef, filt, it, where):
        ids, ds = self.eng.search(V, self.p["K"], ef, filt=filt, iterative=it)
        c = dict(self.contract)
        if c["size"].get("promised") == "unfiltered":   # e.g. SQL LIMIT, except post-filtering
            c = dict(c, size=dict(c["size"], promised=filt is None))
        if it in self.eng.UNORDERED:
            c = dict(c, order=dict(c["order"], follows="none"))
        v, nbad = O.ei_check_search(self.m, c, np.asarray(V, np.float32), ids, ds, self.p["K"], where)
        v += ei_check_filter(self.m, ids, filt, where)
        self.res["decisions"]["ei_answers"] += len(ids)
        self._viol(v)
        self._naive(np.asarray(V, np.float32), ids, ds, filt)
        return ids, ds

    NAIVE = {"score": {"on": True, "formula": "direct", "unit_roundoff": 2.0 ** -53},
             "order": {"follows": "reported"}, "size": {"promised": False}}

    def _naive(self, V, ids, ds, filt):
        """Ablation, no contract layer: scores taken as exact float64 distances, strict
        order in every mode, and min(k, eligible items) results promised. Counted, never
        used for a decision."""
        a = self.res["ablation"]["naive_contract"]
        k = self.p["K"]
        live = self.m.definite_live_ids()
        if filt is None:
            elig = len(live)
        elif filt[0] == "eq":
            elig = int(np.count_nonzero(self.m.attr[live] == filt[1]))
        else:
            elig = int(np.count_nonzero(self.m.attr[live] < filt[1]))
        for q, i_, d_ in zip(V, ids, ds):
            a["answers"] += 1
            hit = False
            if len(i_) < min(k, elig):
                a["size"] += 1
                hit = True
            if any(d_[j] > d_[j + 1] for j in range(len(d_) - 1)):
                a["order"] += 1
                hit = True
            sel = [j for j, i in enumerate(i_) if self.m.status[i] == 1]
            if sel:
                from .tolerance import distances_and_eps
                delta, eps = distances_and_eps(q, self.m.vec[[i_[j] for j in sel]], self.NAIVE)
                if np.any(np.abs(np.asarray([d_[j] for j in sel]) - delta) > eps):
                    a["score"] += 1
                    hit = True
            a["any"] += int(hit)

    def _start_concurrent(self, n=None):
        """Write batches from a second client while the check reads, or while a
        maintenance operation runs. Their items are indeterminate in the model until
        the batches end, so every answer and lookup is judged against both outcomes."""
        n = self.p["CONCURRENT_BATCHES"] if n is None else n
        if not n or not hasattr(self.eng, "concurrent_writer"):
            return None
        batches, used = [], set()
        for _ in range(n):
            kind = self._write_kind()
            op, ids, vecs, attrs = self._make_write(kind, self.p["CONCURRENT_SIZE"])
            keep = np.array([int(i) not in used for i in ids])
            ids = ids[keep]
            vecs = None if vecs is None else vecs[keep]
            attrs = None if attrs is None else np.asarray(attrs)[keep]
            if len(ids) == 0:
                continue
            used.update(int(i) for i in ids)
            news = []
            for j, i in enumerate(ids):
                cur = self.m._outcome(int(i))[0]
                news.append(self.m._apply(op, int(i), None if vecs is None else vecs[j],
                                          None if attrs is None else attrs[j])(cur))
            self.m.unknown(op, ids, vecs, attrs)
            batches.append(dict(op=op, ids=ids, vecs=vecs, attrs=attrs, news=news, outcome=None))
        writer = self.eng.concurrent_writer()

        def run():
            for b in batches:
                try:
                    writer.write(b["op"], b["ids"], b["vecs"], b["attrs"])
                    b["outcome"] = "ack"
                except Exception as e:
                    b["outcome"] = "error"
                    b["err"] = repr(e)[:200]
        th = threading.Thread(target=run)
        th.start()
        return dict(thread=th, batches=batches, writer=writer)

    def _finish_concurrent(self, cc, name, replaced=False):
        """replaced: the operation that ran alongside replaces the collection's state
        (Qdrant's snapshot then restore reverts it to the snapshot's time), so even an
        acknowledged concurrent write may have been reverted: its items stay indeterminate
        and are resolved by lookup."""
        if cc is None:
            return
        cc["thread"].join()
        if hasattr(cc["writer"], "close") and cc["writer"] is not self.eng:
            cc["writer"].close()
        unresolved = []
        for b in cc["batches"]:
            for i, new in zip(b["ids"], b["news"]):
                if int(i) not in self.m.indet:   # the write cannot change it (same outcome)
                    continue
                if b["outcome"] == "ack" and not replaced:
                    self.m.settle(int(i), new)
                else:
                    unresolved.append(int(i))
            self._touch(b["ids"])
        if unresolved:      # a failed concurrent batch: resolve its items by lookup
            lv, vv = self.eng.lookup(unresolved)
            bad = [dict(rule="lookup", where=name + ":concurrent_resolve", id=i)
                   for i, l, v in zip(unresolved, lv, vv) if not self.m.resolve(i, v if l else None)]
            self._viol(bad)
        # acknowledged concurrent writes must be visible now
        ids = [] if replaced else sorted(int(i) for b in cc["batches"] if b["outcome"] == "ack" for i in b["ids"])
        if ids:
            lv, vv = self.eng.lookup(ids)
            self.res["decisions"]["lookups"] += len(ids)
            bad = []
            for i, l, v in zip(ids, lv, vv):
                bad += self._lookup_rule(i, l, v, name + ":concurrent_lookup")
            self._viol(bad)
        self.res.setdefault("concurrent", []).append(dict(
            step=self.step, batches=len(cc["batches"]), ops=[b["op"] for b in cc["batches"]],
            outcomes=[b["outcome"] for b in cc["batches"]], items=sum(len(b["ids"]) for b in cc["batches"])))

    def _check(self, name, plan=None):
        """One check round; returns the answers for an exact-mode comparison."""
        if hasattr(self.eng, "quiesce"):
            self.eng.quiesce()
        t0 = time.time()
        cc = self._start_concurrent() if plan is None and not name.endswith(".pre") else None
        pl = plan or self._plan_check()
        out = dict(se=[], wq={})
        # queries, grouped by (filter, iterative mode) so each group is one batch
        Q = self.queries[pl["qi"]]
        exact_now = bool(self.contract.get("exact_after_delete")
                         and getattr(self.eng, "exactness_promised", lambda: False)())
        for j in range(len(Q)):
            ids, ds = self._search(Q[j:j + 1], self.p["EF"], pl["filts"][j], pl["iters"][j], name + ":q")
            out["se"].append((ids[0], ds[0]))
            if exact_now and pl["filts"][j] is None:
                self._exact_rule(Q[j], ids[0], ds[0], name + ":q_exact")
        # witnesses: unfiltered, then filtered on their own attribute
        theta = self.contract["theta_max_ef"]
        W = pl["W"]
        first_hits = {}
        for mode in ("plain", "filtered"):
            persistent = []
            first_hits[mode] = [0, len(W)]
            for w in W:
                f = None if mode == "plain" else ("eq", int(self.m.attr[w]))
                ids, _ = self._search(self.m.vec[w][None, :], self.p["EF"], f, None, name + ":wq")
                first = np.array([ids[0][0] if ids[0] else -1])
                if O.wq_hits(self.m, self.contract, np.array([w]), first)[0]:
                    first_hits[mode][0] += 1
                    continue
                ids2, ds2 = self._search(self.m.vec[w][None, :], theta, f, None, name + ":wq_esc")
                first2 = np.array([ids2[0][0] if ids2[0] else -1])
                if not O.wq_hits(self.m, self.contract, np.array([w]), first2)[0]:
                    persistent.append(int(w))
                    if len(self.res.setdefault("wq_miss_examples", [])) < 20:
                        ex = dict(step=self.step, where=name, mode=mode, w=int(w), attr=int(self.m.attr[w]),
                                  first_ef=[int(x) for x in ids[0][:3]], escalated=[int(x) for x in ids2[0][:3]],
                                  escalated_d=[float(x) for x in ds2[0][:3]], n_escalated=len(ids2[0]))
                        # triage aid (never part of a decision): the row path and the exact path
                        try:
                            lv, vv = self.eng.lookup([int(w)])
                            ex["lookup_live"] = bool(lv[0])
                            if hasattr(self.eng, "exact_search"):
                                ei, ed = self.eng.exact_search(self.m.vec[w][None, :], 3, filt=f)
                                ex["exact_path"] = [int(x) for x in ei[0]]
                            if hasattr(self.eng, "diagnose"):
                                ex["diagnose"] = self.eng.diagnose(int(w))
                        except Exception as e:  # diagnostics must never change the run
                            ex["diagnose_error"] = repr(e)[:200]
                        self.res["wq_miss_examples"].append(ex)
            out["wq"][mode] = persistent
            # zero rule: where calibration found no persistent miss at an exhaustive
            # theta_max (theta_max >= collection capacity), a persistent miss is a candidate
            # for replay and triage. The contract flag records that calibration result.
            if persistent and self.contract.get("wq_zero_rule") and theta is not None and theta >= self.p["N_POOL"]:
                self._viol([dict(rule="wq_persistent_zero", where=name + ":" + mode, ids=persistent[:20],
                                 count=len(persistent))])
        # negative witnesses: deleted items, and live items under a filter they fail
        for w in pl["NW"]:
            self._search(self.m.vec[w][None, :], self.p["EF"], None, None, name + ":neg")
        for w, val in zip(pl["NF"], pl["nf_val"]):
            self._search(self.m.vec[w][None, :], self.p["EF"], ("eq", val), None, name + ":negf")
        # lookups: recent writes and a sample
        ids = sorted(set(int(i) for i in pl["LK"]) | self.recent)
        lv, vv = self.eng.lookup(ids)
        self.res["decisions"]["lookups"] += len(ids)
        bad = []
        for i, l, v in zip(ids, lv, vv):
            bad += self._lookup_rule(i, l, v, name + ":lookup")
        self._viol(bad)
        self.recent = set()
        self._finish_concurrent(cc, name)
        rec = dict(step=self.step, name=name, secs=round(time.time() - t0, 2),
                   wq_persistent={k: len(v) for k, v in out["wq"].items()}, wq_first_hits=first_hits)
        if name == "init":     # operating point: recall@10 of the check's queries
            rec["recall_at_k"] = self._recall(Q, pl["filts"], out["se"])
            # by filter kind; the operating point is the unfiltered recall
            by = {}
            for kind in ("none", "eq", "lt10", "lt50"):
                sel = [j for j, f in enumerate(pl["filts"])
                       if (f is None and kind == "none") or (f is not None and
                           (f[0] == "eq" and kind == "eq" or f == ("lt", 10) and kind == "lt10"
                            or f == ("lt", 50) and kind == "lt50"))]
                if sel:
                    by[kind] = self._recall(Q[sel], [pl["filts"][j] for j in sel], [out["se"][j] for j in sel])
            rec["recall_at_k_by_filter"] = by
        self.res["checks"].append(rec)
        out["plan"] = pl
        out["token"] = self.eng.state_token() if hasattr(self.eng, "state_token") else None
        return out

    def _recall(self, Q, filts, answers):
        """Mean recall@k of the answers against exact float64 top-k over the definite live
        items that pass each query's filter (recorded, never part of a decision)."""
        live = self.m.definite_live_ids()
        k = self.p["K"]
        V = self.m.vec[live].astype(np.float64)
        vn = np.einsum("ij,ij->i", V, V)
        out = []
        for q, f, (ids, _) in zip(np.asarray(Q, np.float64), filts, answers):
            a = self.m.attr[live]
            ok = np.ones(len(live), bool) if f is None else (a == f[1]) if f[0] == "eq" else (a < f[1])
            if not ok.any():
                continue
            d = vn[ok] - 2.0 * (V[ok] @ q)
            top = set(int(x) for x in live[ok][np.argsort(d, kind="stable")[:k]])
            out.append(len(top & set(int(i) for i in ids)) / min(k, int(ok.sum())))
        return float(np.mean(out)) if out else None

    def _exact_rule(self, q, ids, ds, where):
        """Contract rule for states in which the engine documents an exact search: the
        answer holds min(k, live) items, and its i-th distance equals the i-th exact
        distance within both tolerances."""
        live = self.m.definite_live_ids()
        k = self.p["K"]
        delta, eps = distances_and_eps(q, self.m.vec[live], self.contract)
        order = np.argsort(delta, kind="stable")[:k]
        need = min(k, len(live))
        self.res["decisions"]["exact_rule"] = self.res["decisions"].get("exact_rule", 0) + 1
        if len(ids) != need:
            self._viol([dict(rule="exact_size", where=where, got=len(ids), expected=need)])
            return
        for i in range(need):
            d_true, e_true = delta[order[i]], eps[order[i]]
            if abs(ds[i] - d_true) > 2.0 * e_true + 1e-9:
                self._viol([dict(rule="exact_rank", where=where, rank=i, reported=float(ds[i]),
                                 exact=float(d_true), id=int(ids[i]))])
                return

    def _compare(self, name, a, b):
        """Exact modes across an index-preserving operation. Where the engine exposes its
        index state (ClickHouse: the active parts), a change of that state means a
        background merge rebuilt an index, so the exact modes do not apply."""
        if a.get("token") != b.get("token"):
            self.res.setdefault("compare_skipped", []).append(dict(step=self.step, where=name))
            return
        changed = 0
        for (ia, da), (ib, db) in zip(a["se"], b["se"]):
            sa, sb = dict(zip(ia, da)), dict(zip(ib, db))
            if len(ia) != len(ib) or set(sa) != set(sb) or any(
                    abs(sa[i] - sb[i]) > 1e-9 * max(1.0, abs(sa[i])) for i in sa):
                changed += 1
        self.res["decisions"]["se_exact"] += len(a["se"])
        if changed:
            self._viol([dict(rule="se_changed", where=name, count=changed)])
        for mode in ("plain", "filtered"):
            new = sorted(set(b["wq"][mode]) - set(a["wq"][mode]))
            self.res["decisions"]["wq_exact"] += len(a["plan"]["W"])
            if new:
                self._viol([dict(rule="wq_new_persistent", where=name + ":" + mode, ids=new[:20], count=len(new))])

    # ----------------------------------------------------------------- run
    def execute(self):
        try:
            return self._execute()
        except Exception as e:  # keep the history: an engine error is a finding for triage
            import traceback
            self.res["aborted"] = dict(step=self.step, error=repr(e)[:2000],
                                       traceback=traceback.format_exc()[-4000:])
            self.res["candidates"] = []
            self._candidates()
            self.res["candidates"].append(dict(key=["ERR", "engine_error", self.step],
                                               example=dict(error=repr(e)[:500])))
            return self.res
        finally:
            try:
                self.eng.kill()
            except Exception:
                pass
            try:
                self.eng.stop()
            except Exception:
                pass
            shutil.rmtree(self.dir, ignore_errors=True)

    def _execute(self):
        p = self.p
        t_start = time.time()
        self.eng.start(build_seed=self.seed)
        if hasattr(self.eng, "configure_campaign"):
            self.eng.configure_campaign()
        perm = self.rng.permutation(p["N_POOL"])
        initial = np.sort(perm[:p["N_INITIAL"]])
        for b in range(0, len(initial), p["BATCH_INITIAL"]):
            ids = initial[b:b + p["BATCH_INITIAL"]]
            self._do_write("insert", ids, self.base[ids].copy(), self._attrs(len(ids)))
        if hasattr(self.eng, "after_initial_load"):
            self.eng.after_initial_load()
        self.recent = set()
        self._check("init")
        ops = self.eng.maintenance_ops()
        for s in range(p["STEPS"]):
            self.step = s
            kill = self.rng.random() < p["P_KILL"]
            if self.rng.random() < p["P_WRITE"]:
                kind = self._write_kind()
                size = int(self.rng.integers(p["KILL_BATCH_MIN"] if kill else p["BATCH_MIN"],
                                             (p["KILL_BATCH_MAX"] if kill else p["BATCH_MAX"]) + 1))
                op, ids, vecs, attrs = self._make_write(kind, size)
                if not kill:
                    self._do_write(op, ids, vecs, attrs)
                    self._record(action="write", op=op, n=len(ids))
                    if s % 3 == 2:
                        self._check("s%d" % s)
                    continue
                if self.rng.random() < 0.5:      # kill while the write is in flight
                    delay = float(self.rng.uniform(0, p["KILL_DELAY_WRITE"]))
                    outcome, err = self._killed(
                        lambda op=op, ids=ids, vecs=vecs, attrs=attrs: self.eng.write(op, ids, vecs, attrs), delay)
                    if outcome == "ack":
                        self.m.ack(op, ids, vecs, attrs)
                    else:
                        self.m.unknown(op, ids, vecs, attrs)
                    self._touch(ids)
                    t_rec = self._recover()
                    self._record(action="kill_write", op=op, n=len(ids), delay=delay, outcome=outcome,
                                 err=err, recover_secs=round(t_rec, 2))
                else:                              # write, then kill in the background
                    self._do_write(op, ids, vecs, attrs)
                    delay = float(self.rng.uniform(0, p["BG_KILL_MAX"]))
                    time.sleep(delay)
                    self.eng.signal_kill()
                    t_rec = self._recover()
                    self._record(action="bg_kill", op=op, n=len(ids), delay=delay, recover_secs=round(t_rec, 2))
                self._check("s%d" % s)
                continue
            name = sorted(ops)[int(self.rng.integers(len(ops)))]
            if kill:
                delay = float(self.rng.uniform(0, p["KILL_DELAY_OP"].get(name, 2.0)))
                outcome, err = self._killed(ops[name], delay)
                t_rec = self._recover()
                self._record(action="kill_op", op=name, delay=delay, outcome=outcome, err=err,
                             recover_secs=round(t_rec, 2))
                self._check("s%d" % s)
                continue
            if name in self.eng.PRESERVING:
                before = self._check("s%d.pre" % s)
            # a second client writes while a non-preserving operation runs
            occ = (self._start_concurrent(self.p["OP_CONCURRENT"])
                   if self.p["OP_CONCURRENT"] and name not in self.eng.PRESERVING else None)
            t0 = time.time()
            try:
                ops[name]()
                err = None
            except (EngineUnavailable, EngineError) as e:
                err = repr(e)[:300]
                self.res["errors"].append(dict(step=s, op=name, err=err))
                if isinstance(e, EngineUnavailable):
                    self._recover()
            if occ is not None:
                self._finish_concurrent(occ, "s%d:%s" % (s, name),
                                        replaced=name in getattr(self.eng, "STATE_REPLACING", ()))
            if name == "restart":
                self._post_restart_lookups("restart")
            self._record(action="op", op=name, secs=round(time.time() - t0, 2), err=err)
            after = self._check("s%d.post" % s, plan=before["plan"] if name in self.eng.PRESERVING else None)
            if name in self.eng.PRESERVING and err is None:
                self._compare("s%d:%s" % (s, name), before, after)
        self.step = p["STEPS"]
        self._check("final")
        self.res["times"]["total"] = round(time.time() - t_start, 2)
        self._candidates()
        return self.res

    def _candidates(self):
        seen = set()
        for v in self.res["violations"]:
            key = ("EI" if v["rule"] in ("existence", "score", "order", "uniqueness", "size", "malformed",
                                         "filter", "lookup", "exact_size", "exact_rank") else
                   "SE" if v["rule"] == "se_changed" else "WQ", v["rule"], v["step"])
            if key not in seen:
                seen.add(key)
                self.res["candidates"].append(dict(key=list(key), example=_j(v)))


def _hash_files():
    import glob
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    files = glob.glob(os.path.join(here, "*.py")) + glob.glob(os.path.join(here, "adapters", "*.py"))
    files.append(os.path.join(os.path.dirname(here), "experiments", "campaign.py"))   # the driver
    for f in sorted(files):
        out[os.path.relpath(f, here)] = hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
    return out


# Taken once, when this module is imported, so the hashes describe the code the process
# runs even if the files change on disk while a batch is running.
_CODE_AT_IMPORT = _hash_files()


def _code_hashes():
    """SHA-256 prefixes of the generator, oracles and adapters as loaded by this process."""
    return dict(_CODE_AT_IMPORT)


def _j(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
