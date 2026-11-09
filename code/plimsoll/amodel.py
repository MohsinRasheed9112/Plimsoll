"""Logical model with one integer attribute per item, for filtered workloads.

Same semantics as plimsoll/model.py (possible states per item, indeterminate writes
resolved by lookup), with an outcome being None (absent) or a pair (vector, attribute).
It offers the interface the oracles use (status, vec, possible_vectors, in_exists, ...)
plus the attribute view (attr, possible_attrs) for the filter rule.
"""
import numpy as np

ABSENT_NEVER, LIVE, DELETED, INDET = 0, 1, 2, 3


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return int(a[1]) == int(b[1]) and np.array_equal(np.asarray(a[0], np.float32), np.asarray(b[0], np.float32))


class AModel:
    def __init__(self, cap, dim):
        self.cap, self.dim = cap, dim
        self.status = np.zeros(cap, np.int8)
        self.vec = np.zeros((cap, dim), np.float32)   # live vector, or last vector if deleted
        self.attr = np.zeros(cap, np.int64)            # live attribute, or last attribute
        self.indet = {}                                # id -> list of outcomes
        self.prior = {}

    def _outcome(self, i):
        s = self.status[i]
        if s == LIVE:
            return [(self.vec[i].copy(), int(self.attr[i]))]
        if s == INDET:
            return list(self.indet[i])
        return [None]

    def _apply(self, op, i, v, a):
        """Outcome of applying op to item i in a state where it has outcome cur."""
        def f(cur):
            if op == "delete":
                return None
            if op == "insert":
                return (np.asarray(v, np.float32).copy(), int(a))
            if op == "update":       # vector update keeps the attribute; absent stays absent
                return None if cur is None else (np.asarray(v, np.float32).copy(), cur[1])
            if op == "set_attr":     # attribute update keeps the vector; absent stays absent
                return None if cur is None else (cur[0], int(a))
            raise ValueError(op)
        return f

    def ack(self, op, ids, vecs=None, attrs=None):
        for j, i in enumerate(ids):
            i = int(i)
            if self.status[i] == INDET:
                raise RuntimeError("acknowledged write on an unresolved item %d" % i)
            cur = self._outcome(i)[0]
            new = self._apply(op, i, None if vecs is None else vecs[j], None if attrs is None else attrs[j])(cur)
            self._set(i, new, deleted_if_absent=(op == "delete") or self.status[i] == DELETED)

    def _set(self, i, out, deleted_if_absent):
        if out is None:
            if self.status[i] == LIVE or deleted_if_absent:
                self.status[i] = DELETED
        else:
            self.status[i] = LIVE
            self.vec[i] = out[0]
            self.attr[i] = out[1]

    def unknown(self, op, ids, vecs=None, attrs=None):
        for j, i in enumerate(ids):
            i = int(i)
            outs = self._outcome(i)
            fn = self._apply(op, i, None if vecs is None else vecs[j], None if attrs is None else attrs[j])
            new = [fn(o) for o in outs]
            for n in new:
                if not any(_same(n, o) for o in outs):
                    outs.append(n)
            if len(outs) == 1:
                continue
            if self.status[i] != INDET:
                self.prior[i] = int(self.status[i])
            self.status[i] = INDET
            self.indet[i] = outs

    def resolve(self, i, observed):
        """observed: None (absent) or (vector, attribute) from a lookup."""
        outs = self.indet.pop(i)
        prior = self.prior.pop(i)
        ok = any(_same(observed, o) for o in outs)
        if observed is None:
            self.status[i] = ABSENT_NEVER if prior == ABSENT_NEVER else DELETED
        else:
            self.status[i] = LIVE
            self.vec[i] = observed[0]
            self.attr[i] = observed[1]
        return ok

    def settle(self, i, outcome):
        """An indeterminate item whose write was acknowledged: fix it to that outcome
        (the in-flight write took effect and nothing else wrote the item)."""
        outs = self.indet.pop(i)
        self.prior.pop(i)
        if not any(_same(outcome, o) for o in outs):
            raise RuntimeError("settling item %d to an outcome it could not have" % i)
        if outcome is None:
            self.status[i] = DELETED
        else:
            self.status[i] = LIVE
            self.vec[i] = outcome[0]
            self.attr[i] = outcome[1]

    # ---------------------------------------------------------------- views
    def possible_vectors(self, i):
        s = self.status[i]
        if s == LIVE:
            return [self.vec[i]]
        if s == INDET:
            return [o[0] for o in self.indet[i] if o is not None]
        return []

    def possible_attrs(self, i):
        s = self.status[i]
        if s == LIVE:
            return [int(self.attr[i])]
        if s == INDET:
            return [o[1] for o in self.indet[i] if o is not None]
        return []

    def in_exists(self, i):
        s = self.status[i]
        return s == LIVE or (s == INDET and any(o is not None for o in self.indet[i]))

    def in_forall(self, i):
        s = self.status[i]
        return s == LIVE or (s == INDET and all(o is not None for o in self.indet[i]))

    def definite_live_ids(self):
        return np.flatnonzero(self.status == LIVE)

    def deleted_ids(self):
        return np.flatnonzero(self.status == DELETED)

    def n_forall(self):
        return int(np.count_nonzero(self.status == LIVE)) + sum(
            1 for o in self.indet.values() if all(x is not None for x in o))

    def n_exists(self):
        return int(np.count_nonzero(self.status == LIVE)) + sum(
            1 for o in self.indet.values() if any(x is not None for x in o))

    def has_indeterminate(self):
        return bool(self.indet)


def matches(filt, a):
    """Filters: None (no filter), ("eq", c), ("lt", c)."""
    if filt is None:
        return True
    kind, c = filt
    if kind == "eq":
        return int(a) == int(c)
    if kind == "lt":
        return int(a) < int(c)
    raise ValueError(kind)


def ei_check_filter(model, ids_lists, filt, where):
    """Filter rule: every returned item satisfies the filter in some possible state."""
    viols = []
    if filt is None:
        return viols
    for qi, ids in enumerate(ids_lists):
        for i in ids:
            i = int(i)
            if i < 0:
                continue
            if model.in_exists(i) and not any(matches(filt, a) for a in model.possible_attrs(i)):
                viols.append(dict(rule="filter", where=where, query=qi, id=i,
                                  attrs=[int(a) for a in model.possible_attrs(i)], filt=list(filt)))
    return viols
