"""Logical model of what the database may contain (paper Sec. III-A, possible states).

Each item has either one definite outcome (absent, or live with one vector) or, while
it has indeterminate writes, a list of possible outcomes. An outcome is None (absent)
or a float32 vector (live). The model stores outcomes per item, which ignores
correlations between items (for example, the atomicity of a batch) and so
over-approximates the set of possible states.

The reference engine is linearizable with immediate visibility, and the harness has a
single client, so indeterminacy arises only from writes whose outcome a kill hid.
Those writes cannot take effect after the restart that ends them; the harness resolves
the affected items by lookup right after that restart, before any other read.
"""
import numpy as np

ABSENT_NEVER, LIVE, DELETED, INDET = 0, 1, 2, 3


class Model:
    def __init__(self, cap, dim):
        self.cap = cap
        self.dim = dim
        self.status = np.zeros(cap, np.int8)
        self.vec = np.zeros((cap, dim), np.float32)   # live vector, or last vector if deleted
        self.indet = {}                               # id -> list of outcomes (None or vector)
        self.prior = {}                               # id -> status before it became indeterminate

    # ----------------------------------------------------------------- writes
    def _outcome(self, i):
        s = self.status[i]
        if s == LIVE:
            return [self.vec[i].copy()]
        if s == INDET:
            return list(self.indet[i])
        return [None]

    def ack(self, op, ids, vecs=None):
        """Acknowledged write: the outcome is known."""
        for j, i in enumerate(ids):
            if self.status[i] == INDET:
                raise RuntimeError("acknowledged write on an unresolved item %d" % i)
            if op in ("insert", "update"):
                self.status[i] = LIVE
                self.vec[i] = vecs[j]
            elif op == "delete":
                self.status[i] = DELETED              # keeps the last vector
            else:
                raise ValueError(op)

    def unknown(self, op, ids, vecs=None):
        """Write with an unknown outcome: it may or may not have taken effect."""
        for j, i in enumerate(ids):
            outs = self._outcome(i)
            new = vecs[j].copy() if op in ("insert", "update") else None   # a delete keeps self.vec
            if not any(_same(new, o) for o in outs):
                outs.append(new)
            if len(outs) == 1:
                continue
            if self.status[i] != INDET:
                self.prior[i] = int(self.status[i])
            self.status[i] = INDET
            self.indet[i] = outs

    def resolve(self, i, observed):
        """Fix an indeterminate item to the outcome a lookup observed.

        Returns True if the observation is one of the possible outcomes."""
        outs = self.indet.pop(i)
        prior = self.prior.pop(i)
        ok = any(_same(observed, o) for o in outs)
        if observed is None:
            # absent now; it keeps a last vector only if it was ever stored
            self.status[i] = ABSENT_NEVER if prior == ABSENT_NEVER else DELETED
        else:
            self.status[i] = LIVE
            self.vec[i] = observed
        return ok

    # ---------------------------------------------------------------- queries
    def possible_vectors(self, i):
        s = self.status[i]
        if s == LIVE:
            return [self.vec[i]]
        if s == INDET:
            return [o for o in self.indet[i] if o is not None]
        return []

    def in_exists(self, i):
        s = self.status[i]
        return s == LIVE or (s == INDET and any(o is not None for o in self.indet[i]))

    def in_forall(self, i):
        s = self.status[i]
        return s == LIVE or (s == INDET and all(o is not None for o in self.indet[i]))

    def absent_everywhere(self, i):
        return not self.in_exists(i)

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


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
