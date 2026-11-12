"""Compiled explicit-state exploration of the policy transition system.

Only the entries targeted by some revision action and the targeted members of
the critical set can change. A search state is therefore the tuple
``(v_0, ..., v_{n-1}, mask)`` with the values of the targeted entries and a
bitmask over the targeted critical-set members; every other entry is a
constant folded into per-position bounds.
"""
from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .policy import Instance, initial_policy, is_safe, Policy

State = Tuple[int, ...]


@dataclass
class CRev:
    name: str
    kind: int  # 0 threshold revision, 1 critical-set revision
    pos: int  # state position (kind 0) or mask bit (kind 1)
    d: int
    aref: int  # position of tau_A(rho) or -1 if constant
    aconst: int
    maxtrust: int


@dataclass
class Problem:
    L: int
    names: List[str]  # names of the targeted entries (positions)
    cnames: List[str]  # names of the targeted critical-set members (bits)
    init: State
    revs: List[CRev]
    lo1: List[int]
    hi1: List[int]
    pairs: List[List[Tuple[int, int, int]]]  # incident[p]: (q, strict, role) role 0: p left, 1: p right
    floor: List[int]
    ceil: List[int]
    floor_mask: int
    npos: int

    @property
    def space_size(self) -> int:
        return (self.L ** self.npos) * (1 << len(self.cnames))


def compile_instance(inst: Instance) -> Problem:
    pol0 = initial_policy(inst)
    assert is_safe(inst, pol0), "initial policy must be safe"
    targets: List[str] = []
    for r in inst.revs:
        if r.kind == "thr" and r.target not in targets:
            targets.append(r.target)
    pos = {e: i for i, e in enumerate(targets)}
    cnames: List[str] = []
    for r in inst.revs:
        if r.kind == "crit" and r.target not in cnames:
            cnames.append(r.target)
    bit = {z: i for i, z in enumerate(cnames)}
    n = len(targets)
    lo1 = [0] * n
    hi1 = [inst.L - 1] * n
    pairs: List[List[Tuple[int, int, int]]] = [[] for _ in range(n)]
    for e1, e2, s in inst.constraints():
        p1, p2 = pos.get(e1), pos.get(e2)
        if p1 is not None and p2 is not None:
            pairs[p1].append((p2, s, 0))
            pairs[p2].append((p1, s, 1))
        elif p1 is not None:
            hi1[p1] = min(hi1[p1], inst.theta0[e2] - s)
        elif p2 is not None:
            lo1[p2] = max(lo1[p2], inst.theta0[e1] + s)
    floor = [inst.floor.get(e, 0) for e in targets]
    ceil = [inst.ceiling(e) for e in targets]
    crevs = []
    for r in inst.revs:
        ae = f"A:{r.name}"
        aref = pos.get(ae, -1)
        crevs.append(CRev(r.name, 0 if r.kind == "thr" else 1,
                          pos[r.target] if r.kind == "thr" else bit[r.target],
                          r.d, aref, inst.theta0[ae], inst.maxtrust(r.name)))
    init_mask = sum(1 << bit[z] for z in cnames if z in pol0.crit)
    floor_mask = sum(1 << bit[z] for z in cnames if z in inst.floor_crit)
    init = tuple(inst.theta0[e] for e in targets) + (init_mask,)
    return Problem(inst.L, targets, cnames, init, crevs, lo1, hi1, pairs,
                   floor, ceil, floor_mask, n)


def successors(P: Problem, s: State, g1: bool = True, g2: bool = True, g3: bool = True):
    """Yield (revision name, successor state) under the selected guards."""
    vals, mask = s[:-1], s[-1]
    for r in P.revs:
        ta = vals[r.aref] if r.aref >= 0 else r.aconst
        if ta > r.maxtrust:
            continue
        if r.kind == 0:
            p = r.pos
            nv = vals[p] + r.d
            if nv < 0 or nv >= P.L:
                continue
            if g1:
                if nv < P.lo1[p] or nv > P.hi1[p]:
                    continue
                bad = False
                for q, st, role in P.pairs[p]:
                    if (role == 0 and nv + st > vals[q]) or (role == 1 and vals[q] + st > nv):
                        bad = True
                        break
                if bad:
                    continue
            if g2 and nv < P.floor[p]:
                continue
            if g3 and nv > P.ceil[p]:
                continue
            yield r.name, vals[:p] + (nv,) + vals[p + 1:] + (mask,)
        else:
            b = 1 << r.pos
            if r.d > 0:
                if mask & b:
                    continue
                nm = mask | b
            else:
                if not mask & b:
                    continue
                nm = mask & ~b
            if g2 and (P.floor_mask & ~nm):
                continue
            yield r.name, vals + (nm,)


def violations(P: Problem, s: State) -> Set[str]:
    vals, mask = s[:-1], s[-1]
    v: Set[str] = set()
    for p in range(P.npos):
        x = vals[p]
        if x < P.lo1[p] or x > P.hi1[p]:
            v.add("G1")
        if x < P.floor[p]:
            v.add("G2")
        if x > P.ceil[p]:
            v.add("G3")
        for q, st, role in P.pairs[p]:
            if role == 0 and x + st > vals[q]:
                v.add("G1")
    if P.floor_mask & ~mask:
        v.add("G2")
    return v


@dataclass
class ReachResult:
    depth: Dict[State, int]
    parent: Dict[State, Tuple[Optional[State], Optional[str]]]
    seconds: float
    first_bad: Dict[str, Tuple[int, List[str]]]  # kind -> (depth, path of revision names)
    truncated: bool

    @property
    def size(self) -> int:
        return len(self.depth)

    @property
    def max_depth(self) -> int:
        return max(self.depth.values())


def reach(P: Problem, g1=True, g2=True, g3=True, max_states: int = 3_000_000) -> ReachResult:
    t0 = time.perf_counter()
    depth = {P.init: 0}
    parent: Dict[State, Tuple[Optional[State], Optional[str]]] = {P.init: (None, None)}
    first_bad: Dict[str, Tuple[int, List[str]]] = {}

    def record(s):
        for k in violations(P, s):
            if k not in first_bad:
                path, cur = [], s
                while parent[cur][0] is not None:
                    path.append(parent[cur][1])
                    cur = parent[cur][0]
                first_bad[k] = (depth[s], path[::-1])

    record(P.init)
    q = deque([P.init])
    truncated = False
    while q:
        s = q.popleft()
        for name, t in successors(P, s, g1, g2, g3):
            if t not in depth:
                if len(depth) >= max_states:
                    truncated = True
                    q.clear()
                    break
                depth[t] = depth[s] + 1
                parent[t] = (s, name)
                record(t)
                q.append(t)
    return ReachResult(depth, parent, time.perf_counter() - t0, first_bad, truncated)


def all_states(P: Problem) -> Iterable[State]:
    for vals in itertools.product(range(P.L), repeat=P.npos):
        for m in range(1 << len(P.cnames)):
            yield vals + (m,)


def safe_set(P: Problem) -> Set[State]:
    return {s for s in all_states(P) if not violations(P, s)}


def gfp(P: Problem, S: Set[State], g1=True, g2=True, g3=True) -> Set[State]:
    """Greatest fixed point of Gamma_S(X) = {s in S | Succ(s) subset of X}."""
    X = set(S)
    changed = True
    while changed:
        changed = False
        for s in list(X):
            if any(t not in X for _, t in successors(P, s, g1, g2, g3)):
                X.discard(s)
                changed = True
    return X


def reach_cooldown(P: Problem, delta: int, g1=True, g2=True, g3=True, max_states: int = 2_000_000):
    """Exploration with a cooldown of ``delta`` ticks between two revisions of the same entry.

    The state is extended with one age per targeted entry and per targeted critical-set member. A
    revision resets the age of what it changes to 0 and requires it to be at least ``delta``; a tick
    (cost 1) increments every age, saturating at ``delta``. Revisions cost 0, so a 0-1 BFS gives, for
    every policy, the minimum number of ticks needed to reach it. Returns a dict policy -> min ticks.
    """
    nb = len(P.cnames)
    n = P.npos
    start = (P.init, tuple([delta] * (n + nb)))
    dist = {start: 0}
    q = deque([start])
    best: Dict[State, int] = {P.init: 0}
    while q:
        s, ages = q.popleft()
        d = dist[(s, ages)]
        if d > dist.get((s, ages), d):
            continue
        # revisions (cost 0) go to the front of the queue
        for _, t in successors(P, s, g1, g2, g3):
            idx = next((p for p in range(n) if t[p] != s[p]), None)
            if idx is None:
                idx = n + next(b for b in range(nb) if (t[-1] ^ s[-1]) >> b & 1)
            if ages[idx] < delta:
                continue
            na = ages[:idx] + (0,) + ages[idx + 1:]
            key = (t, na)
            if key not in dist or dist[key] > d:
                if len(dist) >= max_states:
                    return best
                dist[key] = d
                q.appendleft(key)
                if best.get(t, 1 << 30) > d:
                    best[t] = d
        # tick (cost 1) goes to the back
        na = tuple(min(a + 1, delta) for a in ages)
        key = (s, na)
        if key not in dist or dist[key] > d + 1:
            dist[key] = d + 1
            q.append(key)
    return best
