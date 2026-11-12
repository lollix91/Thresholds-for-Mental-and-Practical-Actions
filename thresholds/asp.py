"""Answer Set Programming encoding of bounded policy reachability (clingo).

The encoding mirrors :func:`thresholds.search.successors`: at each time step at
most one revision action is applied, subject to the selected guards, and the
query asks whether a policy violating a given guard is reached within ``H`` steps.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import clingo

from .search import Problem

RULES = r"""
time(0..horizon).
level(0..nlev-1).
val(P,V,0) :- init(P,V).
bit(B,0)   :- initbit(B).

{ apply(R,T) : rev(R) } 1 :- time(T), T < horizon.

% applicability: some agent reaches tau_A of the revision action
:- apply(R,T), aref(R,P), val(P,V,T), maxtrust(R,M), V > M.
:- apply(R,T), aconst(R,V), maxtrust(R,M), V > M.

% effects on thresholds
changed(P,T) :- apply(R,T), revthr(R,P,_).
val(P,V+D,T+1) :- apply(R,T), revthr(R,P,D), val(P,V,T).
val(P,V,T+1)   :- val(P,V,T), not changed(P,T), T < horizon.
:- val(P,V,T), not level(V).

% effects on the critical set
cchanged(B,T) :- apply(R,T), revcrit(R,B,_).
bit(B,T+1) :- apply(R,T), revcrit(R,B,1).
bit(B,T+1) :- bit(B,T), not cchanged(B,T), T < horizon.
:- apply(R,T), revcrit(R,B,1),  bit(B,T).
:- apply(R,T), revcrit(R,B,-1), not bit(B,T).

% guards
:- g1, val(P,V,T), lo1(P,X), V < X.
:- g1, val(P,V,T), hi1(P,X), V > X.
:- g1, pair(P,Q,S), val(P,V,T), val(Q,W,T), V + S > W.
:- g2, val(P,V,T), floor(P,X), V < X.
:- g2, floorbit(B), time(T), not bit(B,T).
:- g3, val(P,V,T), ceil(P,X), V > X.

% bad configurations
bad(g1,T) :- val(P,V,T), lo1(P,X), V < X.
bad(g1,T) :- val(P,V,T), hi1(P,X), V > X.
bad(g1,T) :- pair(P,Q,S), val(P,V,T), val(Q,W,T), V + S > W.
bad(g2,T) :- val(P,V,T), floor(P,X), V < X.
bad(g2,T) :- floorbit(B), time(T), not bit(B,T).
bad(g3,T) :- val(P,V,T), ceil(P,X), V > X.
:- not bad(K,_), query(K).
"""


def facts(P: Problem) -> str:
    f = []
    for p in range(P.npos):
        f.append(f"init({p},{P.init[p]}). lo1({p},{P.lo1[p]}). hi1({p},{P.hi1[p]}). "
                 f"floor({p},{P.floor[p]}). ceil({p},{P.ceil[p]}).")
        for q, st, role in P.pairs[p]:
            if role == 0:
                f.append(f"pair({p},{q},{st}).")
    mask = P.init[-1]
    for b in range(len(P.cnames)):
        if mask >> b & 1:
            f.append(f"initbit({b}).")
        if P.floor_mask >> b & 1:
            f.append(f"floorbit({b}).")
    for i, r in enumerate(P.revs):
        f.append(f"rev(r{i}). maxtrust(r{i},{r.maxtrust}).")
        f.append(f"revthr(r{i},{r.pos},{r.d})." if r.kind == 0 else f"revcrit(r{i},{r.pos},{r.d}).")
        f.append(f"aref(r{i},{r.aref})." if r.aref >= 0 else f"aconst(r{i},{r.aconst}).")
    return "\n".join(f)


def bounded_reach(P: Problem, horizon: int, query: str, guards: Tuple[bool, bool, bool] = (True, True, True)) -> Tuple[bool, float]:
    """Is a policy violating guard ``query`` ('g1'|'g2'|'g3') reachable within ``horizon`` steps?"""
    prog = f"#const horizon={horizon}.\n#const nlev={P.L}.\n" + RULES + facts(P) + f"\nquery({query}).\n"
    for name, on in zip(("g1", "g2", "g3"), guards):
        if on:
            prog += f"{name}.\n"
    t0 = time.perf_counter()
    ctl = clingo.Control(["-n", "1", "--warn=none"], logger=lambda *_: None)
    ctl.add("base", [], prog)
    ctl.ground([("base", [])])
    res = ctl.solve()
    return res.satisfiable, time.perf_counter() - t0
