"""Reference (dictionary based) semantics of revisable trust policies.

This module is a direct, unoptimised transcription of the definitions of the
paper: policies, revision actions, guards (G1)-(G4), decisions and the
proposal/review/validation protocol. It is used as an oracle for the compiled
search code in :mod:`thresholds.search` and to replay the case study.

Levels are integers ``0..L-1`` (``0`` is the minimal trust level).
Entries are strings ``"x:name"``: for physical actions ``x`` in I,C,L,B,A,
for mental actions ``x`` in E,U,R,B,A, and ``"S:content"`` for sources.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

ENTRY_P = ("I", "C", "L", "B", "A")
ENTRY_M = ("E", "U", "R", "B", "A")


@dataclass(frozen=True)
class Rev:
    """A revision action: atomic name, target and direction.

    ``kind`` is ``"thr"`` (target is an entry) or ``"crit"`` (target is an
    action name). ``d`` is +1 for tightening (raise / promote) and -1 for
    loosening (lower / demote).
    """

    name: str
    kind: str
    target: str
    d: int


@dataclass
class Instance:
    L: int
    phys: List[str]
    mental: List[str]  # includes the revision action names
    contents: List[str]
    revs: List[Rev]
    agents: List[str]
    trust: Dict[Tuple[str, str], int]  # (agent, revision name) -> level
    theta0: Dict[str, int]
    floor: Dict[str, int] = field(default_factory=dict)
    crit0: FrozenSet[str] = frozenset()
    floor_crit: FrozenSet[str] = frozenset()
    trust_phys: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # ---- structure -------------------------------------------------
    def entries(self) -> List[str]:
        out = [f"{x}:{a}" for a in self.phys for x in ENTRY_P]
        out += [f"{x}:{m}" for m in self.mental for x in ENTRY_M]
        out += [f"S:{c}" for c in self.contents]
        return out

    def rev(self, name: str) -> Rev:
        return next(r for r in self.revs if r.name == name)

    def maxtrust(self, name: str) -> int:
        return max(self.trust[(j, name)] for j in self.agents)

    def constraints(self) -> List[Tuple[str, str, int]]:
        """Well-formedness (G1): triples (e1, e2, s) meaning theta(e1)+s <= theta(e2)."""
        cs: List[Tuple[str, str, int]] = []
        for a in self.phys:
            cs.append((f"B:{a}", f"A:{a}", 1))
        for m in self.mental:
            cs += [(f"B:{m}", f"E:{m}", 1), (f"E:{m}", f"A:{m}", 0),
                   (f"U:{m}", f"A:{m}", 0), (f"R:{m}", f"A:{m}", 0)]
        for ru in self.revs:
            for rd in self.revs:
                if ru.kind == rd.kind == "thr" and ru.target == rd.target and ru.d > 0 > rd.d:
                    cs.append((f"A:{ru.name}", f"A:{rd.name}", 0))
        return cs

    def ceiling(self, entry: str) -> int:
        """Non-lockout (G3): upper bound on tau_A of a revision action."""
        if entry.startswith("A:"):
            n = entry[2:]
            if any(r.name == n for r in self.revs):
                return self.maxtrust(n)
        return self.L - 1


@dataclass
class Policy:
    theta: Dict[str, int]
    crit: FrozenSet[str]

    def copy(self) -> "Policy":
        return Policy(dict(self.theta), self.crit)

    def key(self):
        return (tuple(sorted(self.theta.items())), tuple(sorted(self.crit)))


def initial_policy(inst: Instance) -> Policy:
    return Policy(dict(inst.theta0), frozenset(inst.crit0))


# ---- guards ----------------------------------------------------------
def violations(inst: Instance, pol: Policy) -> Dict[str, List[str]]:
    v: Dict[str, List[str]] = {"G1": [], "G2": [], "G3": []}
    for e1, e2, s in inst.constraints():
        if pol.theta[e1] + s > pol.theta[e2]:
            v["G1"].append(f"{e1}+{s}<={e2}")
    for e in inst.entries():
        if not 0 <= pol.theta[e] < inst.L:
            v["G1"].append(f"range:{e}")
        if pol.theta[e] < inst.floor.get(e, 0):
            v["G2"].append(f"floor:{e}")
        if pol.theta[e] > inst.ceiling(e):
            v["G3"].append(f"ceil:{e}")
    for z in inst.floor_crit:
        if z not in pol.crit:
            v["G2"].append(f"floorcrit:{z}")
    return v


def is_safe(inst: Instance, pol: Policy, guards: Iterable[str] = ("G1", "G2", "G3")) -> bool:
    v = violations(inst, pol)
    return not any(v[g] for g in guards)


# ---- revision --------------------------------------------------------
def apply_rev(inst: Instance, pol: Policy, rev: Rev) -> Optional[Policy]:
    """rev(pi, rho); None if undefined (out of range / no-op on the critical set)."""
    new = pol.copy()
    if rev.kind == "thr":
        nv = pol.theta[rev.target] + rev.d
        if not 0 <= nv < inst.L:
            return None
        new.theta[rev.target] = nv
    else:
        if rev.d > 0:
            if rev.target in pol.crit:
                return None
            new.crit = pol.crit | {rev.target}
        else:
            if rev.target not in pol.crit:
                return None
            new.crit = pol.crit - {rev.target}
    return new


def authorised(inst: Instance, pol: Policy, rev: Rev) -> bool:
    """Some agent reaches the autonomy threshold of the revision action."""
    return pol.theta[f"A:{rev.name}"] <= inst.maxtrust(rev.name)


def ref_successors(inst: Instance, pol: Policy, guards: Iterable[str] = ("G1", "G2", "G3")) -> List[Tuple[str, Policy]]:
    out = []
    for r in inst.revs:
        if not authorised(inst, pol, r):
            continue
        new = apply_rev(inst, pol, r)
        if new is None or not is_safe(inst, new, guards):
            continue
        out.append((r.name, new))
    return out


# ---- decisions and the proposal/review protocol ------------------------
def decide(inst: Instance, pol: Policy, agent: str, rev_name: str) -> str:
    """allow / review / block / inert for a proposal by ``agent``."""
    t = inst.trust[(agent, rev_name)]
    b, e, a = (pol.theta[f"{x}:{rev_name}"] for x in ("B", "E", "A"))
    if t <= b:
        return "block"
    if t >= a:
        return "allow"
    return "review" if t >= e else "inert"


def decide_phys(inst: Instance, pol: Policy, agent: str, action: str) -> str:
    t = inst.trust_phys[(agent, action)]
    b, a = pol.theta[f"B:{action}"], pol.theta[f"A:{action}"]
    if t >= a:
        return "allow"
    return "block" if t <= b else "delegate"


@dataclass
class StepResult:
    outcome: str  # applied / review / blocked / inert / rejected
    policy: Policy
    detail: str = ""


def _try_apply(inst, pol, rev, traces, guards):
    new = apply_rev(inst, pol, rev)
    if new is None:
        return StepResult("inert", pol, "revision undefined")
    failed = [g for g in ("G1", "G2", "G3") if g in guards and violations(inst, new)[g]]
    if "G4" in guards and rev.name in pol.crit and rev.name not in traces:
        failed.append("G4")
    if failed:
        return StepResult("rejected", pol, "guards failed: " + ",".join(failed))
    return StepResult("applied", new, "")


def propose(inst: Instance, pol: Policy, agent: str, rev_name: str,
            traces: FrozenSet[str] = frozenset(),
            guards: Iterable[str] = ("G1", "G2", "G3", "G4")) -> StepResult:
    rev = inst.rev(rev_name)
    d = decide(inst, pol, agent, rev_name)
    if d == "allow":
        return _try_apply(inst, pol, rev, traces, guards)
    if d == "review":
        return StepResult("review", pol, "pending")
    return StepResult("blocked" if d == "block" else "inert", pol, d)


def validate(inst: Instance, pol: Policy, reviewer: str, rev_name: str,
             traces: FrozenSet[str] = frozenset(),
             guards: Iterable[str] = ("G1", "G2", "G3", "G4")) -> StepResult:
    """Validation of a pending proposal; guards are re-evaluated on the current policy."""
    rev = inst.rev(rev_name)
    if inst.trust[(reviewer, rev_name)] < pol.theta[f"A:{rev_name}"]:
        return StepResult("rejected", pol, "reviewer not trusted enough")
    return _try_apply(inst, pol, rev, traces, guards)
