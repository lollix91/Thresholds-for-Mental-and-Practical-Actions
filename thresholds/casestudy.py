"""The security-operations-center case study of the paper, and its replay."""
from __future__ import annotations

from typing import List, Tuple

from .policy import (Instance, Policy, Rev, decide_phys, initial_policy, propose,
                     validate, is_safe)

VL, LO, M, H, VH = range(5)
NAMES = ["vl", "l", "m", "h", "vh"]


def soc_instance() -> Instance:
    phys = ["block_ip", "isolate_host"]
    contents = ["p"]
    mental = ["rho1", "rho2", "rho3", "rho4", "rho5"]
    revs = [
        Rev("rho1", "thr", "A:block_ip", +1),
        Rev("rho2", "thr", "S:p", +1),
        Rev("rho3", "thr", "S:p", -1),
        Rev("rho4", "thr", "A:rho3", -1),
        Rev("rho5", "crit", "isolate_host", -1),
    ]
    agents = ["a", "r"]
    trust = {("a", "rho1"): M, ("a", "rho2"): M, ("a", "rho3"): M,
             ("a", "rho4"): H, ("a", "rho5"): M,
             ("r", "rho1"): VH, ("r", "rho2"): VH, ("r", "rho3"): VH,
             ("r", "rho4"): VH, ("r", "rho5"): VH}
    # (B, E, A) of each revision action; U and R are set equal to E.
    bea = {"rho1": (VL, LO, LO), "rho2": (VL, LO, LO), "rho3": (LO, M, H),
           "rho4": (M, H, VH), "rho5": (M, H, VH)}
    theta = {"A:block_ip": M, "B:block_ip": LO, "I:block_ip": M, "C:block_ip": M, "L:block_ip": H,
             "A:isolate_host": VH, "B:isolate_host": LO, "I:isolate_host": M,
             "C:isolate_host": M, "L:isolate_host": H, "S:p": H}
    for rho, (b, e, a) in bea.items():
        theta.update({f"B:{rho}": b, f"E:{rho}": e, f"A:{rho}": a, f"U:{rho}": e, f"R:{rho}": e})
    crit = frozenset({"isolate_host", "rho3", "rho4", "rho5"})
    return Instance(
        L=5, phys=phys, mental=mental, contents=contents, revs=revs, agents=agents,
        trust=trust, theta0=theta, floor={"A:rho3": H}, crit0=crit, floor_crit=crit,
        trust_phys={("a", "block_ip"): M, ("r", "block_ip"): VH})


TRACES = frozenset({"rho3", "rho4", "rho5"})


def replay() -> List[Tuple[str, str, str, str]]:
    """Replay the six steps of the case study; returns (step, action, outcome, detail)."""
    inst = soc_instance()
    pol = initial_policy(inst)
    assert is_safe(inst, pol)
    rows: List[Tuple[str, str, str, str]] = []

    def phys_row(tag):
        da = decide_phys(inst, pol, "a", "block_ip")
        dr = decide_phys(inst, pol, "r", "block_ip")
        rows.append((tag, "block_ip decision", f"a:{da} r:{dr}", ""))

    phys_row("0p")
    res = propose(inst, pol, "a", "rho1", TRACES)
    pol = res.policy
    rows.append(("1", "rho1 by a", res.outcome, f"tauA(block_ip)={NAMES[pol.theta['A:block_ip']]}"))
    phys_row("1p")
    res = propose(inst, pol, "a", "rho2", TRACES)
    pol = res.policy
    rows.append(("2", "rho2 by a", res.outcome, f"tauS(p)={NAMES[pol.theta['S:p']]}"))
    res = propose(inst, pol, "a", "rho3", TRACES)
    rows.append(("3", "rho3 by a", res.outcome, res.detail))
    res = validate(inst, pol, "r", "rho3", TRACES)
    pol = res.policy
    rows.append(("4", "rho3 validated by r", res.outcome, f"tauS(p)={NAMES[pol.theta['S:p']]}"))
    res = propose(inst, pol, "a", "rho4", TRACES)
    rows.append(("5a", "rho4 by a", res.outcome, res.detail))
    res = validate(inst, pol, "r", "rho4", TRACES)
    rows.append(("5b", "rho4 validated by r", res.outcome, res.detail))
    res = propose(inst, pol, "r", "rho5", TRACES)
    rows.append(("6", "rho5 by r", res.outcome, res.detail))
    return rows


if __name__ == "__main__":
    for row in replay():
        print(" | ".join(row))
