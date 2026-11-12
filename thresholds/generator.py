"""Random generation of revisable-policy instances (always with a safe initial policy)."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from .policy import ENTRY_M, ENTRY_P, Instance, Rev, initial_policy, is_safe


def generate(L: int = 5, k_thr: int = 4, k_crit: int = 1, n_phys: int = 3, n_src: int = 2,
             n_agents: int = 3, p_meta: float = 0.5, p_floor: float = 0.5,
             seed: int = 0, self_target: bool = True) -> Instance:
    """Build an instance.

    ``k_thr`` threshold targets, each with one tightening and one loosening
    revision action; ``k_crit`` critical-set targets, each with a promotion and a
    demotion. A fraction ``p_meta`` of the threshold targets are autonomy
    thresholds of revision actions themselves (meta-level, possibly self-referential).
    """
    rng = random.Random(seed)
    phys = [f"a{i}" for i in range(n_phys)]
    contents = [f"c{i}" for i in range(n_src)]
    base_m = ["m0", "m1"]
    n_rev = 2 * (k_thr + k_crit)
    rev_names = [f"rho{i}" for i in range(n_rev)]
    mental = base_m + rev_names
    agents = [f"j{i}" for i in range(n_agents)]
    trust = {(j, r): rng.randint(1, L - 1) for j in agents for r in rev_names}

    # choose targets
    thr_pool = [f"A:{a}" for a in phys] + [f"S:{c}" for c in contents] + [f"C:{a}" for a in phys]
    meta_pool = [f"A:{r}" for r in rev_names]
    targets: List[str] = []
    guard = 0
    while len(targets) < k_thr and guard < 1000:
        guard += 1
        pool = meta_pool if (rng.random() < p_meta and meta_pool) else thr_pool
        t = rng.choice(pool)
        if t not in targets:
            targets.append(t)
    crit_pool = phys + base_m
    crit_targets = rng.sample(crit_pool, min(k_crit, len(crit_pool)))

    revs: List[Rev] = []
    idx = 0
    for t in targets:
        revs.append(Rev(rev_names[idx], "thr", t, +1)); idx += 1
        revs.append(Rev(rev_names[idx], "thr", t, -1)); idx += 1
    for z in crit_targets:
        revs.append(Rev(rev_names[idx], "crit", z, +1)); idx += 1
        revs.append(Rev(rev_names[idx], "crit", z, -1)); idx += 1
    revs = revs[:n_rev]
    # unused revision names become plain mental actions
    used = {r.name for r in revs}

    maxt = {r: max(trust[(j, r)] for j in agents) for r in rev_names}
    theta: Dict[str, int] = {}
    for a in phys:
        A = rng.randint(1, L - 1)
        theta[f"A:{a}"] = A
        theta[f"B:{a}"] = rng.randint(0, A - 1)
        for x in ("I", "C", "L"):
            theta[f"{x}:{a}"] = rng.randint(0, L - 1)
    for m in mental:
        hi = maxt.get(m, L - 1)
        A = rng.randint(1, hi)
        theta[f"A:{m}"] = A
    # asymmetry: tau_A(rho_up) <= tau_A(rho_down) for the same target
    for ru in revs:
        for rd in revs:
            if ru.kind == rd.kind == "thr" and ru.target == rd.target and ru.d > 0 > rd.d:
                theta[f"A:{ru.name}"] = min(theta[f"A:{ru.name}"], theta[f"A:{rd.name}"])
    for m in mental:
        A = theta[f"A:{m}"]
        E = rng.randint(1, A)
        theta[f"E:{m}"] = E
        theta[f"B:{m}"] = rng.randint(0, E - 1)
        theta[f"U:{m}"] = rng.randint(0, A)
        theta[f"R:{m}"] = rng.randint(0, A)
    for c in contents:
        theta[f"S:{c}"] = rng.randint(0, L - 1)
    # targeted entries must start inside the bounds induced by neighbouring entries (already satisfied)

    floor: Dict[str, int] = {}
    for t in targets:
        if t.startswith("A:") and t[2:] in rev_names:
            floor[t] = theta[t] if rng.random() < p_floor else 0
        else:
            floor[t] = rng.randint(0, theta[t]) if rng.random() < p_floor else 0
    loose_crit = {r.target for r in revs if r.kind == "crit" and r.d < 0}
    floor_crit = frozenset(z for z in crit_targets if rng.random() < p_floor)
    crit0 = frozenset(floor_crit | {z for z in crit_targets if rng.random() < 0.5})

    inst = Instance(L=L, phys=phys, mental=mental, contents=contents, revs=revs,
                    agents=agents, trust=trust, theta0=theta, floor=floor,
                    crit0=crit0, floor_crit=floor_crit)
    assert is_safe(inst, initial_policy(inst)), "generator produced an unsafe initial policy"
    return inst
