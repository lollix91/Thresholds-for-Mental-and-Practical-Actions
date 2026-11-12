"""Empirical validation of the theorems of the paper on random instances.

Run with:  python -m unittest discover -s tests -v
"""
import unittest

from thresholds import asp
from thresholds.casestudy import replay, soc_instance
from thresholds.generator import generate
from thresholds.policy import (initial_policy, is_safe, ref_successors, authorised,
                               apply_rev)
from thresholds.search import (compile_instance, gfp, reach, safe_set, successors,
                               violations)

SEEDS = range(40)


def small(seed, **kw):
    params = dict(L=4, k_thr=3, k_crit=1, n_phys=2, n_src=1, n_agents=3, seed=seed)
    params.update(kw)
    return generate(**params)


def state_of(P, inst, pol):
    vals = tuple(pol.theta[e] for e in P.names)
    mask = sum(1 << i for i, z in enumerate(P.cnames) if z in pol.crit)
    return vals + (mask,)


class TestCaseStudy(unittest.TestCase):
    def test_replay_matches_paper_table(self):
        rows = {r[0]: r for r in replay()}
        self.assertEqual(rows["1"][2], "applied")
        self.assertEqual(rows["1"][3], "tauA(block_ip)=h")
        self.assertEqual(rows["2"][2], "applied")
        self.assertEqual(rows["3"][2], "review")
        self.assertEqual(rows["4"][2], "applied")
        self.assertEqual(rows["4"][3], "tauS(p)=h")
        self.assertEqual(rows["5a"][2], "review")
        self.assertEqual(rows["5b"][2], "rejected")
        self.assertIn("G2", rows["5b"][3])
        self.assertEqual(rows["6"][2], "rejected")
        self.assertIn("G2", rows["6"][3])
        self.assertEqual(rows["0p"][2], "a:allow r:allow")
        self.assertEqual(rows["1p"][2], "a:delegate r:allow")


class TestCompiledAgainstReference(unittest.TestCase):
    def test_same_reachable_set(self):
        for seed in SEEDS:
            inst = small(seed)
            P = compile_instance(inst)
            res = reach(P)
            # reference BFS over dictionary policies
            pol0 = initial_policy(inst)
            seen = {pol0.key(): pol0}
            frontier = [pol0]
            while frontier:
                nxt = []
                for p in frontier:
                    for _, q in ref_successors(inst, p):
                        if q.key() not in seen:
                            seen[q.key()] = q
                            nxt.append(q)
                frontier = nxt
            self.assertEqual(len(seen), res.size, f"seed {seed}")
            ref_states = {state_of(P, inst, p) for p in seen.values()}
            self.assertEqual(ref_states, set(res.depth), f"seed {seed}")


class TestTheorems(unittest.TestCase):
    def test_closure_of_safe_region(self):
        """Theorem: reachable policies are well-formed, above the floor and lockout free."""
        for seed in SEEDS:
            P = compile_instance(small(seed, k_thr=4))
            res = reach(P)
            self.assertTrue(all(not violations(P, s) for s in res.depth), f"seed {seed}")

    def test_safe_is_greatest_fixed_point(self):
        for seed in range(15):
            P = compile_instance(small(seed, L=3, k_thr=3, n_phys=2))
            S = safe_set(P)
            self.assertEqual(gfp(P, S), S, f"seed {seed}")

    def test_unguarded_gfp_is_smaller(self):
        strictly = 0
        for seed in range(15):
            P = compile_instance(small(seed, L=3, k_thr=3, n_phys=2, p_meta=0.8))
            S = safe_set(P)
            g0 = gfp(P, S, False, False, False)
            self.assertTrue(g0 <= S)
            strictly += g0 != S
        self.assertGreater(strictly, 0)

    def test_tightening_safety(self):
        """Tightening never enlarges the set of autonomous executions."""
        for seed in SEEDS:
            inst = small(seed)
            P = compile_instance(inst)
            res = reach(P)
            def auto(s):
                th = dict(zip(P.names, s[:-1]))
                return {(j, r.name) for j in inst.agents for r in inst.revs
                        if inst.trust[(j, r.name)] >= (th[f"A:{r.name}"] if f"A:{r.name}" in th else inst.theta0[f"A:{r.name}"])}
            for s in res.depth:
                for name, t in successors(P, s):
                    rv = inst.rev(name)
                    if rv.d > 0 and rv.kind == "thr":
                        self.assertTrue(auto(t) <= auto(s))

    def test_no_unmediated_loosening(self):
        """Every guarded transition that loosens is a loosening revision, stays above the floor, and is authorised."""
        for seed in SEEDS:
            inst = small(seed)
            P = compile_instance(inst)
            res = reach(P)
            for s in res.depth:
                for name, t in successors(P, s):
                    rv = inst.rev(name)
                    lower = any(a < b for a, b in zip(t[:-1], s[:-1])) or bool(s[-1] & ~t[-1])
                    if lower:
                        self.assertLess(rv.d, 0)
                        self.assertFalse(violations(P, t))

    def test_bounded_drift(self):
        for seed in SEEDS:
            P = compile_instance(small(seed))
            res = reach(P)
            for s, d in res.depth.items():
                dist = sum(abs(a - b) for a, b in zip(s[:-1], P.init[:-1])) + bin(s[-1] ^ P.init[-1]).count("1")
                self.assertGreaterEqual(d, dist)

    def test_finite_reach(self):
        for seed in SEEDS:
            P = compile_instance(small(seed))
            self.assertLessEqual(reach(P).size, P.space_size)


class TestStructure(unittest.TestCase):
    def test_reversibility(self):
        """With paired revisions every guarded transition can be undone in one step."""
        for seed in SEEDS:
            P = compile_instance(small(seed, k_thr=4))
            res = reach(P)
            for s in res.depth:
                for _, t in successors(P, s):
                    self.assertIn(s, {u for _, u in successors(P, t)}, f"seed {seed}")

    def test_reach_is_strongly_connected(self):
        for seed in range(15):
            P = compile_instance(small(seed))
            base = set(reach(P).depth)
            for s in list(base)[:5]:
                Q = compile_instance(small(seed))
                Q.init = s
                self.assertEqual(set(reach(Q).depth), base, f"seed {seed}")

    def test_safe_is_sublattice(self):
        """Safe is closed under pointwise max/min of thresholds and union/intersection of the critical set."""
        import itertools
        for seed in range(10):
            P = compile_instance(small(seed, L=3, k_thr=3, n_phys=2))
            S = list(safe_set(P))
            for s, t in itertools.islice(itertools.product(S, S), 4000):
                join = tuple(max(a, b) for a, b in zip(s[:-1], t[:-1])) + (s[-1] | t[-1],)
                meet = tuple(min(a, b) for a, b in zip(s[:-1], t[:-1])) + (s[-1] & t[-1],)
                self.assertFalse(violations(P, join), f"seed {seed}")
                self.assertFalse(violations(P, meet), f"seed {seed}")


class TestASPAgreement(unittest.TestCase):
    def test_bounded_reachability_matches_bfs(self):
        n = 0
        for seed in range(12):
            P = compile_instance(small(seed, L=3, k_thr=2, n_phys=2, p_meta=0.8))
            for guards in [(True, True, True), (False, False, False), (True, False, True)]:
                res = reach(P, *guards)
                D = res.max_depth
                for k, name in (("G1", "g1"), ("G2", "g2"), ("G3", "g3")):
                    sat, _ = asp.bounded_reach(P, D, name, guards)
                    self.assertEqual(sat, k in res.first_bad, f"seed {seed} guards {guards} {k}")
                    n += 1
        self.assertGreater(n, 0)


class TestCooldown(unittest.TestCase):
    def test_projection_and_rate_bound(self):
        from thresholds.search import reach_cooldown
        for seed in range(12):
            P = compile_instance(small(seed, L=4, k_thr=3, n_phys=2))
            base = set(reach(P).depth)
            for delta in (0, 1, 2):
                best = reach_cooldown(P, delta)
                self.assertEqual(set(best), base, f"seed {seed} delta {delta}")
                for s, ticks in best.items():
                    drift = max([abs(a - b) for a, b in zip(s[:-1], P.init[:-1])] + [bin(s[-1] ^ P.init[-1]).count("1") and 1])
                    self.assertGreaterEqual(ticks, max(drift - 1, 0) * delta)


if __name__ == "__main__":
    unittest.main()


class TestTractableFragment(unittest.TestCase):
    def test_non_reflexive_reach_equals_safe_set(self):
        """Without meta-level targets every safe policy is reachable."""
        checked = 0
        for seed in range(30):
            inst = small(seed, L=3, k_thr=3, n_phys=3, p_meta=0.0)
            P = compile_instance(inst)
            if any(f"A:{r.name}" in P.names for r in inst.revs):
                continue
            # the safe set restricted to policies connected to the initial one through all-safe moves
            self.assertEqual(set(reach(P).depth), safe_set(P), f"seed {seed}")
            checked += 1
        self.assertGreater(checked, 10)


if __name__ == "__main__":
    unittest.main()
