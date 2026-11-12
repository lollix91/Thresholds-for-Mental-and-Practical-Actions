"""Experiments of the paper: case study verification, guard ablation, scalability.

Usage:  python experiments/run_experiments.py [--quick]
Results (CSV, LaTeX tables, figures) are written to ``results/``.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from thresholds import asp  # noqa: E402
from thresholds.casestudy import replay, soc_instance  # noqa: E402
from thresholds.generator import generate  # noqa: E402
from thresholds.search import compile_instance, reach  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

CONFIGS = {
    "none": (False, False, False),
    "G1": (True, False, False),
    "G2": (False, True, False),
    "G3": (False, False, True),
    "G1+G2": (True, True, False),
    "G1+G3": (True, False, True),
    "G2+G3": (False, True, True),
    "G1+G2+G3": (True, True, True),
}


def e1_case_study():
    rows = replay()
    pd.DataFrame(rows, columns=["step", "action", "outcome", "detail"]).to_csv(
        os.path.join(OUT, "case_study_replay.csv"), index=False)
    inst = soc_instance()
    P = compile_instance(inst)
    out = []
    for name, g in CONFIGS.items():
        r = reach(P, *g)
        out.append(dict(config=name, reach=r.size, space=P.space_size,
                        floor=("G2" in r.first_bad), wf=("G1" in r.first_bad), lockout=("G3" in r.first_bad),
                        floor_path=" ".join(r.first_bad["G2"][1]) if "G2" in r.first_bad else "",
                        seconds=round(r.seconds, 4)))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "case_study_reach.csv"), index=False)
    # ASP certificate: guarded policy cannot reach a floor violation within the BFS diameter
    r = reach(P)
    sat, t = asp.bounded_reach(P, r.max_depth, "g2", CONFIGS["G1+G2+G3"])
    print("[E1] guarded: ASP floor violation reachable within diameter:", sat, f"({t:.3f}s)")
    sat0, t0 = asp.bounded_reach(P, 6, "g2", CONFIGS["none"])
    print("[E1] unguarded: ASP floor violation reachable within 6 steps:", sat0, f"({t0:.3f}s)")
    print(df.to_string(index=False))
    return df


def e2_ablation(n: int, L=4, k_thr=4):
    rows = []
    for seed in range(n):
        inst = generate(L=L, k_thr=k_thr, k_crit=1, n_phys=3, n_src=2, n_agents=3,
                        p_meta=0.6, p_floor=0.6, seed=1000 + seed)
        P = compile_instance(inst)
        for name, g in CONFIGS.items():
            r = reach(P, *g)
            rows.append(dict(seed=seed, config=name, reach=r.size, space=P.space_size,
                             G1=("G1" in r.first_bad), G2=("G2" in r.first_bad), G3=("G3" in r.first_bad),
                             depth_G2=r.first_bad["G2"][0] if "G2" in r.first_bad else None,
                             seconds=r.seconds, truncated=r.truncated))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ablation_raw.csv"), index=False)
    agg = df.groupby("config", sort=False).agg(
        reach_mean=("reach", "mean"), reach_max=("reach", "max"),
        pct_G1=("G1", "mean"), pct_G2=("G2", "mean"), pct_G3=("G3", "mean"),
        depth_G2=("depth_G2", "mean")).reset_index()
    for c in ("pct_G1", "pct_G2", "pct_G3"):
        agg[c] = (100 * agg[c]).round(1)
    agg["reach_mean"] = agg["reach_mean"].round(1)
    agg["depth_G2"] = agg["depth_G2"].round(2)
    agg.to_csv(os.path.join(OUT, "ablation_summary.csv"), index=False)
    print(agg.to_string(index=False))
    return agg


def e3_scalability(ks, Ls, seeds, asp_max_k):
    rows = []
    for L, k in itertools.product(Ls, ks):
        for seed in range(seeds):
            inst = generate(L=L, k_thr=k, k_crit=1, n_phys=max(3, k), n_src=2, n_agents=3,
                            p_meta=0.6, p_floor=0.3, seed=5000 + 100 * k + seed)
            P = compile_instance(inst)
            g = reach(P)
            u = reach(P, False, False, False)
            row = dict(L=L, k=k, seed=seed, entries=len(inst.entries()), revisions=len(inst.revs),
                       space=P.space_size, reach_guarded=g.size, reach_unguarded=u.size,
                       diameter=g.max_depth, t_bfs_guarded=g.seconds, t_bfs_unguarded=u.seconds,
                       unguarded_floor=("G2" in u.first_bad))
            if k <= asp_max_k:
                sat, t = asp.bounded_reach(P, g.max_depth, "g2", CONFIGS["G1+G2+G3"])
                satu, tu = asp.bounded_reach(P, u.max_depth, "g2", CONFIGS["none"])
                row.update(t_asp=t, asp_agrees=(not sat), t_asp_unguarded=tu, asp_agrees_unguarded=(satu == ("G2" in u.first_bad)))
            rows.append(row)
        print(f"[E3] L={L} k={k} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "scalability_raw.csv"), index=False)
    agg = df.groupby(["L", "k"]).agg(
        space=("space", "mean"), reach_g=("reach_guarded", "mean"), reach_u=("reach_unguarded", "mean"),
        diam=("diameter", "mean"), t_g=("t_bfs_guarded", "mean"), t_u=("t_bfs_unguarded", "mean"),
        t_asp=("t_asp", "mean"), t_asp_u=("t_asp_unguarded", "mean"),
        asp_ok=("asp_agrees", "mean"), asp_ok_u=("asp_agrees_unguarded", "mean")).reset_index()
    agg.to_csv(os.path.join(OUT, "scalability_summary.csv"), index=False)
    print(agg.to_string(index=False))
    return df, agg




def e4_large(seeds):
    """Guarded exploration in spaces where exhaustive unguarded exploration is infeasible."""
    rows = []
    for L, ks in ((5, [10, 11, 12, 13]), (9, [7, 8, 9])):
        for k in ks:
            for seed in range(seeds):
                inst = generate(L=L, k_thr=k, k_crit=1, n_phys=k, n_src=2, n_agents=3,
                                p_meta=0.6, p_floor=0.3, seed=7000 + 10 * k + seed)
                P = compile_instance(inst)
                g = reach(P, max_states=2_000_000)
                sat, t = asp.bounded_reach(P, g.max_depth, "g2", CONFIGS["G1+G2+G3"])
                rows.append(dict(L=L, k=k, seed=seed, space=P.space_size, reach=g.size,
                                 diameter=g.max_depth, truncated=g.truncated,
                                 t_bfs=g.seconds, t_asp=t, asp_certifies=(not sat and not g.truncated)))
            print(f"[E4] L={L} k={k} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "large_raw.csv"), index=False)
    agg = df.groupby(["L", "k"]).agg(space=("space", "mean"), reach=("reach", "mean"),
                                    diam=("diameter", "mean"), t_bfs=("t_bfs", "mean"),
                                    t_asp=("t_asp", "mean"), truncated=("truncated", "sum"),
                                    certified=("asp_certifies", "mean")).reset_index()
    agg.to_csv(os.path.join(OUT, "large_summary.csv"), index=False)
    print(agg.to_string(index=False))
    return agg




def e5_cooldown(n: int = 40):
    """Cooldown between consecutive revisions of the same entry: reachability is preserved, time grows."""
    from thresholds.search import reach_cooldown
    rows = []
    P = compile_instance(soc_instance())
    for delta in range(4):
        best = reach_cooldown(P, delta)
        rows.append(dict(instance="case study", delta=delta, policies=len(best),
                         max_ticks=max(best.values()), mean_ticks=sum(best.values()) / len(best)))
    for delta in range(4):
        pol, mx, mn = [], [], []
        for seed in range(n):
            inst = generate(L=4, k_thr=3, k_crit=1, n_phys=3, n_src=2, n_agents=3,
                            p_meta=0.6, p_floor=0.3, seed=9000 + seed)
            P = compile_instance(inst)
            best = reach_cooldown(P, delta)
            assert len(best) == reach(P).size
            pol.append(len(best)); mx.append(max(best.values())); mn.append(sum(best.values()) / len(best))
        rows.append(dict(instance=f"random ({n})", delta=delta, policies=sum(pol) / n,
                         max_ticks=sum(mx) / n, mean_ticks=sum(mn) / n))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "cooldown.csv"), index=False)
    print(df.to_string(index=False))
    return df


def figures(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for L, d in agg.groupby("L"):
        ax[0].plot(d["k"], d["reach_g"], "o-", label=f"guarded, L={L}")
        ax[0].plot(d["k"], d["reach_u"], "s--", label=f"unguarded, L={L}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("targeted entries k"); ax[0].set_ylabel("reachable policies")
    ax[0].legend(fontsize=6)
    for L, d in agg.groupby("L"):
        ax[1].plot(d["k"], d["t_g"], "o-", label=f"BFS guarded, L={L}")
        dd = d.dropna(subset=["t_asp"])
        if len(dd):
            ax[1].plot(dd["k"], dd["t_asp"], "^:", label=f"ASP (clingo), L={L}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("targeted entries k"); ax[1].set_ylabel("time (s)")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scalability.pdf"))
    fig.savefig(os.path.join(OUT, "scalability.png"), dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-large", action="store_true", help="run only the large-instance experiment")
    ap.add_argument("--only-cooldown", action="store_true", help="run only the cooldown experiment")
    a = ap.parse_args()
    if a.only_cooldown:
        e5_cooldown()
        return
    if a.only_large:
        e4_large(4)
        return
    e1_case_study()
    e2_ablation(40 if a.quick else 200)
    ks = [2, 3, 4, 5] if a.quick else [2, 3, 4, 5, 6, 7, 8]
    Ls = [3, 5]
    df, agg = e3_scalability(ks, Ls, 3 if a.quick else 8, asp_max_k=4 if a.quick else 8)
    figures(agg)
    if not a.quick:
        e4_large(4)
        e5_cooldown()


if __name__ == "__main__":
    main()
