# Self-Revisable Trust Thresholds for Mental and Practical Actions

Executable companion of the journal paper *Self-Revisable Trust Thresholds for Mental and Practical
Actions in a Dynamic Epistemic Framework* (Costantini, Pitoni, De Lauretis; Intelligenza Artificiale,
RCRA-it 2026 special issue). It extends the trust-regulated L-DINF framework of the RCRA-it 2026
workshop paper by making the trust thresholds revisable from within the system.

The code implements the finite policy transition system of the paper and lets one

* explore the set of reachable policies (least fixed point) by explicit-state search,
* check bounded reachability of "bad" policies with an Answer Set Programming (ASP) encoding (clingo),
* ablate the guards (G1) well-formedness, (G2) entrenchment floors, (G3) non-lockout,
* replay the security-operations case study of the paper,
* validate empirically the theorems of the paper on random instances.

(G4), the requirement of a justification trace for loosening, depends on the epistemic layer and is
checked by the reference semantics (`thresholds/policy.py`) but not by the search, which is therefore a
sound over-approximation of the dynamics.

## Layout

| path | content |
|---|---|
| `thresholds/policy.py` | reference (dictionary based) semantics: policies, revision actions, guards, proposal/review/validation |
| `thresholds/search.py` | compiled state space, successor relation, BFS reachability, greatest fixed point |
| `thresholds/asp.py` | clingo encoding of bounded policy reachability |
| `thresholds/generator.py` | random instances with a safe initial policy |
| `thresholds/casestudy.py` | the SOC case study and its replay |
| `tests/test_properties.py` | tests of the theorems and of the agreement of the three implementations |
| `experiments/run_experiments.py` | experiments of the paper |
| `results/` | CSV files, log and figures produced by the experiments |

## Requirements and usage

Python 3.10+ and the packages in `requirements.txt` (`pip install -r requirements.txt`).

```
python -m thresholds.casestudy                      # replay of the case study (Table of the paper)
python -m unittest discover -s tests -v             # property tests and implementation agreement
python experiments/run_experiments.py --quick       # small run (a few seconds)
python experiments/run_experiments.py               # full run used in the paper (a few minutes)
```

## Model in one paragraph

A policy assigns a trust level to every threshold entry (autonomy, blocking, execution, ... of each
action, and one threshold per informational source) and fixes the set of critical actions. Revision
actions raise or lower one entry by one level, or add or remove one action from the critical set. They
are atomic names with a target, so an action may target an entry about itself. A revision is applied only
if some agent reaches its autonomy threshold in the current policy (pre-state evaluation) and the revised
policy satisfies the guards. Only the entries targeted by some revision can change, so the search state is
the vector of their values plus a bitmask for the critical set.

## Results

Generated files in `results/`: `case_study_replay.csv`, `case_study_reach.csv`, `ablation_summary.csv`,
`scalability_summary.csv`, `large_summary.csv`, `cooldown.csv`, `scalability.pdf`.
