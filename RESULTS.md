# Validation run log

Run date: 2026-10-02 · Python 3.12.10 (stdlib only; `pytest` for `tests/`)

Commands:
```
python run_demo.py all --n 20000
python -m pytest tests/ -v
```

This file is a snapshot of one full, reproducible run of the artifact
described in `README.md`. All seeds are fixed, so re-running the same
commands reproduces these numbers exactly.

## 1. Section 6 narratives (`run_demo.py examples`)

Both scripted replays completed with every inline assertion holding (no
`AssertionError`), i.e. the engine reproduces the paper's narrative exactly:

- **Healthcare case study**: `+p`, `+v` admitted (Pass) → `β=p∧v` autonomously
  accepted (Pass, two verifiers agree) → `α=down(p∧v,q)` routed to **review**
  (Inc, despite `very_high` trust) → physician `h` validates → **commit**
  `c_q` created → `transfer_higher_care` **delegated** (Inc) → `h` re-evaluated
  fresh → **executed**. After the lab **retracts** `p`, `c_q` becomes stale and
  a repeated authorization attempt is **blocked** (`trace_ok=False`).
- **Backup deletion** (`deleteBackup(b)`, irreversible/critical): Inc →
  **delegate** → administrator re-evaluated fresh → **executed**; a separate
  run with a **Fail** verdict is **blocked outright even at `very_high`
  trust**; after the backup-state evidence is **revoked**, the same trace no
  longer authorizes the action.

## 2. Randomized semantic-engine stress test (`stress/random_trajectories.py`, n=20,000/config)

Counts are against the *real* `agt.engine` transitions (certificates,
review/commit, resolver, delegation), not a statistical model.

| Configuration | Phi_ctl violations | Psi violations | Psi / n |
|---|---:|---:|---:|
| B1–B4 hold (sound verifiers, sound review, safe resolver) | 0 | 0 | 0.000% |
| B2 violated: 15% false-positive, single verifier | 0 | 58 | 0.290% |
| B3 violated: unsound human validation | 0 | 12 | 0.060% |
| B4 violated: unsafe conflict resolver | 0 | 43 | 0.215% |
| Guard ablation: lifecycle invalidation disabled | 57 (`phi2`) | 57 | 0.285% |
| Guard ablation: conflict guard disabled | 3448 (`phi5`) | 0 | 0.000% |
| Guard ablation: delegate re-evaluation disabled | 4313 (`phi4`) | 0 | 0.000% |

Pattern: violating an external trust premise (B2/B3/B4) can cost domain-level
(Psi) safety but never breaks the five structural Phi_ctl invariants;
disabling an implementation guard breaks Phi_ctl itself. This is the
"Failure localization" distinction (Sec. 5) made executable.

**Theorem 1 (strict expressivity), paired-episode check** (B1–B4-hold config):
- trust-only `π_T`-invariance checks: 6151, violations: **0** (expected 0)
- pairs with identical trust-only projection but different assurance: 3707
- of those, the assurance-aware decision also differed: 3507 (**94.6%**)

## 3. Section 5 Table 2 reproduction (`evaluation/synthetic_workload.py`, 30 seeds × 2,000 episodes)

| Metric | Trust-only | Assurance-aware (sound) |
|---|---:|---:|
| Review rate | 14.30 [14.04, 14.56]% | 36.57 [36.05, 37.09]% |
| Delegation rate | 15.11 [14.78, 15.45]% | 35.10 [34.68, 35.52]% |
| Human interventions / episode | 0.284 [0.281, 0.288] | 0.788 [0.780, 0.796] |
| Normalized routing cost / episode | 1.424 [1.407, 1.441] | 7.674 [7.625, 7.723] |
| Invalid events / episode | 0.365 [0.359, 0.370] | 0.000 [0.000, 0.000] |
| Episodes with ≥1 invalid event | 24.06 [23.74, 24.38]% | 0.00 [0.00, 0.00]% |

Paired Wilcoxon (invalid-episode rate, trust-only > assurance-aware):
W+=465.0, n=30, **p = 9.31×10⁻¹⁰**. Total Phi violations under sound
assurance across all 60,000 trajectories: **0**.

5% false-positive ablations (invalid events/episode): 2 verifiers
0.00043 [0.00026, 0.00061]; 1 verifier 0.00693 [0.00626, 0.00761]; no
lifecycle guard 0.00057 [0.00038, 0.00075]. Paired Wilcoxon: 1 verifier >
2 verifiers p = 9.31×10⁻¹⁰; no lifecycle > full lifecycle p = 0.0039.

Sensitivity sweep (invalid events/episode, 1 vs 2 verifiers): at 0% false
positives, 0.000000 / 0.000000; at 15%, 0.022967 / 0.003433.

All four targeted lifecycle/routing stress tests (conflict guard,
delegate re-evaluation, both directions) report **PASS**.

## 4. Pytest suite (`python -m pytest tests/ -v`)

**20 passed, 0 failed** in 9.36s:

```
test_core_properties.py   7 passed  (decision partition, trust-monotonicity,
                                      Fail=>block, allow=>Pass, decision recovery)
test_examples.py          2 passed  (both Sec. 6 narratives)
test_invariants_smoke.py  8 passed  (Phi_ctl / Psi separation, Theorem 1 check)
test_overhead.py          3 passed  (exact O(m+d) operation counts)
```
