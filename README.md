# Assurance-Gated Trust — reference implementation

This is a **demonstration artifact** for a vision/formal paper that proposes
an assurance-gated extension of a trust-regulated dynamic epistemic
framework for heterogeneous multi-agent systems with LLM agents. This code
is not meant to be a production verification system. Its purpose is to show
that the semantics in the paper's formal sections can be executed, that the
two narrative scenarios behave exactly as described, and that the paper's
control-safety and conditional domain-safety claims hold up empirically over
a large number of randomized trajectories — including showing *where
exactly* things break when an assumption or a structural guard is
deliberately violated.

## What's here

```
agt/                      the reference-monitor engine (Sections 3–4)
  core.py                   trust levels, Omega={Pass,Inc,Fail}, Agg, Certificate, Commit, resolve_Pi, World, Policy
  decisions.py               source_OK, m_decision, reviewer_OK, ValidTrace/trace_OK, decision, DelCand
  beliefs.py                 the mental-action grammar's belief transformations (+phi, vdash, cap, down, dashv)
  engine.py                   state transitions: mental_step, validate_review, retract_dependency, practical_step

examples/                 scripted replays of Section 6, step by step with console narration
  healthcare_case_study.py   source admit -> LLM review -> commit -> delegate -> authorize -> retraction invalidates trace
  backup_deletion.py          "Transfer beyond healthcare": irreversible deleteBackup(b), Inc/Fail/revocation paths

stress/                   large-scale randomized testing of the *real* engine (not a statistical model)
  invariants.py              post-hoc checkers for the five Phi_ctl clauses (Sec. 4.5)
  random_trajectories.py     N random heterogeneous trajectories; empirical checks for semantic containment,
                              conditional domain-safety preservation (B1–B4), and strict expressivity (Theorem 1)

evaluation/               Section 5's statistical workload (Table 2, ablations, sensitivity sweep)
  synthetic_workload.py      packaged copy of prototype.py (see below), unmodified

tests/                    pytest suite
  test_core_properties.py    Proposition "sanity properties": partition, monotonicity, Fail=>block, allow=>Pass, decision recovery
  test_overhead.py            Proposition "finite-state decidability/overhead": exact O(m+d) operation counts
  test_examples.py            runs both Section 6 narratives end to end
  test_invariants_smoke.py    small-n smoke test of the stress harness's qualitative pattern

prototype.py              original, standalone script for Section 5's aggregate-counter workload;
                          kept at the repo root for transparency. evaluation/synthetic_workload.py
                          is the same code, bit-for-bit, just packaged for `run_demo.py`.

run_demo.py               single CLI entry point (see below)
```

## Running it

```bash
cd "Assurance Gated Trust"
python run_demo.py examples          # narrated Section 6 replay
python run_demo.py stress --n 20000  # randomized engine stress test (default n=20000)
python run_demo.py evaluation        # Section 5 Table 2 / ablations / sensitivity reproduction
python run_demo.py all               # everything above

python -m pytest tests/ -v           # the pytest suite (requires `pip install pytest`)
```

Only the Python standard library is required for the demo itself (Python
3.9+; tested on 3.12). `pytest` is only needed to run `tests/`.

A full, reproducible snapshot of one such run (narratives, stress test,
Table 2 reproduction, pytest) is logged in [`RESULTS.md`](RESULTS.md).

## Paper-to-code map

| Paper object / equation | Section | Code |
|---|---|---|
| `Levels`, `<=_T`, `Agg(S)`, `Omega={Pass,Inc,Fail}` | 3, 4.1 | `agt/core.py`: `LEVELS`, `leT`/`geT`, `aggregate` |
| `kappa = <id,V,i,xi,omega,w0,D,S,status>`, `cert_valid` | 4.1 | `agt/core.py`: `Certificate`, `Certificate.valid` |
| Commit record, `cmt_valid`, `conflict`, `resolve_Pi` | 4.1 | `agt/core.py`: `Commit`, `conflict`, `resolve_conservative`/`resolve_by_priority` |
| `source_OK`, `m_decision` | 4.2 | `agt/decisions.py`: `source_ok`, `mdecision` |
| Belief transformations for `+phi, vdash, cap, down, dashv` | 3, 4.2 | `agt/beliefs.py` |
| `reviewer_OK`, pending-review store `Rvw`, commit on validation | 4.2 | `agt/decisions.py: reviewer_ok`; `agt/engine.py: mental_step`, `validate_review` |
| `Crit_Pi`, `Impact_Pi`, trusted trace `J`, `ValidTrace`, `trace_OK` | 4.3 | `agt/decisions.py`: `Policy.is_critical`, `TraceItem`, `validtrace`, `traceok` |
| `decision` (practical), `DelCand`, delegation-as-routing | 4.3 | `agt/decisions.py: decision_practical, delegation_candidates`; `agt/engine.py: practical_step` |
| `Phi_ctl = {phi1..phi5}` | 4.5 | `stress/invariants.py` |
| Theorem 1 (strict expressivity) | 5 | `stress/random_trajectories.py`: paired-episode `pi_T`-invariance check |
| Theorem (decision recovery / static conservativity) | 5 | `tests/test_core_properties.py: test_ordinary_agent_bypass_matches_trust_only_policy` |
| Lemma (stale-evidence exclusion) | 5 | exercised by `examples/healthcare_case_study.py` step 7–8 and `examples/backup_deletion.py` |
| Theorem (domain-safety lifting), premises B1–B4 | 5 | `stress/random_trajectories.py`: `default_experiments` (B2/B3/B4-violated configs) |
| Corollary (interface-compositional preservation) | 5 | not executable here (it is a composition argument over components not instantiated in this single-MAS demo); see Limitations below |
| Proposition (sanity properties) | 4.5 | `tests/test_core_properties.py` |
| Proposition (finite-state decidability / `O(m+d)` overhead) | 4.5 | `tests/test_overhead.py` |
| Section 5 Table 2 / ablations / sensitivity sweep | 5 | `prototype.py` / `evaluation/synthetic_workload.py` (same code, packaged) |
| Section 6 healthcare narrative | 6 | `examples/healthcare_case_study.py` |
| Section 6 "Transfer beyond healthcare" (`deleteBackup`) | 6 | `examples/backup_deletion.py` |

## What the stress test actually shows

`stress/random_trajectories.py` runs the *real* engine (certificates,
review/commit lifecycle, resolver, delegation) over thousands of randomized
heterogeneous episodes, not just an aggregate statistical counter. At
`n=20000` per configuration it reproduces exactly the distinction the paper's
"Failure localization" paragraph (Sec. 5) argues for:

```
configuration                                         phi_viol  psi_viol
B1-B4 hold (sound verifiers, sound review, safe resolver)       0        0
B2 violated: 15% false-positive, single verifier                0       58
B3 violated: unsound human validation                           0       12
B4 violated: unsafe conflict resolver                            0       43
Guard ablation: lifecycle invalidation disabled                 57       57
Guard ablation: conflict guard disabled                        3448       0
Guard ablation: delegate re-evaluation disabled                 4313      0
```

Violating an external trust premise (B2/B3/B4: an unsound verifier, an
unsound human reviewer, an unsafe conflict resolver) can introduce
domain-level (`Psi`) safety violations, but it **never** breaks the five
structural `Phi_ctl` invariants. Conversely, disabling one of the three
implementation guards (certificate-lifecycle invalidation, conflict
resolution, delegatee re-evaluation) breaks `Phi_ctl` itself — exactly the
separation the paper draws between "accepts a step" and "preserves safety."

The same run also gives an empirical witness of Theorem 1 (strict
expressivity): across thousands of paired episodes sharing an identical
trust-only projection, the trust-only policy's decision never changes
(`pi_T`-invariance, 0 violations), while the assurance-aware policy's
decision tracks the run-specific assurance outcome in the large majority of
cases where that outcome differs between the paired runs.

## Honest scope and simplifications

This is a demonstration, not a full implementation of every object in the
paper:

- **Beliefs are minimal propositional atoms**, not a full L-DINF epistemic
  model with possible worlds, accessibility relations, budgets, or group
  cooperation. This is enough to drive the mental-action grammar's
  preconditions and the control semantics being demonstrated, which is the
  paper's actual contribution — not the base epistemic logic, which is
  inherited unchanged from the referenced framework.
- **Verifiers are mock/randomized functions**, not real LLM critics, symbolic
  checkers, or human reviewers. `VerifierSpec` is a pluggable interface;
  wiring in a real verifier means implementing that one callable.
- **The Corollary on interface-compositional preservation** is not
  separately exercised: this demo instantiates one heterogeneous MAS, not
  multiple composed components with their own local invariants.
- `prototype.py` (and its packaged copy `evaluation/synthetic_workload.py`)
  is an intentionally *separate*, simpler aggregate-counter model — it is
  what produced Table 2's numbers and is kept bit-for-bit identical to the
  paper; it does not share code with `agt/`, which is the actual
  object-level engine built for this demo.
