"""A fast (small-n) smoke test over the randomized semantic-engine stress
harness (stress/random_trajectories.py), checking the qualitative pattern
the full-scale run (run_demo.py stress, or the module's own __main__) is
expected to show at n=20,000: Phi_ctl violations occur only when a
structural guard is explicitly disabled; domain-level (Psi) violations
occur only when a B2/B3/B4 trust premise is explicitly violated.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stress.random_trajectories import ExperimentConfig, run_experiment

N = 12000  # large enough that the rarer B3 violation (~0.06%/episode) reliably triggers at least once
SEED = 20270923


def _phi_total(result) -> int:
    return sum(result.phi_violations.values())


def test_sound_full_policy_has_no_phi_or_psi_violations():
    cfg = ExperimentConfig(name="sound", n=N, seed=SEED, false_positive_rate=0.0,
                            human_review_sound=True, resolver_mode="safe_priority")
    result = run_experiment(cfg)
    assert _phi_total(result) == 0
    assert result.psi_violations == 0


def test_b2_violation_raises_psi_but_not_phi():
    cfg = ExperimentConfig(name="b2", n=N, seed=SEED, verifier_count=1, false_positive_rate=0.15,
                            human_review_sound=True, resolver_mode="safe_priority")
    result = run_experiment(cfg)
    assert _phi_total(result) == 0
    assert result.psi_violations > 0


def test_b3_violation_raises_psi_but_not_phi():
    cfg = ExperimentConfig(name="b3", n=N, seed=SEED, false_positive_rate=0.0,
                            human_review_sound=False, resolver_mode="safe_priority")
    result = run_experiment(cfg)
    assert _phi_total(result) == 0
    assert result.psi_violations > 0


def test_b4_violation_raises_psi_but_not_phi():
    cfg = ExperimentConfig(name="b4", n=N, seed=SEED, false_positive_rate=0.0,
                            human_review_sound=True, resolver_mode="unsafe")
    result = run_experiment(cfg)
    assert _phi_total(result) == 0
    assert result.psi_violations > 0


def test_lifecycle_guard_ablation_raises_phi():
    cfg = ExperimentConfig(name="lifecycle-off", n=N, seed=SEED, false_positive_rate=0.0,
                            human_review_sound=True, resolver_mode="safe_priority",
                            enforce_lifecycle_guard=False)
    result = run_experiment(cfg)
    assert result.phi_violations.get("phi2_no_stale_trace", 0) > 0


def test_conflict_guard_ablation_raises_phi():
    cfg = ExperimentConfig(name="conflict-off", n=N, seed=SEED, false_positive_rate=0.0,
                            human_review_sound=True, resolver_mode="conservative",
                            enforce_conflict_guard=False, conflict_rate=0.30)
    result = run_experiment(cfg)
    assert result.phi_violations.get("phi5_no_unresolved_conflict_support", 0) > 0


def test_delegate_reeval_ablation_raises_phi():
    cfg = ExperimentConfig(name="reeval-off", n=N, seed=SEED, false_positive_rate=0.0,
                            human_review_sound=True, resolver_mode="safe_priority",
                            fresh_delegate_reeval=False, retract_rate=0.30)
    result = run_experiment(cfg)
    assert result.phi_violations.get("phi4_delegation_is_rechecked", 0) > 0


def test_trust_only_policy_is_pi_T_invariant():
    cfg = ExperimentConfig(name="expressivity", n=N, seed=SEED, false_positive_rate=0.0,
                            human_review_sound=True, resolver_mode="safe_priority")
    result = run_experiment(cfg)
    assert result.trust_only_pi_invariance_violations == 0
    assert result.trust_only_pi_invariance_checks > 0
    assert result.assurance_sensitivity_divergences > 0
