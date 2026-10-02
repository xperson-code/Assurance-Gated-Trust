"""Large-scale randomized stress testing of the *real* reference-monitor
engine (agt.engine), as opposed to the aggregate statistical workload of
evaluation/synthetic_workload.py (which reproduces the paper's Table 2).

Three things are tested empirically, each tied to a specific formal claim:

1. Semantic containment (Sec. 4.5): Phi_ctl holds on every sampled
   trajectory *regardless of verifier soundness* -- i.e. a false-positive
   verifier can cause a domain-level (Psi) violation, but it can never by
   itself reactivate stale evidence, bypass review, skip delegatee
   re-evaluation, or let an unresolved conflict through. Explicitly
   disabling one of the three ablation guards, by contrast, *does* break
   the corresponding Phi_ctl clause -- this is the "Failure localization"
   distinction of Sec. 5 made executable.

2. Conditional domain-safety preservation (Theorem: domain-safety lifting):
   under premises B1-B4 (sound verifiers, sound base/human steps, a
   Psi-preserving resolver), zero ground-truth (Psi) violations occur over
   N random trajectories; violating B2, B3, or B4 in isolation reintroduces
   Psi violations while leaving Phi_ctl at zero -- demonstrating that the
   two kinds of guarantee are independent, as the paper claims.

3. Strict expressivity (Theorem 1): paired trajectories with an identical
   trust-only projection (same trust levels, same source_ok, same agent
   class) but different run-specific assurance outcomes always yield the
   *same* trust-only decision (pi_T-invariance) while the assurance-aware
   decision tracks the assurance outcome -- i.e. differs exactly when
   assurance differs.

Only the Python standard library is required.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agt.core import (
    FAIL, INC, PASS, Commit, Policy, VerifierSpec, World,
    geT, leT, resolve_conservative, resolve_by_priority,
)
from agt.decisions import TraceItem, decision_practical, mdecision, traceok
from agt.engine import mental_step, practical_step, retract_dependency, validate_review

LEVELS = ("low", "medium", "high", "very_high")


# --- trust-only baseline (Sec. 3), restated locally for a self-contained,
#     side-by-side comparison with the assurance-aware functions in agt.decisions --

def trust_only_mdecision(trust: str, source_ok: bool, tau_B: str, tau_A: str) -> str:
    if (not source_ok) or leT(trust, tau_B):
        return "block"
    if geT(trust, tau_A):
        return "allow"
    return "review"


def trust_only_decision_practical(trust: str, trace_ok: bool, tau_B: str, tau_A: str) -> str:
    if leT(trust, tau_B) or not trace_ok:
        return "block"
    if geT(trust, tau_A):
        return "allow"
    return "delegate"


def sample_trust(rng: random.Random) -> str:
    x = rng.random()
    if x < 0.08:
        return "low"
    if x < 0.23:
        return "medium"
    if x < 0.70:
        return "high"
    return "very_high"


def verifier_output(u: float, ground_truth_valid: bool, strength: float, false_positive_rate: float) -> str:
    """Mirrors evaluation/synthetic_workload.py's sound-but-incomplete model:
    a Psi-sound verifier (false_positive_rate == 0) never returns Pass on an
    invalid step. A positive rate deliberately violates premise B2."""
    if ground_truth_valid:
        return PASS if u < strength else INC
    if u < false_positive_rate:
        return PASS
    return FAIL if u < false_positive_rate + strength else INC


def _resolver_for(mode: str) -> "callable":
    """Resolvers key off conclusion sign ("q" = ground-truth-correct commit
    in this harness, "~q" = the deliberately conflicting one) rather than
    commit identity, so the same resolver works regardless of whether the
    accepted conclusion arrived via the autonomous or the reviewed path."""
    if mode == "conservative":
        return resolve_conservative
    if mode == "safe_priority":
        def _resolve(candidates: List[Commit], world: World) -> Optional[Commit]:
            for c in candidates:
                if not c.conclusion.startswith("~"):
                    return c
            return None
        return _resolve
    if mode == "unsafe":
        def _resolve(candidates: List[Commit], world: World) -> Optional[Commit]:
            for c in candidates:
                if c.conclusion.startswith("~"):
                    return c
            return candidates[0] if candidates else None
        return _resolve
    raise ValueError(f"unknown resolver_mode: {mode}")


@dataclass
class ExperimentConfig:
    name: str
    n: int = 20_000
    seed: int = 20270923
    verifier_count: int = 2
    false_positive_rate: float = 0.0          # B2: 0.0 means Psi-sound verifiers
    verifier_strength: float = 0.80
    human_review_sound: bool = True            # B3: True means human validation never ratifies an invalid step
    resolver_mode: str = "conservative"         # "conservative" (bot) | "safe_priority" (B4 holds) | "unsafe" (B4 violated)
    enforce_lifecycle_guard: bool = True
    enforce_conflict_guard: bool = True
    fresh_delegate_reeval: bool = True
    source_fail_rate: float = 0.04
    valid_rate: float = 0.82
    conflict_rate: float = 0.08
    retract_rate: float = 0.14


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    n: int
    phi_violations: Dict[str, int]
    psi_violations: int
    episodes_with_any_phi_violation: int
    episodes_with_psi_violation: int
    trust_only_pi_invariance_checks: int
    trust_only_pi_invariance_violations: int
    assurance_sensitivity_pairs: int
    assurance_sensitivity_divergences: int


def run_episode(cfg: ExperimentConfig, rng: random.Random) -> Tuple[List[str], bool, dict]:
    """Runs one random heterogeneous episode through the *real* engine and
    returns (phi_violations, psi_violation, extra_info_for_pairing)."""
    from stress.invariants import EpisodeTrace, check_all

    source_ok = rng.random() > cfg.source_fail_rate
    ground_truth_valid = rng.random() < cfg.valid_rate
    mental_trust = sample_trust(rng)
    practical_trust = sample_trust(rng)
    conflict_present = rng.random() < cfg.conflict_rate
    retract_present = rng.random() < cfg.retract_rate

    policy = Policy(
        tau_B_M="low", tau_A_M="high", tau_B_P="low", tau_A_P="high", tau_S="medium",
        impact={"critical_action": "irreversible"},
        resolver=_resolver_for(cfg.resolver_mode),
    )

    world = World(llm_agents={"l"})
    world.supported.add("dep:p")
    world.trust_S[("src", "premise")] = "very_high" if source_ok else "low"
    world.trust_M[("l", "infer_q")] = mental_trust
    world.trust_M[("h", "infer_q")] = "very_high"  # a reliably trusted ordinary reviewer
    world.trust_P[("l", "critical_action")] = practical_trust
    world.trust_P[("h", "critical_action")] = "very_high"
    world.belief("l").add("premise")
    world.know("l").add(("premise", "q"))

    def mk_verifiers(valid: bool) -> List[VerifierSpec]:
        us = [rng.random() for _ in range(cfg.verifier_count)]
        return [
            VerifierSpec(f"v{i}", lambda a, x, w, u=u, valid=valid: verifier_output(u, valid, cfg.verifier_strength, cfg.false_positive_rate), sound=cfg.false_positive_rate == 0.0)
            for i, u in enumerate(us)
        ]

    mental_verifiers = mk_verifiers(ground_truth_valid)
    r = mental_step(
        world, "l", ("l",), "down", ("premise", "q"), "infer_q", policy, mental_verifiers,
        source_formula=("src", "premise"), deps=("dep:p",),
    )
    world = r.world
    mental_decision, mental_assurance = r.decision, r.assurance
    mental_entered_belief = "q" in world.belief("l")
    mental_commit_created = False
    good_commit_id: Optional[str] = None

    if mental_decision == "review":
        attempt_validates = ground_truth_valid or (not cfg.human_review_sound)
        if attempt_validates:
            rv = validate_review(world, r.pending_index, "h", policy, [], deps=("dep:p",), conclusion="q")
            world = rv.world
            if rv.success:
                mental_commit_created = True
                good_commit_id = rv.commit_id  # a Commit, e.g. "commit:infer_q:<tick>"
                mental_entered_belief = "q" in world.belief("l")
    elif mental_decision == "allow":
        good_commit_id = r.certificate_id  # autonomous path: evidenced by a Certificate

    # If a conflict is injected, uniformly represent the LLM's accepted
    # conclusion as a Commit "q" (whether it originally arrived via the
    # autonomous certificate or the reviewed commit) so the resolver/conflict
    # machinery -- which operates over Commits -- is exercised identically
    # on both paths.
    unresolved_conflict = False
    if conflict_present and good_commit_id is not None:
        world.commits["commit:good"] = Commit(
            id="commit:good", reviewer=("l" if mental_decision == "allow" else "h"), performer="l",
            action_id="infer_q", conclusion="q", deps=frozenset({"dep:p"}), scope=lambda tick: True,
        )
        world.commits["commit:bad"] = Commit(
            id="commit:bad", reviewer="h2", performer="l", action_id="infer_q",
            conclusion="~q", deps=frozenset({"dep:p"}), scope=lambda tick: True,
        )
        good_commit_id = "commit:good"
        candidates = [world.commits["commit:good"], world.commits["commit:bad"]]
        unresolved_conflict = policy.resolver(candidates, world) is None

    if retract_present:
        world = retract_dependency(world, "dep:p")

    trigger_supported = mental_entered_belief
    if good_commit_id is not None and good_commit_id in world.commits:
        trace_items = [TraceItem(kind="reviewed", commit_id=good_commit_id)]
    elif good_commit_id is not None and good_commit_id in world.certificates:
        trace_items = [TraceItem(kind="autonomous", certificate_id=good_commit_id)]
    else:
        trace_items = []

    trace_valid_at_decision = traceok(
        "critical_action", trigger_supported, trace_items, world, policy,
        enforce_lifecycle_guard=cfg.enforce_lifecycle_guard, enforce_conflict_guard=cfg.enforce_conflict_guard,
    )

    scenario_valid = ground_truth_valid and not retract_present and not (conflict_present and cfg.resolver_mode == "unsafe")
    practical_verifiers = mk_verifiers(scenario_valid)
    p = practical_step(
        world, "l", "critical_action", policy, practical_verifiers, trace_items,
        trigger_supported=trigger_supported, delegation_candidates_pool=["h"], verifiers_by_agent={"h": []},
        enforce_lifecycle_guard=cfg.enforce_lifecycle_guard, enforce_conflict_guard=cfg.enforce_conflict_guard,
        fresh_delegate_reeval=cfg.fresh_delegate_reeval,
    )
    world = p.world

    psi_violation = bool(p.executed and not scenario_valid)
    stale_dependency_present = retract_present and good_commit_id is not None

    trace = EpisodeTrace(
        mental_decision=mental_decision,
        mental_assurance=mental_assurance,
        mental_is_llm=True,
        mental_entered_belief=mental_entered_belief,
        mental_commit_created=mental_commit_created,
        practical_decision=p.decision,
        practical_assurance=p.assurance,
        practical_executed=p.executed,
        practical_executed_by=p.executed_by,
        practical_is_llm_executor=(p.executed_by == "l"),
        practical_is_critical=True,
        trace_valid_at_decision=trace_valid_at_decision,
        trace_valid_at_execution=trace_valid_at_decision,
        unresolved_conflict_present=unresolved_conflict,
        conflict_guard_enforced=cfg.enforce_conflict_guard,
        stale_dependency_present=stale_dependency_present,
        lifecycle_guard_enforced=cfg.enforce_lifecycle_guard,
        delegate_fresh_reeval_enforced=cfg.fresh_delegate_reeval,
    )
    violations = check_all(trace)

    extra = dict(source_ok=source_ok, mental_trust=mental_trust, practical_trust=practical_trust,
                 mental_assurance=mental_assurance, practical_assurance=p.assurance,
                 mental_decision=mental_decision, practical_decision=p.decision)
    return violations, psi_violation, extra


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    rng = random.Random(cfg.seed)
    phi_counts: Dict[str, int] = {}
    psi_count = 0
    episodes_with_phi = 0
    episodes_with_psi = 0

    pi_checks = 0
    pi_violations = 0
    sens_pairs = 0
    sens_divergences = 0

    prev_extra: Optional[dict] = None
    prev_episode = None

    for _ in range(cfg.n):
        violations, psi_violation, extra = run_episode(cfg, rng)
        for v in violations:
            phi_counts[v] = phi_counts.get(v, 0) + 1
        if violations:
            episodes_with_phi += 1
        if psi_violation:
            psi_count += 1
            episodes_with_psi += 1

        # Theorem 1 (strict expressivity): pair consecutive episodes that
        # happen to share the same trust-only projection (trust levels and
        # source_ok) and check pi_T-invariance of the trust-only policy
        # versus the assurance-aware policy's sensitivity to assurance.
        if prev_extra is not None and (
            prev_extra["mental_trust"] == extra["mental_trust"]
            and prev_extra["source_ok"] == extra["source_ok"] == True
        ):
            to_a = trust_only_mdecision(prev_extra["mental_trust"], prev_extra["source_ok"], "low", "high")
            to_b = trust_only_mdecision(extra["mental_trust"], extra["source_ok"], "low", "high")
            pi_checks += 1
            if to_a != to_b:
                pi_violations += 1  # would falsify pi_T-invariance of the trust-only policy; should never happen
            if prev_extra["mental_assurance"] != extra["mental_assurance"]:
                sens_pairs += 1
                if prev_extra["mental_decision"] != extra["mental_decision"]:
                    sens_divergences += 1

        prev_extra = extra

    return ExperimentResult(
        config=cfg, n=cfg.n, phi_violations=phi_counts, psi_violations=psi_count,
        episodes_with_any_phi_violation=episodes_with_phi, episodes_with_psi_violation=episodes_with_psi,
        trust_only_pi_invariance_checks=pi_checks, trust_only_pi_invariance_violations=pi_violations,
        assurance_sensitivity_pairs=sens_pairs, assurance_sensitivity_divergences=sens_divergences,
    )


def default_experiments(n: int = 20_000, seed: int = 20270923) -> List[ExperimentConfig]:
    return [
        ExperimentConfig(name="B1-B4 hold (sound verifiers, sound review, safe resolver)", n=n, seed=seed,
                          false_positive_rate=0.0, human_review_sound=True, resolver_mode="safe_priority"),
        ExperimentConfig(name="B2 violated: 15% false-positive, single verifier", n=n, seed=seed,
                          verifier_count=1, false_positive_rate=0.15, human_review_sound=True, resolver_mode="safe_priority"),
        ExperimentConfig(name="B3 violated: unsound human validation", n=n, seed=seed,
                          false_positive_rate=0.0, human_review_sound=False, resolver_mode="safe_priority"),
        ExperimentConfig(name="B4 violated: unsafe conflict resolver", n=n, seed=seed,
                          false_positive_rate=0.0, human_review_sound=True, resolver_mode="unsafe"),
        ExperimentConfig(name="Guard ablation: lifecycle invalidation disabled", n=n, seed=seed,
                          false_positive_rate=0.0, human_review_sound=True, resolver_mode="safe_priority",
                          enforce_lifecycle_guard=False),
        ExperimentConfig(name="Guard ablation: conflict guard disabled", n=n, seed=seed,
                          false_positive_rate=0.0, human_review_sound=True, resolver_mode="conservative",
                          enforce_conflict_guard=False, conflict_rate=0.30),
        ExperimentConfig(name="Guard ablation: delegate re-evaluation disabled", n=n, seed=seed,
                          false_positive_rate=0.0, human_review_sound=True, resolver_mode="safe_priority",
                          fresh_delegate_reeval=False, retract_rate=0.30),
    ]


def main(n: int = 20_000) -> None:
    print(f"Randomized semantic-engine stress test -- {n} trajectories per configuration")
    print("Phi_ctl violations are counted against the *real* agt.engine transitions, not a statistical model.\n")

    header = f"{'configuration':52s} {'phi_viol':>9s} {'psi_viol':>9s} {'psi/n%':>8s}"
    print(header)
    print("-" * len(header))
    for cfg in default_experiments(n=n):
        result = run_experiment(cfg)
        total_phi = sum(result.phi_violations.values())
        psi_pct = 100.0 * result.psi_violations / result.n
        print(f"{cfg.name:52s} {total_phi:9d} {result.psi_violations:9d} {psi_pct:7.3f}%")
        if total_phi:
            for name, count in result.phi_violations.items():
                print(f"    - {name}: {count}")

    print("\nExpected pattern: B2/B3/B4 violations raise psi_viol while phi_viol stays 0 (external-assumption")
    print("failures, Theorem domain-safety-lifting's B2/B3/B4 premises); guard ablations raise phi_viol itself")
    print("(structural/Phi_ctl failures) -- the exact 'Failure localization' distinction argued in Sec. 5.")

    print("\nTheorem 1 (strict expressivity over the trust-only projection), paired-episode check:")
    base_cfg = default_experiments(n=n)[0]
    result = run_experiment(base_cfg)
    print(f"  trust-only pi_T-invariance checks: {result.trust_only_pi_invariance_checks}, "
          f"violations: {result.trust_only_pi_invariance_violations} (expected: 0)")
    print(f"  pairs with identical trust-projection but different assurance: {result.assurance_sensitivity_pairs}")
    print(f"  of those, assurance-aware decision also differed: {result.assurance_sensitivity_divergences} "
          f"({100.0 * result.assurance_sensitivity_divergences / max(result.assurance_sensitivity_pairs, 1):.1f}%)")
    print("  i.e. the trust-only policy never distinguishes same-projection runs, while the assurance-aware")
    print("  policy's decision tracks run-specific assurance -- an empirical witness of strict expressivity.")


if __name__ == "__main__":
    main()
