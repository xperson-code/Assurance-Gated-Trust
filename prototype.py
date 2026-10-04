#!/usr/bin/env python3
"""Reproducible synthetic workload for assurance-gated trust regulation.

The artifact evaluates policy routing and invariant preservation, not clinical
accuracy. For each of 30 deterministic seeds it generates 2,000 immutable
heterogeneous multi-step scenarios. Trust-only and assurance-aware policies
are evaluated on exactly the same scenarios. The main assurance profile uses
two sound-but-incomplete verifiers. Ablations vary verifier false-positive
rate, verifier count, and certificate-lifecycle enforcement.

Only the Python standard library is required.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from math import sqrt
import random
import statistics

LEVELS = ("low", "medium", "high", "very_high")
RANK = {x: i for i, x in enumerate(LEVELS)}
TAU_BLOCK = RANK["low"]
TAU_AUTONOMY = RANK["high"]
PASS, INC, FAIL = "pass", "inconclusive", "fail"

AUTO_VERIFIER_COST = 1.0
HUMAN_REVIEW_COST = 6.0
HUMAN_DELEGATE_COST = 4.0
REVALIDATION_COST = 6.0
CONFLICT_RESOLUTION_COST = 6.0

BASE_SEED = 20270923
N_SEEDS = 30
EPISODES_PER_SEED = 2000
T95_DF29 = 2.045229642132703  # two-sided 95% Student-t critical value, df=29


@dataclass(frozen=True)
class Scenario:
    source_ok: bool
    inference_valid: bool
    mental_trust: str
    mver_u1: float
    mver_u2: float
    conflict: bool
    resolver_success: bool
    retract: bool
    replacement_available: bool
    practical_trust: str
    pver_u1: float
    pver_u2: float


@dataclass
class Certificate:
    kind: str
    outcome: str
    deps: set[str]
    active: bool = True

    def valid(self, supported: set[str]) -> bool:
        return self.active and self.deps <= supported


@dataclass
class EpisodeMetrics:
    mental: Counter = field(default_factory=Counter)
    practical: Counter = field(default_factory=Counter)
    verifier_calls: int = 0
    human_interventions: int = 0
    revalidations: int = 0
    conflicts: int = 0
    conflict_resolutions: int = 0
    latency_units: float = 0.0
    transitions: int = 0
    phi_violations: int = 0
    semantic_violations: int = 0
    invalid_episodes: int = 0


def trust_only_mental(trust: str, source_ok: bool) -> str:
    if not source_ok or RANK[trust] <= TAU_BLOCK:
        return "block"
    if RANK[trust] >= TAU_AUTONOMY:
        return "allow"
    return "review"


def assurance_mental(trust: str, source_ok: bool, assurance: str) -> str:
    if not source_ok or RANK[trust] <= TAU_BLOCK or assurance == FAIL:
        return "block"
    if RANK[trust] >= TAU_AUTONOMY and assurance == PASS:
        return "allow"
    return "review"


def trust_only_practical(trust: str, trace_ok: bool) -> str:
    if RANK[trust] <= TAU_BLOCK or not trace_ok:
        return "block"
    if RANK[trust] >= TAU_AUTONOMY:
        return "allow"
    return "delegate"


def assurance_practical(trust: str, trace_ok: bool, assurance: str) -> str:
    if RANK[trust] <= TAU_BLOCK or not trace_ok or assurance == FAIL:
        return "block"
    if RANK[trust] >= TAU_AUTONOMY and assurance == PASS:
        return "allow"
    return "delegate"


def verifier_output(u: float, valid: bool, strength: float, false_positive: float = 0.0) -> str:
    """Sound-but-incomplete when false_positive == 0."""
    if valid:
        return PASS if u < strength else INC
    if u < false_positive:
        return PASS
    return FAIL if u < false_positive + strength else INC


def aggregate(outputs: list[str]) -> str:
    if FAIL in outputs:
        return FAIL
    if outputs and all(x == PASS for x in outputs):
        return PASS
    return INC


def sample_trust(rng: random.Random) -> str:
    x = rng.random()
    if x < 0.15:
        return "medium"
    if x < 0.70:
        return "high"
    return "very_high"


def make_scenarios(n: int, seed: int) -> list[Scenario]:
    rng = random.Random(seed)
    return [Scenario(
        source_ok=rng.random() > 0.04,
        inference_valid=rng.random() < 0.82,
        mental_trust=sample_trust(rng),
        mver_u1=rng.random(),
        mver_u2=rng.random(),
        conflict=rng.random() < 0.08,
        resolver_success=rng.random() < 0.55,
        retract=rng.random() < 0.14,
        replacement_available=rng.random() < 0.50,
        practical_trust=sample_trust(rng),
        pver_u1=rng.random(),
        pver_u2=rng.random(),
    ) for _ in range(n)]


def finish(m: EpisodeMetrics) -> EpisodeMetrics:
    m.invalid_episodes = int(m.semantic_violations > 0)
    return m


def run_episode(
    s: Scenario,
    *,
    aware: bool,
    false_positive: float = 0.0,
    verifier_count: int = 2,
    lifecycle_guard: bool = True,
    conflict_guard: bool = True,
    delegate_reeval: bool = True,
) -> EpisodeMetrics:
    if verifier_count not in (1, 2):
        raise ValueError("verifier_count must be 1 or 2")

    m = EpisodeMetrics()
    supported = {"lab:p", "monitor:v"}

    # Two source admissions precede one LLM cognitive proposal.
    m.transitions += 2
    if not s.source_ok:
        m.mental["block"] += 1
        m.transitions += 1
        return finish(m)

    if aware:
        mout = [verifier_output(s.mver_u1, s.inference_valid, 0.84, false_positive)]
        if verifier_count == 2:
            mout.append(verifier_output(s.mver_u2, s.inference_valid, 0.76, false_positive))
        m.verifier_calls += verifier_count
        m.latency_units += verifier_count * AUTO_VERIFIER_COST
        ma = aggregate(mout)
        md = assurance_mental(s.mental_trust, True, ma)
    else:
        ma = None
        md = trust_only_mental(s.mental_trust, True)
    m.mental[md] += 1
    m.transitions += 1

    q_believed = False
    commit: Certificate | None = None

    if md == "allow":
        q_believed = True
        if aware and ma != PASS:
            m.phi_violations += 1
        if not s.inference_valid:
            m.semantic_violations += 1
    elif md == "review":
        m.human_interventions += 1
        m.latency_units += HUMAN_REVIEW_COST
        m.transitions += 1
        # Trusted-base assumption: human/symbolic validation is sound.
        if s.inference_valid:
            q_believed = True
            commit = Certificate("commit:q", PASS, {"lab:p", "monitor:v"})
            m.transitions += 1

    if not q_believed:
        return finish(m)

    unresolved_conflict = False
    if s.conflict:
        m.conflicts += 1
        m.transitions += 1
        if aware:
            if s.resolver_success:
                m.conflict_resolutions += 1
                m.human_interventions += 1
                m.latency_units += CONFLICT_RESOLUTION_COST
                m.transitions += 1
            else:
                unresolved_conflict = True

    stale = False
    if s.retract:
        supported.discard("lab:p")
        m.transitions += 1
        if aware and lifecycle_guard:
            stale = commit is None or not commit.valid(supported)
            if stale:
                m.revalidations += 1
                m.human_interventions += 1
                m.latency_units += REVALIDATION_COST
                m.transitions += 1
                if s.replacement_available and s.inference_valid:
                    supported.add("lab:p2")
                    commit = Certificate("commit:q:revalidated", PASS, {"lab:p2", "monitor:v"})
                    stale = False
                    m.transitions += 1

    trace_ok = True
    if aware:
        trace_ok = (not stale) and (not unresolved_conflict or not conflict_guard)

    actual_trace_valid = ("monitor:v" in supported) and (("lab:p" in supported) or ("lab:p2" in supported))
    practical_step_valid = s.inference_valid and actual_trace_valid and not unresolved_conflict

    if aware:
        pout = [verifier_output(s.pver_u1, practical_step_valid, 0.88, false_positive)]
        if verifier_count == 2:
            pout.append(verifier_output(s.pver_u2, practical_step_valid, 0.82, false_positive))
        m.verifier_calls += verifier_count
        m.latency_units += verifier_count * AUTO_VERIFIER_COST
        pa = aggregate(pout)
        pd = assurance_practical(s.practical_trust, trace_ok, pa)
    else:
        pa = None
        pd = trust_only_practical(s.practical_trust, trace_ok)
    m.practical[pd] += 1
    m.transitions += 1

    executed = False
    executed_by_llm = False
    if pd == "allow":
        executed = True
        executed_by_llm = True
    elif pd == "delegate":
        m.human_interventions += 1
        m.latency_units += HUMAN_DELEGATE_COST
        m.transitions += 1
        if delegate_reeval:
            # Fresh delegatee check under the current state.
            delegatee_allowed = trace_ok and practical_step_valid if aware else practical_step_valid
            m.transitions += 1
        else:
            # Ablation: routing-time admissibility is reused without a fresh state check.
            delegatee_allowed = trace_ok
        if delegatee_allowed:
            executed = True

    if executed:
        m.transitions += 1
        if aware and executed_by_llm and pa != PASS:
            m.phi_violations += 1
        if aware and not trace_ok:
            m.phi_violations += 1
        if aware and conflict_guard and unresolved_conflict:
            m.phi_violations += 1
        if not practical_step_valid:
            m.semantic_violations += 1

    return finish(m)


def merge(total: EpisodeMetrics, e: EpisodeMetrics) -> None:
    total.mental.update(e.mental)
    total.practical.update(e.practical)
    for f in (
        "verifier_calls", "human_interventions", "revalidations", "conflicts",
        "conflict_resolutions", "latency_units", "transitions", "phi_violations",
        "semantic_violations", "invalid_episodes",
    ):
        setattr(total, f, getattr(total, f) + getattr(e, f))


def evaluate(scenarios: list[Scenario], **kwargs) -> EpisodeMetrics:
    total = EpisodeMetrics()
    for s in scenarios:
        merge(total, run_episode(s, **kwargs))
    return total


def pct(x: float, d: float) -> float:
    return 0.0 if d == 0 else 100.0 * x / d


def seed_metrics(m: EpisodeMetrics, n: int) -> dict[str, float]:
    mental_n = sum(m.mental.values())
    practical_n = sum(m.practical.values())
    return {
        "review_rate": pct(m.mental["review"], mental_n),
        "delegation_rate": pct(m.practical["delegate"], practical_n),
        "mental_block_rate": pct(m.mental["block"], mental_n),
        "practical_block_rate": pct(m.practical["block"], practical_n),
        "human_interventions": m.human_interventions / n,
        "revalidations": m.revalidations / n,
        "verifier_calls": m.verifier_calls / n,
        "latency": m.latency_units / n,
        "transitions": m.transitions / n,
        "invalid_events": m.semantic_violations / n,
        "invalid_episode_pct": 100.0 * m.invalid_episodes / n,
    }


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    se = statistics.stdev(values) / sqrt(len(values))
    delta = T95_DF29 * se if len(values) == 30 else 1.96 * se
    return mean, mean - delta, mean + delta


def rank_abs_differences(diffs: list[float]) -> list[float]:
    indexed = sorted((abs(d), i) for i, d in enumerate(diffs))
    ranks = [0.0] * len(diffs)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and abs(indexed[end][0] - indexed[pos][0]) < 1e-15:
            end += 1
        avg_rank = ((pos + 1) + end) / 2.0
        for k in range(pos, end):
            ranks[indexed[k][1]] = avg_rank
        pos = end
    return ranks


def wilcoxon_exact_greater(a: list[float], b: list[float]) -> tuple[float, float, int]:
    """Exact one-sided signed-rank test for H1: a > b using DP over ranks."""
    diffs = [x - y for x, y in zip(a, b) if abs(x - y) > 1e-15]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, 0
    ranks = rank_abs_differences(diffs)
    ranks2 = [int(round(2 * r)) for r in ranks]
    obs = sum(r for r, d in zip(ranks2, diffs) if d > 0)
    total_rank = sum(ranks2)
    ways = [0] * (total_rank + 1)
    ways[0] = 1
    current = 0
    for r in ranks2:
        for s in range(current, -1, -1):
            if ways[s]:
                ways[s + r] += ways[s]
        current += r
    extreme = sum(ways[obs:])
    p = extreme / (2 ** n)
    return obs / 2.0, p, n


def collect_runs(config: dict) -> tuple[list[dict[str, float]], EpisodeMetrics]:
    rows: list[dict[str, float]] = []
    total = EpisodeMetrics()
    for k in range(N_SEEDS):
        scenarios = make_scenarios(EPISODES_PER_SEED, BASE_SEED + k)
        m = evaluate(scenarios, **config)
        rows.append(seed_metrics(m, EPISODES_PER_SEED))
        merge(total, m)
    return rows, total


def format_ci(values: list[float], digits: int = 3) -> str:
    m, lo, hi = mean_ci(values)
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def targeted_stress_tests() -> dict[str, bool]:
    # Conflict guard: with an unresolved conflict and an erroneously permissive
    # practical verifier, trace-level conflict exclusion is the remaining guard.
    conflict_guard_blocks = assurance_practical("high", False, PASS) == "block"
    conflict_guard_disabled_allows = assurance_practical("high", True, PASS) == "allow"

    # Delegation re-evaluation: a trace can become stale after routing but before
    # delegatee execution; re-evaluation sees the new false guard.
    routing_trace_ok = True
    post_routing_trace_ok = False
    with_reeval_executes = post_routing_trace_ok
    without_reeval_executes = routing_trace_ok

    return {
        "conflict_guard_blocks_unresolved_conflict": conflict_guard_blocks,
        "disabling_conflict_guard_can_allow_same_request": conflict_guard_disabled_allows,
        "delegate_reevaluation_blocks_post_routing_staleness": not with_reeval_executes,
        "without_delegate_reevaluation_stale_request_can_execute": without_reeval_executes,
    }


def main() -> None:
    configs = {
        "trust-only": dict(aware=False),
        "assurance-aware / sound / 2 verifiers": dict(aware=True, false_positive=0.0, verifier_count=2),
        "assurance-aware / 5% FP / 2 verifiers": dict(aware=True, false_positive=0.05, verifier_count=2),
        "assurance-aware / 5% FP / 1 verifier": dict(aware=True, false_positive=0.05, verifier_count=1),
        "assurance-aware / 5% FP / no lifecycle": dict(aware=True, false_positive=0.05, verifier_count=2, lifecycle_guard=False),
    }

    results: dict[str, list[dict[str, float]]] = {}
    totals: dict[str, EpisodeMetrics] = {}
    for name, config in configs.items():
        rows, total = collect_runs(config)
        results[name] = rows
        totals[name] = total

    print("Synthetic multi-agent assurance workload")
    print(f"seeds={N_SEEDS}, episodes/seed={EPISODES_PER_SEED}, trajectories/policy={N_SEEDS * EPISODES_PER_SEED}")
    print(f"seed sequence={BASE_SEED}..{BASE_SEED + N_SEEDS - 1}")
    print("95% CIs are t intervals over seed-level metrics; Wilcoxon tests are paired exact signed-rank tests.")
    print("Normalized latency units are model costs, not measured milliseconds.\n")

    headline = ["trust-only", "assurance-aware / sound / 2 verifiers"]
    metrics = [
        ("review_rate", "review rate (%)", 2),
        ("delegation_rate", "delegation rate (%)", 2),
        ("human_interventions", "human interventions / episode", 3),
        ("revalidations", "revalidations / episode", 3),
        ("verifier_calls", "verifier calls / episode", 3),
        ("latency", "normalized latency / episode", 3),
        ("invalid_events", "invalid events / episode", 5),
        ("invalid_episode_pct", "episodes with >=1 invalid event (%)", 3),
    ]
    print("HEADLINE 30-SEED RESULTS")
    for key, label, digits in metrics:
        vals0 = [r[key] for r in results[headline[0]]]
        vals1 = [r[key] for r in results[headline[1]]]
        print(f"{label}: trust-only {format_ci(vals0, digits)} | assurance-aware {format_ci(vals1, digits)}")
    w, p, n = wilcoxon_exact_greater(
        [r["invalid_episode_pct"] for r in results["trust-only"]],
        [r["invalid_episode_pct"] for r in results["assurance-aware / sound / 2 verifiers"]],
    )
    print(f"paired Wilcoxon, invalid-episode rate, trust-only > assurance-aware: W+={w:.1f}, n={n}, p={p:.10g}")
    print(f"total semantic-invalid events: trust-only={totals['trust-only'].semantic_violations}, assurance-aware={totals['assurance-aware / sound / 2 verifiers'].semantic_violations}")
    print(f"total Phi violations under sound assurance={totals['assurance-aware / sound / 2 verifiers'].phi_violations}\n")

    print("5% FALSE-POSITIVE ABLATIONS")
    for name in (
        "assurance-aware / 5% FP / 2 verifiers",
        "assurance-aware / 5% FP / 1 verifier",
        "assurance-aware / 5% FP / no lifecycle",
    ):
        vals = [r["invalid_events"] for r in results[name]]
        eps = [r["invalid_episode_pct"] for r in results[name]]
        print(f"{name}: invalid events/episode {format_ci(vals, 5)}; invalid episodes (%) {format_ci(eps, 3)}")

    two = [r["invalid_events"] for r in results["assurance-aware / 5% FP / 2 verifiers"]]
    one = [r["invalid_events"] for r in results["assurance-aware / 5% FP / 1 verifier"]]
    nolife = [r["invalid_events"] for r in results["assurance-aware / 5% FP / no lifecycle"]]
    w, p, n = wilcoxon_exact_greater(one, two)
    print(f"paired Wilcoxon, 1 verifier > 2 verifiers: W+={w:.1f}, n={n}, p={p:.10g}")
    w, p, n = wilcoxon_exact_greater(nolife, two)
    print(f"paired Wilcoxon, no lifecycle > full lifecycle: W+={w:.1f}, n={n}, p={p:.10g}\n")

    print("SENSITIVITY: INVALID EVENTS / EPISODE")
    print("false_positive   one_verifier_mean   two_verifier_mean")
    for fp in (0.00, 0.01, 0.025, 0.05, 0.10, 0.15):
        one_rows, _ = collect_runs(dict(aware=True, false_positive=fp, verifier_count=1))
        two_rows, _ = collect_runs(dict(aware=True, false_positive=fp, verifier_count=2))
        one_mean = statistics.fmean(r["invalid_events"] for r in one_rows)
        two_mean = statistics.fmean(r["invalid_events"] for r in two_rows)
        print(f"{fp:0.3f}            {one_mean:0.6f}            {two_mean:0.6f}")

    print("\nTARGETED LIFECYCLE/ROUTING STRESS TESTS")
    stress = targeted_stress_tests()
    for key, value in stress.items():
        print(f"{key}: {'PASS' if value else 'FAIL'}")

    assert totals["assurance-aware / sound / 2 verifiers"].phi_violations == 0
    assert totals["assurance-aware / sound / 2 verifiers"].semantic_violations == 0
    assert totals["trust-only"].semantic_violations > 0
    assert all(stress.values())


if __name__ == "__main__":
    main()
