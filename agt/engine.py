"""The transition engine: ties core objects, decisions, and belief updates
into the concrete state transitions w =>_TA w' used by the examples and by
the stress-testing harness.

Each public function here returns a *new* world (functional update,
``World.clone``) together with a small record describing what happened, so
that callers (examples, stress tests) can both narrate and audit a run.

Paper correspondence: this module operationalizes the clauses in Sec. 4.2
("Assurance-aware mental decisions") and Sec. 4.3 ("Valid traces and
assurance-aware practical decisions"), including cost consumption, the
pending-review store, commit creation on validation, and delegation as
routing-then-re-evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .beliefs import apply_belief_transformation, preconditions_hold
from .core import Certificate, Commit, PASS, Policy, VerifierSpec, World
from .decisions import TraceItem, decision_practical, mdecision, reviewer_ok, traceok


@dataclass
class MentalStepResult:
    world: World
    decision: str  # "allow" | "review" | "block" | "not_enabled"
    assurance: Optional[str]
    certificate_id: Optional[str] = None
    pending_index: Optional[int] = None


def mental_step(
    world: World,
    agent: str,
    group: Tuple[str, ...],
    mental_op: str,
    args: Tuple[str, ...],
    action_id: str,
    policy: Policy,
    verifiers: List[VerifierSpec],
    source_formula: Optional[Tuple[str, str]] = None,
    deps: Tuple[str, ...] = (),
    cost: float = 1.0,
) -> MentalStepResult:
    """One [G:alpha] mental-action transition (Sec. 4.2)."""
    w = world.clone()
    if not preconditions_hold(w, agent, mental_op, args):
        w.say(f"{agent}: {mental_op}{args} -- ordinary preconditions fail, action not enabled")
        return MentalStepResult(w, "not_enabled", None)

    decision, assur = mdecision(agent, action_id, w, policy, verifiers, source_formula)
    if decision in ("allow", "review"):
        # Sec. 3: "a mental step that reaches either allow or review consumes
        # its cognitive execution cost; a blocked ... step does not."
        w.budgets[agent] = w.budgets.get(agent, float("inf")) - cost / max(len(group), 1)

    if decision == "allow":
        apply_belief_transformation(w, agent, mental_op, args)
        cert_id = f"cert:{action_id}:{w.tick}"
        w.certificates[cert_id] = Certificate(
            id=cert_id,
            verifiers=frozenset(v.name for v in verifiers),
            agent=agent,
            action_id=action_id,
            omega=assur,
            origin_tick=w.tick,
            deps=frozenset(deps),
            scope=lambda tick: True,
        )
        w.say(f"{agent}: {mental_op}{args} -- ALLOW (assurance={assur}); belief updated, cert {cert_id} issued")
        return MentalStepResult(w, decision, assur, certificate_id=cert_id)

    if decision == "review":
        from .core import PendingReview

        pending = PendingReview(group=group, performer=agent, action_id=action_id, mental_op=mental_op, args=args, created_tick=w.tick)
        w.review_store.append(pending)
        w.say(f"{agent}: {mental_op}{args} -- REVIEW (assurance={assur}); pending, no belief change yet")
        return MentalStepResult(w, decision, assur, pending_index=len(w.review_store) - 1)

    w.say(f"{agent}: {mental_op}{args} -- BLOCK (assurance={assur})")
    return MentalStepResult(w, decision, assur)


@dataclass
class ValidationResult:
    world: World
    success: bool
    commit_id: Optional[str] = None


def validate_review(
    world: World,
    pending_index: int,
    reviewer: str,
    policy: Policy,
    reviewer_check_verifiers: List[VerifierSpec],
    deps: Tuple[str, ...],
    conclusion: str,
) -> ValidationResult:
    """A reviewer validates a pending step: removes it from Rvw, applies the
    belief transformation, and records a persistent Commit (Sec. 4.2)."""
    w = world.clone()
    if pending_index >= len(w.review_store):
        return ValidationResult(w, False)
    pending = w.review_store[pending_index]

    if not reviewer_ok(reviewer, pending.performer, pending.action_id, w, policy, reviewer_check_verifiers):
        w.say(f"{reviewer}: validation of {pending.action_id} by {pending.performer} -- reviewer_OK fails")
        return ValidationResult(w, False)
    if not preconditions_hold(w, pending.performer, pending.mental_op, pending.args):
        w.say(f"{reviewer}: validation of {pending.action_id} -- ordinary preconditions no longer hold")
        return ValidationResult(w, False)

    apply_belief_transformation(w, pending.performer, pending.mental_op, pending.args)
    w.review_store.pop(pending_index)
    commit_id = f"commit:{pending.action_id}:{w.tick}"
    w.commits[commit_id] = Commit(
        id=commit_id,
        reviewer=reviewer,
        performer=pending.performer,
        action_id=pending.action_id,
        conclusion=conclusion,
        deps=frozenset(deps),
        scope=lambda tick: True,
    )
    w.say(f"{reviewer}: validated {pending.action_id} for {pending.performer} -- COMMIT {commit_id} ({conclusion})")
    return ValidationResult(w, True, commit_id=commit_id)


def retract_dependency(world: World, dep_id: str) -> World:
    """A source/retraction transition inherited from the base MAS (B3):
    removes dep_id from the current support set, which may make dependent
    certificates/commits stale (Lemma: stale-evidence exclusion)."""
    w = world.clone()
    w.supported.discard(dep_id)
    w.say(f"retract({dep_id}) -- dependency no longer supported")
    return w


@dataclass
class PracticalStepResult:
    world: World
    decision: str  # "allow" | "delegate" | "block"
    assurance: Optional[str]
    executed: bool
    executed_by: Optional[str] = None


def practical_step(
    world: World,
    agent: str,
    action_id: str,
    policy: Policy,
    verifiers: List[VerifierSpec],
    trace_items: List[TraceItem],
    trigger_supported: bool,
    delegation_candidates_pool: List[str] = (),
    verifiers_by_agent: Optional[Dict[str, List[VerifierSpec]]] = None,
    cost: float = 1.0,
    *,
    enforce_lifecycle_guard: bool = True,
    enforce_conflict_guard: bool = True,
    fresh_delegate_reeval: bool = True,
) -> PracticalStepResult:
    """One critical/ordinary practical-action transition (Sec. 4.3), including
    delegation as routing followed by a *fresh* re-evaluation of the
    delegatee under the current guards.

    The three keyword-only flags default to the full semantics and exist
    only to drive the paper's ablations (Sec. 5) through these same
    decision functions: disabling ``enforce_lifecycle_guard`` reproduces
    "disabling certificate lifecycle invalidation"; disabling
    ``enforce_conflict_guard`` reproduces the unresolved-conflict ablation;
    disabling ``fresh_delegate_reeval`` reuses the routing-time
    admissibility for the first candidate instead of re-checking it under
    the current state ("disabling delegatee re-evaluation").
    """
    w = world.clone()
    trace_valid = traceok(
        action_id, trigger_supported, trace_items, w, policy,
        enforce_lifecycle_guard=enforce_lifecycle_guard, enforce_conflict_guard=enforce_conflict_guard,
    )
    decision, assur = decision_practical(agent, action_id, w, policy, verifiers, trace_valid)
    w.budgets[agent] = w.budgets.get(agent, float("inf")) - cost

    if decision == "allow":
        w.say(f"{agent}: {action_id} -- ALLOW (assurance={assur}, trace_ok={trace_valid}); executed autonomously")
        return PracticalStepResult(w, decision, assur, True, executed_by=agent)

    if decision == "block":
        w.say(f"{agent}: {action_id} -- BLOCK (assurance={assur}, trace_ok={trace_valid})")
        return PracticalStepResult(w, decision, assur, False)

    # decision == "delegate": routing, then (by default) a fresh re-evaluation of each candidate
    w.say(f"{agent}: {action_id} -- DELEGATE (assurance={assur}, trace_ok={trace_valid})")
    verifiers_by_agent = verifiers_by_agent or {}

    if not fresh_delegate_reeval:
        for candidate in delegation_candidates_pool:
            w.say(f"{candidate}: {action_id} -- ABLATION: executed on routing-time trace_ok without re-check")
            return PracticalStepResult(w, "delegate", assur, bool(trace_valid), executed_by=candidate)
        return PracticalStepResult(w, "delegate", assur, False)

    for candidate in delegation_candidates_pool:
        trace_valid_fresh = traceok(
            action_id, trigger_supported, trace_items, w, policy,
            enforce_lifecycle_guard=enforce_lifecycle_guard, enforce_conflict_guard=enforce_conflict_guard,
        )
        cand_decision, cand_assur = decision_practical(
            candidate, action_id, w, policy, verifiers_by_agent.get(candidate, []), trace_valid_fresh
        )
        if cand_decision == "allow":
            w.say(f"{candidate}: {action_id} -- fresh re-evaluation ALLOW; executed as delegatee")
            return PracticalStepResult(w, "delegate", assur, True, executed_by=candidate)
        w.say(f"{candidate}: {action_id} -- fresh re-evaluation {cand_decision}; not executed")
    w.say(f"{action_id} -- no admissible delegatee found; action does not occur")
    return PracticalStepResult(w, "delegate", assur, False)
