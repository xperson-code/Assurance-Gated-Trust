"""The decision layer: source conditions, mental/practical decisions,
reviewer admissibility, trusted justification traces.

Paper correspondence
---------------------
- ``source_ok``                 -> Sec. 4.2, source_OK(K, Pi, +chi, w)
- ``mdecision``                 -> Sec. 4.2, m_decision_{K,Pi}(i, alpha, w) (Eq. in Sec. 4.2)
- ``reviewer_ok``                -> Sec. 4.2, reviewer_OK_{K,Pi}(r, i, alpha, w)
- ``TraceItem`` / ``validtrace`` / ``traceok`` -> Sec. 4.3, ValidTrace_{K,Pi}(J, a, w) and trace_OK
- ``decision_practical``        -> Sec. 4.3, decision_{K,Pi}(i, a, w)
- ``delegation_candidates``     -> Sec. 4.3, DelCand_{K,Pi}(i, a, w)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .core import FAIL, INC, PASS, Commit, Policy, VerifierSpec, World, assurance, geT, leT


def source_ok(world: World, policy: Policy, source_formula: Optional[Tuple[str, str]]) -> bool:
    """source_OK(K, Pi, +chi, w); true vacuously when no external source is involved."""
    if source_formula is None:
        return True
    source, formula = source_formula
    level = world.trust_S.get((source, formula), "low")
    return geT(level, policy.tau_S)


def mdecision(
    agent: str,
    action_id: str,
    world: World,
    policy: Policy,
    verifiers: List[VerifierSpec],
    source_formula: Optional[Tuple[str, str]] = None,
) -> Tuple[str, str]:
    """m_decision_{K,Pi}(i, alpha, w) -> (decision, assurance outcome).

    decision in {"allow", "review", "block"}; for an ordinary agent the
    assurance outcome is always Pass by convention (Sec. 4.1), so the
    "A_alpha = Fail" disjunct in the block clause never fires and the
    function reduces exactly to the trust-only policy of Sec. 3.
    """
    s_ok = source_ok(world, policy, source_formula)
    t = world.trust_M.get((agent, action_id), "low")
    a_out = assurance(agent, action_id, world, world.llm_agents, verifiers)
    if (not s_ok) or leT(t, policy.tau_B_M) or a_out == FAIL:
        return "block", a_out
    if geT(t, policy.tau_A_M) and a_out == PASS:
        return "allow", a_out
    return "review", a_out


def reviewer_ok(
    reviewer: str,
    performer: str,
    action_id: str,
    world: World,
    policy: Policy,
    reviewer_check_verifiers: List[VerifierSpec],
) -> bool:
    """reviewer_OK_{K,Pi}(r, i, alpha, w): a distinct, sufficiently trusted
    reviewer, with an independent Pass if the reviewer is itself an LLM."""
    if reviewer == performer:
        return False
    t = world.trust_M.get((reviewer, action_id), "low")
    if not geT(t, policy.tau_A_M):
        return False
    if reviewer not in world.llm_agents:
        return True
    check_action_id = f"check:{performer}:{action_id}"
    return assurance(reviewer, check_action_id, world, world.llm_agents, reviewer_check_verifiers) == PASS


@dataclass(frozen=True)
class TraceItem:
    """One evidence record e_k of a trusted justification trace J.

    ``kind`` is "autonomous" (an accepted mental step backed by currently
    valid assurance/source evidence, carried via a Certificate) or
    "reviewed" (backed by a currently valid Commit).
    """

    kind: str  # "autonomous" | "reviewed"
    certificate_id: Optional[str] = None
    commit_id: Optional[str] = None


def validtrace(
    items: List[TraceItem],
    world: World,
    policy: Policy,
    *,
    enforce_lifecycle_guard: bool = True,
    enforce_conflict_guard: bool = True,
) -> bool:
    """ValidTrace_{K,Pi}(J, a, w): every certificate/commit used is currently
    valid, and any conflict set among the trace's commits is resolved
    (|C| > 1 => resolve_Pi(C, w) != bot).

    The two ``enforce_*`` flags exist only to run the paper's own ablations
    (Sec. 5, "disabling certificate lifecycle invalidation" / conflict
    guard) *through the real decision functions* rather than a separate
    statistical model; both default to True, i.e. the full semantics.
    """
    commits_in_trace: List[Commit] = []
    for item in items:
        if item.kind == "autonomous":
            cert = world.certificates.get(item.certificate_id or "")
            if cert is None:
                return False
            if enforce_lifecycle_guard and not cert.valid(world):
                return False
        elif item.kind == "reviewed":
            commit = world.commits.get(item.commit_id or "")
            if commit is None:
                return False
            if enforce_lifecycle_guard and not commit.valid(world):
                return False
            commits_in_trace.append(commit)
        else:
            raise ValueError(f"unknown trace item kind: {item.kind}")

    if not enforce_conflict_guard:
        return True

    from .core import conflict  # local import to avoid a module cycle at import time

    conflicting: List[Commit] = []
    for c in commits_in_trace:
        for c2 in world.commits.values():
            if c2 is not c and conflict(c, c2, world) and c2 not in conflicting:
                conflicting.append(c2)
    if conflicting:
        conflict_set = commits_in_trace + [c for c in conflicting if c not in commits_in_trace]
        if len(conflict_set) > 1 and policy.resolver(conflict_set, world) is None:
            return False
    return True


def traceok(
    action_id: str,
    trigger_supported: bool,
    items: List[TraceItem],
    world: World,
    policy: Policy,
    *,
    enforce_lifecycle_guard: bool = True,
    enforce_conflict_guard: bool = True,
) -> bool:
    """trace_OK_{K,Pi}(a, w): vacuously true for non-critical actions,
    otherwise requires a currently valid trace with a supported trigger."""
    if not policy.is_critical(action_id):
        return True
    return trigger_supported and validtrace(
        items, world, policy,
        enforce_lifecycle_guard=enforce_lifecycle_guard,
        enforce_conflict_guard=enforce_conflict_guard,
    )


def decision_practical(
    agent: str,
    action_id: str,
    world: World,
    policy: Policy,
    verifiers: List[VerifierSpec],
    trace_ok: bool,
) -> Tuple[str, str]:
    """decision_{K,Pi}(i, a, w) -> (decision, assurance outcome)."""
    t = world.trust_P.get((agent, action_id), "low")
    a_out = assurance(agent, action_id, world, world.llm_agents, verifiers)
    if leT(t, policy.tau_B_P) or (not trace_ok) or a_out == FAIL:
        return "block", a_out
    if geT(t, policy.tau_A_P) and a_out == PASS:
        return "allow", a_out
    return "delegate", a_out


def delegation_candidates(
    candidates: List[str],
    action_id: str,
    world: World,
    policy: Policy,
    verifiers_by_agent: Dict[str, List[VerifierSpec]],
    trace_ok: bool,
) -> List[str]:
    """DelCand_{K,Pi}(i, a, w): role-admissible agents that would themselves
    reach "allow" under the *current* guards (fresh re-evaluation, Sec. 4.3)."""
    admissible = []
    for j in candidates:
        decision, _ = decision_practical(j, action_id, world, policy, verifiers_by_agent.get(j, []), trace_ok)
        if decision == "allow":
            admissible.append(j)
    return admissible
