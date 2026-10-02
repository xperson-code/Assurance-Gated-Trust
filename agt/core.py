"""Core semantic objects: trust levels, assurance outcomes, certificates,
commits, and the world/policy records that the decision layer (decisions.py)
and the transition engine (engine.py) operate on.

Paper correspondence
---------------------
- ``Levels``, ``leT``/``ltT``/``geT``/``gtT``                 -> Sec. 3, totally ordered Levels and <=_T, <_T, >=_T, >_T
- ``PASS, INC, FAIL`` / ``aggregate``                          -> Sec. 4.1, Omega = {Pass, Inconclusive, Fail} and Agg(S)
- ``Certificate`` / ``Certificate.valid``                      -> Sec. 4.1, kappa = <id, V, i, xi, omega, w0, D, S, status> and cert_valid
- ``Commit`` / ``Commit.valid``                                -> Sec. 4.1, commit record and cmt_valid
- ``conflict`` / ``resolve_conservative`` / ``resolve_by_priority`` -> Sec. 4.1, conflict(c, c', w) and resolve_Pi(C, w)
- ``World``                                                    -> a concrete possible world w (beliefs, Rvw, Cmt, supported deps, trust tables)
- ``Policy``                                                   -> policy authority Pi (thresholds, impact map, Crit_Pi, resolver)
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

# --- Sec. 3: finite totally ordered set of trust Levels -------------------

LEVELS: Tuple[str, ...] = ("low", "medium", "high", "very_high")
RANK: Dict[str, int] = {level: i for i, level in enumerate(LEVELS)}


def leT(a: str, b: str) -> bool:
    return RANK[a] <= RANK[b]


def ltT(a: str, b: str) -> bool:
    return RANK[a] < RANK[b]


def geT(a: str, b: str) -> bool:
    return RANK[a] >= RANK[b]


def gtT(a: str, b: str) -> bool:
    return RANK[a] > RANK[b]


# --- Sec. 4.1: assurance outcomes Omega = {Pass, Inconclusive, Fail} -------

PASS, INC, FAIL = "pass", "inconclusive", "fail"
OMEGA = (PASS, INC, FAIL)


def aggregate(outputs: List[str]) -> str:
    """Agg(S) from Sec. 4.1: conservative disagreement resolution.

    Fail if any result is Fail; Pass iff the (non-empty) set is unanimously
    Pass; Inconclusive otherwise -- including the no-verifier case, matching
    the paper's "no authorized independent verifier -> Inc" convention.
    """
    if FAIL in outputs:
        return FAIL
    if outputs and all(o == PASS for o in outputs):
        return PASS
    return INC


Verifier = Callable[[str, str, "World"], str]
"""A policy-approved verifier: (agent, action_id, world) -> outcome in Omega.

The independence predicate indep(v, i) is enforced by construction: a
verifier function must not consult the very mental/practical step it
checks beyond what callers pass it, and callers never attach an agent's own
verifier to its own output. ``VerifierSpec.sound`` records whether the
verifier is Psi-sound (assumption B2), used only for the stress-test
bookkeeping in stress/random_trajectories.py.
"""


@dataclass(frozen=True)
class VerifierSpec:
    name: str
    fn: Verifier
    sound: bool = True  # whether this verifier instance satisfies Sec.4.3's Psi-soundness premise (B2)

    def __call__(self, agent: str, action_id: str, world: "World") -> str:
        return self.fn(agent, action_id, world)


def assurance(
    agent: str,
    action_id: str,
    world: "World",
    llm_agents: Set[str],
    verifiers: List[VerifierSpec],
) -> str:
    """assur^X_Pi(i, xi, w) from Sec. 4.1.

    Ordinary agents (i in Agt_ord) bypass the layer: Pass is the embedding
    default, representing the *absence* of an assurance obligation rather
    than a hidden verification outcome (Table 2 / Def. 1).
    """
    if agent not in llm_agents:
        return PASS
    only_valid = [v for v in verifiers]  # independence assumed by construction (see Verifier docstring)
    outputs = [v(agent, action_id, world) for v in only_valid]
    return aggregate(outputs)


# --- Sec. 4.1: certificates and commits ------------------------------------


@dataclass
class Certificate:
    """kappa = <id, V, i, xi, omega, w0, D, S, status>."""

    id: str
    verifiers: FrozenSet[str]
    agent: str
    action_id: str
    omega: str
    origin_tick: int
    deps: FrozenSet[str]
    scope: Callable[[int], bool]  # S subseteq W, represented as a predicate on world ticks
    status: str = "active"  # "active" | "revoked"

    def valid(self, world: "World") -> bool:
        """cert_valid(kappa, w): status active, w in S, and every dependency supported."""
        return (
            self.status == "active"
            and self.scope(world.tick)
            and all(world.is_supported(d) for d in self.deps)
        )

    def revoke(self) -> None:
        self.status = "revoked"


@dataclass
class Commit:
    """A persistent, revocable commit certificate produced by review/validation."""

    id: str
    reviewer: str
    performer: str
    action_id: str
    conclusion: str  # a propositional atom; "~x" denotes its negation for conflict detection
    deps: FrozenSet[str]
    scope: Callable[[int], bool]
    status: str = "active"

    def valid(self, world: "World") -> bool:
        """cmt_valid(c, w): status active, w in S_c, and every dependency in D_c supported."""
        return (
            self.status == "active"
            and self.scope(world.tick)
            and all(world.is_supported(d) for d in self.deps)
        )

    def revoke(self) -> None:
        self.status = "revoked"


def _negate(atom: str) -> str:
    return atom[1:] if atom.startswith("~") else "~" + atom


def conflict(c1: Commit, c2: Commit, world: "World") -> bool:
    """conflict(c, c', w): both currently valid and logically incompatible conclusions."""
    if c1.id == c2.id:
        return False
    return c1.valid(world) and c2.valid(world) and c1.conclusion == _negate(c2.conclusion)


Resolver = Callable[[List[Commit], "World"], Optional[Commit]]


def resolve_conservative(candidates: List[Commit], world: "World") -> Optional[Commit]:
    """Conservative default resolve_Pi(C, w) = bot (Sec. 4.1)."""
    return None


def resolve_by_priority(priority: Dict[str, int]) -> Resolver:
    """A deterministic precedence resolver keyed by reviewer-role priority.

    Returns bot (None) on a tie, matching the paper's requirement that a
    resolver either picks a unique winner or reports an unresolved conflict.
    """

    def _resolve(candidates: List[Commit], world: "World") -> Optional[Commit]:
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda c: priority.get(c.reviewer, -1), reverse=True)
        if len(ranked) > 1 and priority.get(ranked[0].reviewer, -1) == priority.get(ranked[1].reviewer, -1):
            return None
        return ranked[0]

    return _resolve


# --- pending (reviewed, not yet committed) mental steps --------------------


@dataclass(frozen=True)
class PendingReview:
    """An element of Rvw(i, w): a reviewed-but-not-committed mental step."""

    group: Tuple[str, ...]
    performer: str
    action_id: str
    mental_op: str
    args: Tuple[str, ...]
    created_tick: int


# --- Sec. 3-4: the policy authority Pi and a concrete world w --------------


@dataclass
class Policy:
    """Policy authority Pi: thresholds, impact classification, and resolver."""

    tau_B_M: str = "low"
    tau_A_M: str = "high"
    tau_B_P: str = "low"
    tau_A_P: str = "high"
    tau_S: str = "medium"
    impact: Dict[str, str] = field(default_factory=dict)  # action_id -> low|high|irreversible
    crit: Set[str] = field(default_factory=set)  # explicitly policy-designated critical actions
    resolver: Resolver = resolve_conservative

    def is_critical(self, action_id: str) -> bool:
        """Crit_Pi(w) membership, including the impact-forced inclusion rule
        (Sec. 4.2): high/irreversible impact actions cannot be routine."""
        return action_id in self.crit or self.impact.get(action_id) in ("high", "irreversible")


@dataclass
class World:
    """A concrete possible world w carrying the mutable control-relevant state.

    Beliefs/knowledge are deliberately minimal (sets of propositional atoms)
    -- enough to exercise the mental-action grammar's preconditions and the
    control-semantics decisions that are this paper's actual contribution,
    not a full L-DINF epistemic model.
    """

    tick: int = 0
    beliefs: Dict[str, Set[str]] = field(default_factory=dict)
    knowledge: Dict[str, Set[Tuple[str, str]]] = field(default_factory=dict)  # agent -> {(chi, psi)} known chi->psi
    supported: Set[str] = field(default_factory=set)  # dependency ids not retracted/revoked
    review_store: List[PendingReview] = field(default_factory=list)
    certificates: Dict[str, Certificate] = field(default_factory=dict)
    commits: Dict[str, Commit] = field(default_factory=dict)
    trust_M: Dict[Tuple[str, str], str] = field(default_factory=dict)
    trust_P: Dict[Tuple[str, str], str] = field(default_factory=dict)
    trust_S: Dict[Tuple[str, str], str] = field(default_factory=dict)
    budgets: Dict[str, float] = field(default_factory=dict)
    llm_agents: Set[str] = field(default_factory=set)
    log: List[str] = field(default_factory=list)

    # -- helpers --
    def is_supported(self, dep: str) -> bool:
        return dep in self.supported

    def belief(self, agent: str) -> Set[str]:
        return self.beliefs.setdefault(agent, set())

    def know(self, agent: str) -> Set[Tuple[str, str]]:
        return self.knowledge.setdefault(agent, set())

    def clone(self, advance_tick: bool = True) -> "World":
        """Functional-update style successor state w -> w'."""
        new = deepcopy(self)
        if advance_tick:
            new.tick += 1
        return new

    def say(self, message: str) -> None:
        self.log.append(f"[t={self.tick}] {message}")
