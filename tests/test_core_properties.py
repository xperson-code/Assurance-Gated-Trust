"""Executable checks for Proposition "Sanity properties" (Sec. 4.5):
partition, trust-monotonicity, Fail => block, LLM allow => Pass, and the
ordinary-agent bypass used by the Decision-recovery theorem.

The factor domain (trust level x source_ok/trace_ok x assurance outcome) is
small and finite, so these tests are *exhaustive* over it rather than
randomly sampled -- a strictly stronger guarantee than random testing here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agt.core import FAIL, INC, LEVELS, PASS, Policy, VerifierSpec, World
from agt.decisions import decision_practical, mdecision

ORDER = {"block": 0, "review": 1, "delegate": 1, "allow": 2}


def _fixed(outcome: str) -> VerifierSpec:
    return VerifierSpec("fixed", lambda agent, action_id, world: outcome)


def _world(llm: bool) -> World:
    return World(llm_agents={"i"} if llm else set())


def _mental_policy() -> Policy:
    return Policy(tau_B_M="low", tau_A_M="high", tau_S="medium")


def _practical_policy() -> Policy:
    return Policy(tau_B_P="low", tau_A_P="high")


def test_mental_decision_is_a_total_partition():
    policy = _mental_policy()
    seen = set()
    for trust in LEVELS:
        for source_ok in (True, False):
            for assurance_outcome in (PASS, INC, FAIL):
                world = _world(llm=True)
                world.trust_M[("i", "a")] = trust
                world.trust_S[("s", "x")] = "very_high" if source_ok else "low"
                decision, _ = mdecision("i", "a", world, policy, [_fixed(assurance_outcome)], source_formula=("s", "x"))
                assert decision in {"allow", "review", "block"}
                seen.add(decision)
    assert seen == {"allow", "review", "block"}


def test_practical_decision_is_a_total_partition():
    policy = _practical_policy()
    seen = set()
    for trust in LEVELS:
        for trace_ok in (True, False):
            for assurance_outcome in (PASS, INC, FAIL):
                world = _world(llm=True)
                world.trust_P[("i", "a")] = trust
                decision, _ = decision_practical("i", "a", world, policy, [_fixed(assurance_outcome)], trace_ok)
                assert decision in {"allow", "delegate", "block"}
                seen.add(decision)
    assert seen == {"allow", "delegate", "block"}


def test_mental_decision_is_monotone_in_trust():
    policy = _mental_policy()
    for source_ok in (True, False):
        for assurance_outcome in (PASS, INC, FAIL):
            prev = None
            for trust in LEVELS:  # LEVELS is already ordered low -> very_high
                world = _world(llm=True)
                world.trust_M[("i", "a")] = trust
                world.trust_S[("s", "x")] = "very_high" if source_ok else "low"
                decision, _ = mdecision("i", "a", world, policy, [_fixed(assurance_outcome)], source_formula=("s", "x"))
                if prev is not None:
                    assert ORDER[decision] >= ORDER[prev], (
                        f"decision moved downward as trust increased: {prev} -> {decision} "
                        f"(source_ok={source_ok}, assurance={assurance_outcome})"
                    )
                prev = decision


def test_practical_decision_is_monotone_in_trust():
    policy = _practical_policy()
    for trace_ok in (True, False):
        for assurance_outcome in (PASS, INC, FAIL):
            prev = None
            for trust in LEVELS:
                world = _world(llm=True)
                world.trust_P[("i", "a")] = trust
                decision, _ = decision_practical("i", "a", world, policy, [_fixed(assurance_outcome)], trace_ok)
                if prev is not None:
                    assert ORDER[decision] >= ORDER[prev]
                prev = decision


def test_fail_blocks_regardless_of_trust():
    mpolicy, ppolicy = _mental_policy(), _practical_policy()
    for trust in LEVELS:
        world = _world(llm=True)
        world.trust_M[("i", "a")] = trust
        world.trust_S[("s", "x")] = "very_high"
        decision, assur = mdecision("i", "a", world, mpolicy, [_fixed(FAIL)], source_formula=("s", "x"))
        assert decision == "block" and assur == FAIL

        world2 = _world(llm=True)
        world2.trust_P[("i", "a")] = trust
        decision2, assur2 = decision_practical("i", "a", world2, ppolicy, [_fixed(FAIL)], True)
        assert decision2 == "block" and assur2 == FAIL


def test_llm_allow_implies_pass_certificate():
    mpolicy, ppolicy = _mental_policy(), _practical_policy()
    for assurance_outcome in (PASS, INC, FAIL):
        world = _world(llm=True)
        world.trust_M[("i", "a")] = "very_high"
        world.trust_S[("s", "x")] = "very_high"
        decision, assur = mdecision("i", "a", world, mpolicy, [_fixed(assurance_outcome)], source_formula=("s", "x"))
        if decision == "allow":
            assert assur == PASS

        world2 = _world(llm=True)
        world2.trust_P[("i", "a")] = "very_high"
        decision2, assur2 = decision_practical("i", "a", world2, ppolicy, [_fixed(assurance_outcome)], True)
        if decision2 == "allow":
            assert assur2 == PASS


def test_ordinary_agent_bypass_matches_trust_only_policy():
    """Decision-recovery theorem: for i not in Agt_LLM, assurance is Pass by
    convention regardless of verifier verdicts, so the Fail-blocks clause
    never fires and the decision reduces to the trust-only policy."""
    mpolicy, ppolicy = _mental_policy(), _practical_policy()
    for assurance_outcome in (PASS, INC, FAIL):
        for trust in LEVELS:
            world = _world(llm=False)  # "i" not in llm_agents
            world.trust_M[("i", "a")] = trust
            world.trust_S[("s", "x")] = "very_high"
            decision, assur = mdecision("i", "a", world, mpolicy, [_fixed(assurance_outcome)], source_formula=("s", "x"))
            assert assur == PASS
            expected = "allow" if trust in ("high", "very_high") else ("block" if trust == "low" else "review")
            assert decision == expected
