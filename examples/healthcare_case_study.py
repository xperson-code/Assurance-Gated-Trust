"""Scripted replay of Sec. 6's healthcare decision-support narrative.

    source admit -> LLM review -> commit -> delegate -> authorize
    ... later: retraction -> stale trace -> no new authorization

Agents: s_L (lab source), s_V (monitor source), l (LLM clinical-support
agent, in Agt_LLM), h (physician, ordinary agent). This is a deliberately
small, illustrative instantiation (as the paper states) -- not a validated
clinical workflow.

Run directly for a narrated console trace:
    python -m examples.healthcare_case_study
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agt.core import VerifierSpec, Policy, World, PASS, INC, FAIL
from agt.decisions import TraceItem
from agt.engine import mental_step, validate_review, retract_dependency, practical_step


def _always(outcome: str) -> VerifierSpec:
    return VerifierSpec(name=f"mock:{outcome}", fn=lambda agent, action_id, world: outcome, sound=True)


@dataclass
class CaseStudyResult:
    world: World
    steps: List[dict]


def build_policy() -> Policy:
    return Policy(
        tau_B_M="low", tau_A_M="high",
        tau_B_P="low", tau_A_P="high",
        tau_S="medium",
        impact={"transfer_higher_care": "irreversible"},
    )


def run(verbose: bool = True) -> CaseStudyResult:
    policy = build_policy()
    world = World(llm_agents={"l"})
    world.supported |= {"lab:p", "monitor:v"}

    # Source trust (s_L, s_V) and task-sensitive trust for l (LLM) and h (physician).
    world.trust_S[("s_L", "p")] = "very_high"
    world.trust_S[("s_V", "v")] = "very_high"
    world.trust_M[("l", "admit_p")] = "very_high"
    world.trust_M[("l", "admit_v")] = "very_high"
    world.trust_M[("l", "beta_pv")] = "very_high"
    world.trust_M[("l", "alpha_q")] = "very_high"
    world.trust_M[("h", "alpha_q")] = "high"  # exactly at tau_A_M: a sufficiently trusted reviewer
    world.trust_P[("l", "transfer_higher_care")] = "very_high"
    world.trust_P[("h", "transfer_higher_care")] = "very_high"

    # K_l((p & v) -> q): the background knowledge licensing the down(.) step.
    world.know("l").add(("p&v", "q"))

    steps: List[dict] = []

    def record(label: str, **data) -> None:
        steps.append({"label": label, **data})
        if verbose:
            print(f"\n== {label} ==")
            for line in world.log[-3:]:
                print(" ", line)
            for k, v in data.items():
                print(f"   {k} = {v}")

    # 1-2. Source-sensitive admissions +p, +v (routine provenance check -> Pass).
    r = mental_step(world, "l", ("l",), "admit", ("p",), "admit_p", policy,
                     [_always(PASS)], source_formula=("s_L", "p"), deps=("lab:p",))
    world = r.world
    record("Admit +p from lab s_L", decision=r.decision, assurance=r.assurance)

    r = mental_step(world, "l", ("l",), "admit", ("v",), "admit_v", policy,
                     [_always(PASS)], source_formula=("s_V", "v"), deps=("monitor:v",))
    world = r.world
    record("Admit +v from monitor s_V", decision=r.decision, assurance=r.assurance)

    # 3. beta = cap(p, v): two independent verifiers agree -> Pass -> autonomous allow.
    r = mental_step(world, "l", ("l",), "infercon", ("p", "v"), "beta_pv", policy,
                     [_always(PASS), _always(PASS)], deps=("lab:p", "monitor:v"))
    world = r.world
    beta_cert = r.certificate_id
    record("beta = p AND v (two sound verifiers agree)", decision=r.decision, assurance=r.assurance,
           certificate=beta_cert, belief_l=sorted(world.belief("l")))
    assert r.decision == "allow" and r.assurance == PASS

    # 4. alpha = down(p&v, q): very_high trust but Inconclusive assurance -> REVIEW, not allow.
    r = mental_step(world, "l", ("l",), "down", ("p&v", "q"), "alpha_q", policy,
                     [_always(INC)], deps=("lab:p", "monitor:v"))
    world = r.world
    pending_index = r.pending_index
    record("alpha = down(p&v, q): very_high trust, Inc assurance", decision=r.decision, assurance=r.assurance,
           belief_l=sorted(world.belief("l")))
    assert r.decision == "review"
    assert "q" not in world.belief("l"), "an Inconclusive LLM result must not silently enter ordinary belief"

    # 5. Physician h validates the pending step -> commit c_q with deps {p, v}.
    r2 = validate_review(world, pending_index, "h", policy, [], deps=("lab:p", "monitor:v"), conclusion="q")
    world = r2.world
    commit_id = r2.commit_id
    record("Physician h validates alpha", success=r2.success, commit=commit_id, belief_l=sorted(world.belief("l")))
    assert r2.success and "q" in world.belief("l")

    # 6. Practical proposal a_P = transfer_higher_care by l: Inc assurance -> delegate; h re-evaluated fresh -> allow.
    trace = [TraceItem(kind="reviewed", commit_id=commit_id)]
    r3 = practical_step(
        world, "l", "transfer_higher_care", policy, [_always(INC)], trace,
        trigger_supported=True, delegation_candidates_pool=["h"], verifiers_by_agent={"h": []},
    )
    world = r3.world
    record("Propose transfer_higher_care (Inc assurance for l)", decision=r3.decision,
           executed=r3.executed, executed_by=r3.executed_by)
    assert r3.decision == "delegate" and r3.executed and r3.executed_by == "h"

    # 7. Lab retracts p: the commit's dependency is no longer supported -> stale.
    world = retract_dependency(world, "lab:p")
    record("Lab retracts p", supported=sorted(world.supported))
    assert not world.commits[commit_id].valid(world), "the commit must become stale once p is retracted"

    # 8. A later authorization attempt on the same (now stale) trace must fail closed.
    r4 = practical_step(
        world, "l", "transfer_higher_care", policy, [_always(INC)], trace,
        trigger_supported=True, delegation_candidates_pool=["h"], verifiers_by_agent={"h": []},
    )
    world = r4.world
    record("Re-attempt transfer_higher_care after retraction", decision=r4.decision, executed=r4.executed)
    assert not r4.executed, "a stale trace must not authorize a new critical action"

    return CaseStudyResult(world=world, steps=steps)


if __name__ == "__main__":
    result = run(verbose=True)
    print("\nAll Sec. 6 narrative assertions hold: source admit -> LLM review -> commit -> delegate -> "
          "authorize, with later trace invalidation after retraction.")
