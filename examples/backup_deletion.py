"""Scripted replay of the "Transfer beyond healthcare" paragraph (Sec. 6):
an irreversible administrative action governed by the same lifecycle,
with no healthcare-specific concepts.

Agents: l_admin (LLM administrative agent, in Agt_LLM), admin (ordinary
system administrator). Action: a_D = deleteBackup(b), impact = irreversible
=> critical by the impact-forced-inclusion rule (Sec. 4.3).

Run directly for a narrated console trace:
    python -m examples.backup_deletion
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agt.core import VerifierSpec, Policy, World, PASS, INC, FAIL
from agt.decisions import TraceItem
from agt.engine import mental_step, retract_dependency, practical_step


def _always(outcome: str) -> VerifierSpec:
    return VerifierSpec(name=f"mock:{outcome}", fn=lambda agent, action_id, world: outcome, sound=True)


@dataclass
class BackupResult:
    steps: List[dict]


def build_policy() -> Policy:
    return Policy(
        tau_B_M="low", tau_A_M="high",
        tau_B_P="low", tau_A_P="high",
        tau_S="medium",
        impact={"deleteBackup:b": "irreversible"},
    )


def run(verbose: bool = True) -> BackupResult:
    policy = build_policy()
    assert policy.is_critical("deleteBackup:b"), "irreversible impact must force Crit_Pi membership"

    steps: List[dict] = []

    def record(label: str, world: World, **data) -> None:
        steps.append({"label": label, **data})
        if verbose:
            print(f"\n== {label} ==")
            for line in world.log[-2:]:
                print(" ", line)
            for k, v in data.items():
                print(f"   {k} = {v}")

    # --- Scenario A: Inconclusive assurance routes to the ordinary administrator ---
    world = World(llm_agents={"l_admin"})
    world.supported.add("state:b")
    world.trust_M[("l_admin", "confirm_redundancy")] = "very_high"
    world.trust_P[("l_admin", "deleteBackup:b")] = "very_high"
    world.trust_P[("admin", "deleteBackup:b")] = "very_high"

    r = mental_step(world, "l_admin", ("l_admin",), "admit", ("state_confirmed",), "confirm_redundancy",
                     policy, [_always(PASS)], deps=("state:b",))
    world = r.world
    assert r.decision == "allow"
    trace = [TraceItem(kind="autonomous", certificate_id=r.certificate_id)]
    record("l_admin confirms backup redundancy (Pass)", world, certificate=r.certificate_id)

    r2 = practical_step(world, "l_admin", "deleteBackup:b", policy, [_always(INC)], trace,
                         trigger_supported=True, delegation_candidates_pool=["admin"], verifiers_by_agent={"admin": []})
    world = r2.world
    record("Propose deleteBackup(b) with Inc assurance", world, decision=r2.decision,
           executed=r2.executed, executed_by=r2.executed_by)
    assert r2.decision == "delegate" and r2.executed and r2.executed_by == "admin"

    # --- Scenario B: a Fail verdict blocks regardless of trust level ---
    world_b = World(llm_agents={"l_admin"})
    world_b.supported.add("state:b")
    world_b.trust_M[("l_admin", "confirm_redundancy")] = "very_high"
    world_b.trust_P[("l_admin", "deleteBackup:b")] = "very_high"
    rb = mental_step(world_b, "l_admin", ("l_admin",), "admit", ("state_confirmed",), "confirm_redundancy",
                      policy, [_always(PASS)], deps=("state:b",))
    world_b = rb.world
    trace_b = [TraceItem(kind="autonomous", certificate_id=rb.certificate_id)]
    r_fail = practical_step(world_b, "l_admin", "deleteBackup:b", policy, [_always(FAIL)], trace_b,
                             trigger_supported=True, delegation_candidates_pool=["admin"], verifiers_by_agent={"admin": []})
    world_b = r_fail.world
    record("Propose deleteBackup(b) with Fail assurance (very_high trust)", world_b,
           decision=r_fail.decision, executed=r_fail.executed)
    assert r_fail.decision == "block" and not r_fail.executed, "Fail must block even at very_high trust"

    # --- Scenario A continued: revoking the backup-state evidence invalidates the trace ---
    world = retract_dependency(world, "state:b")
    record("Backup-state evidence revoked", world, supported=sorted(world.supported))

    r3 = practical_step(world, "l_admin", "deleteBackup:b", policy, [_always(INC)], trace,
                         trigger_supported=True, delegation_candidates_pool=["admin"], verifiers_by_agent={"admin": []})
    world = r3.world
    record("Re-attempt deleteBackup(b) after revocation", world, decision=r3.decision, executed=r3.executed)
    assert not r3.executed, "revoked evidence must invalidate the trace before any later authorization"

    return BackupResult(steps=steps)


if __name__ == "__main__":
    result = run(verbose=True)
    print("\nAll backup-deletion assertions hold: Inc->delegate->fresh-allow, Fail->block regardless of trust, "
          "and evidence revocation invalidates the trace before re-authorization.")
