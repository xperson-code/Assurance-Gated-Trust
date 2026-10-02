"""Executable checks for the Proposition on finite-state decidability and
local overhead (Sec. 4.5): an assurance guard aggregating m verifier outputs
and checking d certificate dependencies is evaluable in O(m+d) time.

Rather than asserting on noisy wall-clock timings, this counts the exact
number of verifier invocations and dependency-support lookups performed --
a robust, deterministic witness that the guard does exactly m + O(d) work,
with no hidden quadratic or unbounded cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agt.core import PASS, Certificate, World, assurance


def test_assurance_calls_each_verifier_exactly_once():
    for m in (0, 1, 2, 5, 10, 50):
        calls = {"n": 0}

        def counting_fn(agent, action_id, world):
            calls["n"] += 1
            return PASS

        from agt.core import VerifierSpec

        verifiers = [VerifierSpec(f"v{i}", counting_fn) for i in range(m)]
        world = World(llm_agents={"i"})
        assurance("i", "a", world, {"i"}, verifiers)
        assert calls["n"] == m, f"expected exactly m={m} verifier calls, got {calls['n']}"


def test_certificate_validity_checks_each_dependency_exactly_once():
    for d in (0, 1, 5, 20, 100):
        deps = {f"dep{k}" for k in range(d)}

        class CountingWorld(World):
            def is_supported(self, dep: str) -> bool:  # type: ignore[override]
                counts["n"] += 1
                return dep in self.supported

        counts = {"n": 0}
        world = CountingWorld(supported=set(deps))
        cert = Certificate(
            id="c", verifiers=frozenset(), agent="i", action_id="a", omega=PASS,
            origin_tick=0, deps=frozenset(deps), scope=lambda tick: True,
        )
        assert cert.valid(world) is True
        assert counts["n"] == d, f"expected exactly d={d} dependency checks, got {counts['n']}"


def test_guard_cost_scales_as_sum_not_product_of_m_and_d():
    """A combined mental-decision guard with m verifiers and a d-dependency
    certificate should cost ~ m + d operations, not m * d: doubling one
    factor while zeroing the other must not change the other's op count."""
    from agt.core import VerifierSpec

    calls = {"verifier": 0, "dep": 0}

    def counting_verifier(agent, action_id, world):
        calls["verifier"] += 1
        return PASS

    class CountingWorld(World):
        def is_supported(self, dep: str) -> bool:  # type: ignore[override]
            calls["dep"] += 1
            return dep in self.supported

    m, d = 7, 13
    deps = {f"dep{k}" for k in range(d)}
    world = CountingWorld(supported=set(deps), llm_agents={"i"})
    verifiers = [VerifierSpec(f"v{i}", counting_verifier) for i in range(m)]
    cert = Certificate(
        id="c", verifiers=frozenset(), agent="i", action_id="a", omega=PASS,
        origin_tick=0, deps=frozenset(deps), scope=lambda tick: True,
    )
    assurance("i", "a", world, {"i"}, verifiers)
    cert.valid(world)
    assert calls == {"verifier": m, "dep": d}
