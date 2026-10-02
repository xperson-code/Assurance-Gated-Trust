"""Runs the two Sec. 6 scripted narratives end-to-end. Both ``run()``
functions already assert, inline, every outcome the paper's narrative
claims (Inc -> review -> commit -> delegate -> allow; Fail -> block
regardless of trust; retraction/revocation -> stale trace -> no new
authorization) -- so a clean run with no AssertionError *is* the check.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_healthcare_case_study_matches_sec6_narrative():
    from examples.healthcare_case_study import run

    result = run(verbose=False)
    assert len(result.steps) == 8
    assert "q" in result.world.belief("l")


def test_backup_deletion_matches_transfer_paragraph():
    from examples.backup_deletion import run

    result = run(verbose=False)
    assert len(result.steps) == 5
