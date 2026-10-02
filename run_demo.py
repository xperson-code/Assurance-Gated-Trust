#!/usr/bin/env python3
"""Single entry point for the Assurance-Gated Trust demonstration.

    python run_demo.py examples      # narrated replay of the Sec. 6 case studies
    python run_demo.py stress [--n N]  # randomized semantic-engine stress test (Sec. 4.5 / Theorem checks)
    python run_demo.py evaluation     # Section 5 Table 2 reproduction (prototype.py, packaged)
    python run_demo.py all            # everything above, in sequence

Only the Python standard library is required for the demo itself (pytest is
only needed to run tests/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_examples() -> None:
    from examples import backup_deletion, healthcare_case_study

    print("#" * 78)
    print("# Sec. 6 narrative 1/2: healthcare decision-support case study")
    print("#" * 78)
    healthcare_case_study.run(verbose=True)

    print("\n" + "#" * 78)
    print("# Sec. 6 narrative 2/2: 'Transfer beyond healthcare' -- deleteBackup(b)")
    print("#" * 78)
    backup_deletion.run(verbose=True)


def run_stress(n: int) -> None:
    from stress import random_trajectories

    random_trajectories.main(n=n)


def run_evaluation() -> None:
    from evaluation import synthetic_workload

    synthetic_workload.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["examples", "stress", "evaluation", "all"])
    parser.add_argument("--n", type=int, default=20_000, help="trajectories per stress-test configuration")
    args = parser.parse_args()

    if args.mode in ("examples", "all"):
        run_examples()
    if args.mode in ("stress", "all"):
        print("\n" + "#" * 78)
        print("# Randomized semantic-engine stress test")
        print("#" * 78)
        run_stress(args.n)
    if args.mode in ("evaluation", "all"):
        print("\n" + "#" * 78)
        print("# Section 5 evaluation: Table 2 / ablations / sensitivity reproduction")
        print("#" * 78)
        run_evaluation()


if __name__ == "__main__":
    main()
