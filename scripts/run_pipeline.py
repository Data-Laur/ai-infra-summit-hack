#!/usr/bin/env python3
"""Run the full pipeline end-to-end: voice -> perception -> policy -> execute -> verify.

Works today with stubs: python scripts/run_pipeline.py [--command "..."] [--seed 3]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import EXAMPLE_COMMAND
from common.pipeline import run_once


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the table-setting pipeline end-to-end.")
    parser.add_argument("--command", default=EXAMPLE_COMMAND, help="Natural-language command.")
    parser.add_argument("--seed", type=int, default=0, help="Randomization seed for reset_scene.")
    args = parser.parse_args()

    print("=" * 72)
    print("PIPELINE  voice -> perception -> policy -> bimanual -> verify")
    print("=" * 72)

    result = run_once(args.command, seed=args.seed)
    for line in result.log:
        print(line)

    if result.success:
        print("\nRESULT: SUCCESS")
        return 0
    print(f"\nRESULT: FAIL (after {result.attempts} attempts)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
