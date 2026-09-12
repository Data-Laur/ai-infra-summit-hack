from __future__ import annotations

import argparse
import random
from typing import Dict, Any, List


def run_seed(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    params = {
        "seed": seed,
        "table_offset_x": rng.uniform(-0.05, 0.05),
        "table_offset_y": rng.uniform(-0.05, 0.05),
        "object_shift": rng.uniform(0.0, 0.03),
        "lighting": rng.uniform(0.9, 1.1),
        "background": rng.choice(["wood", "gray", "dark"]),
    }

    return {
        "seed": seed,
        "success": True,
        "params": params,
        "notes": "Starter harness only. Wire in the full scene pipeline when it is ready.",
    }


def run_suite(num_seeds: int = 10) -> List[Dict[str, Any]]:
    return [run_seed(seed) for seed in range(num_seeds)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run randomized evaluation seeds for the scene pipeline.")
    parser.add_argument("--num-seeds", type=int, default=10)
    args = parser.parse_args()

    results = run_suite(args.num_seeds)
    success_count = sum(1 for result in results if result["success"])
    print({
        "num_seeds": args.num_seeds,
        "success_count": success_count,
        "results": results,
    })


if __name__ == "__main__":
    main()
