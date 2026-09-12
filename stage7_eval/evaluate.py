"""Stage 7: domain-randomization evaluation harness.

Runs the full pipeline across N seeds and reports aggregate success rate.
"""

# TODO(Abdullah): replace stub with real implementation

from common import EXAMPLE_COMMAND
from common.pipeline import run_once
from common.types import EvalReport, SeedResult


def evaluate(seeds: list[int]) -> EvalReport:
    """Run the full pipeline once per seed via common.pipeline.run_once."""
    results: list[SeedResult] = []
    successes = 0
    for seed in seeds:
        try:
            run = run_once(EXAMPLE_COMMAND, seed=seed)
            success = run.success
            details = f"{run.attempts} attempt(s)"
        except Exception as exc:  # a stage blew up: count as failure
            success = False
            details = f"pipeline error: {exc}"
        successes += success
        results.append(SeedResult(seed=seed, success=success, details=details))

    return EvalReport(
        seeds=seeds,
        successes=successes,
        success_rate=successes / len(seeds) if seeds else 0.0,
        results=results,
    )
