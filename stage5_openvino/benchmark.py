"""Stage 5: OpenVINO model conversion + benchmark on Intel hardware (stub)."""

# TODO(Lauren): replace stub with real implementation

from common.types import BenchmarkResult


def run_benchmark() -> BenchmarkResult:
    """Return fake benchmark numbers; real version converts the model and runs on Intel HW."""
    return BenchmarkResult(
        model_name="so101_policy (stub)",
        device="intel_core_ultra (stub)",
        precision="FP16 (stub)",
        latency_ms=42.7,
        throughput=23.4,
    )
