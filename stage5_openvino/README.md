# stage5_openvino — owner: Lauren
Converts the model to OpenVINO and benchmarks it on Intel hardware (currently a stub with fake numbers). Standalone — not part of the run_pipeline chain.
Input: none (real version reads model from `configs/default.yaml`).
Output: `BenchmarkResult` (`model_name`, `device`, `precision`, `latency_ms`, `throughput`).
Test: `python3 -c "from stage5_openvino import run_benchmark; print(run_benchmark())"`
