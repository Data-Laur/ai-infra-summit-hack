# stage7_eval — owner: Abdullah
Domain-randomization harness: runs the full pipeline (stages 1-4 + 6) once per seed and reports success rate (stub: no actual randomization yet).
Input: `seeds: list[int]`. Output: `EvalReport` (seeds, successes, success_rate, per-seed results).
Run standalone: `python3 -c "from stage7_eval import evaluate; print(evaluate([0, 1, 2]).model_dump_json(indent=2))"`
