# stage1_voice — Owner: Alex
Parses a natural-language command into a structured `Task` (Speechmatics ASR -> Claude parser; stubbed).
Input: text string (`parse_text`) or audio file path (`parse_command`, not implemented yet).
Output: `common.types.Task` (steps, dependencies, constraints).
Test: `python3 -c "from stage1_voice import parse_text; print(parse_text('put the plate on the table').model_dump_json(indent=2))"`
