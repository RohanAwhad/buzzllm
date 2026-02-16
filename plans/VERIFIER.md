# Verifier

1. Command chain: uv run pytest
2. Pass criteria: exit code 0; no failing tests.
3. Anti-thrash rule: if the same failure persists for 3 iterations, rollback to a prior checkpoint or adjust plan; record in devlogs.md.

## OpenAI Responses Verifier

1. Command chain: uv run pytest
2. Pass criteria: exit code 0; integration tests are skipped without OPENAI_API_KEY.
3. Anti-thrash rule: if the same failure persists for 3 iterations, rollback or adjust plan; record in devlogs.md.

## Structured Outputs Verifier Notes

1. Expected stdout (non-SSE) includes one JSON blob and DONE marker.
2. Expected SSE includes output_structured event and response_end; no output_text deltas.
