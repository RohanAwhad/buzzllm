# Risks

1. SSE parsing misses content due to event ordering.
   1. Mitigation: parse by event type; unit tests cover mixed events.
2. Tool name mismatch between catalog and schema.
   1. Mitigation: single catalog; validate tool_subset; unit tests.
3. Subprocess overhead and flaky IO.
   1. Mitigation: minimal stdout parsing; integration test with mock stdout.
4. Recursive subagents create infinite loops.
   1. Mitigation: optional max depth guard in call_subagent.

Acceptance
1. Tests cover SSE parsing and allowlist enforcement.

## OpenAI Responses Risks

1. Responses streaming event schema mismatch with parser.
   1. Mitigation: add unit fixtures for output_item.added/tool_call_arguments.delta/output_item.done; adjust parser to exact fields.
2. previous_response_id not propagated, causing tool loop to stall.
   1. Mitigation: store last response id on response.created and assert in unit tests.
3. Tool result input shape rejected by API.
   1. Mitigation: keep tool_result items minimal; integration test validates end-to-end.

Acceptance
1. Tool-call parsing covered by unit tests and integration test (when API key is set).

## Structured Outputs Risks

1. Partial JSON streamed to stdout and parsed by callers.
   1. Mitigation: buffer output_text; print JSON once or output_structured only.
2. Schema mismatch between OpenAI and Anthropic formats.
   1. Mitigation: keep provider-specific mapping in request args; unit tests for both.
