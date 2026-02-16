# Definition of Done

1. call_subagent tool available when tools enabled.
2. Subagent tool_subset allowlist enforced.
3. SSE parser returns only output_text.
4. uv run pytest passes.
5. devlogs.md updated for final iteration.
6. No new dependencies added.

## OpenAI Responses Definition of Done

1. make_openai_responses_request_args builds input list and tools when present.
2. handle_openai_responses_stream_response emits tool_call + output_text events.
3. Tool results are sent with previous_response_id and tool_result input items.
4. uv run pytest passes (integration skipped without OPENAI_API_KEY).
5. devlogs.md updated for final iteration.

## Structured Outputs Definition of Done

1. OpenAI chat uses response_format when output_mode is set.
2. Anthropic/Vertex uses output_config.format when output_mode is set.
3. Non-SSE prints JSON once; SSE emits output_structured.
4. uv run pytest passes.
