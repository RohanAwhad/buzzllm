# Test Plan

1. Unit tests
   1. tests/unit/test_subagent_sse_parser.py
      1. Parses SSE with mixed events; concatenates only output_text.
      2. Handles multiple output_text chunks.
   2. tests/unit/test_tool_catalog.py
      1. Accepts valid tool_subset names.
      2. Rejects unknown tool names with error string.
   3. tests/unit/test_subagent_payload.py
      1. Payload builder preserves model/provider/url/api_key_name/think/temperature/max_tokens.
2. Integration tests
   1. tests/integration/test_subagent_runner.py
      1. Mock subprocess stdout to SSE fixture; raw_call_subagent returns expected output.

Fixtures/mocks
1. SSE fixture string with output_text, tool_call, reasoning_content events.
2. Mock subprocess.Popen with controllable stdout/returncode.

Acceptance
1. New tests pass without API keys.
2. No new integration tests require external services.

## OpenAI Responses Test Plan

1. Unit tests
   1. tests/unit/test_llm_request_args.py
      1. input list uses role=user + input_text content.
      2. tools + tool_choice=auto are included when opts.tools is set.
      3. reasoning effort toggles with opts.think.
      4. max_output_tokens + temperature set from opts.
   2. tests/unit/test_llm_stream_handlers.py
      1. response.output_item.added(tool_call) creates ToolCall with id/name.
      2. response.tool_call_arguments.delta appends arguments.
      3. response.output_item.done(tool_call) fills arguments if missing.
   3. tests/unit/test_llm_tool_messages.py
      1. tool_call_response_to_openai_responses_messages sets previous_response_id.
      2. tool_result input items are constructed correctly.
2. Integration tests (skipped without OPENAI_API_KEY)
   1. tests/integration/test_openai_responses_integration.py
      1. simple responses output_text flow.
      2. tool call + tool result continuation (single tool) using a tiny stub tool.
   2. Manual real-API smoke test (requires OPENAI_API_KEY)
      1. Run openai-responses and openai-chat against live OpenAI APIs using OPENAI_API_KEY.

Fixtures/mocks
1. JSON lines for response.created, output_item.added(tool_call), tool_call_arguments.delta, output_text.delta.
2. Stub tool function for tool_result mapping.

Acceptance
1. Unit tests cover tool-call parsing and continuation request shape.
2. Integration test is skipped without OPENAI_API_KEY.
3. Real-API smoke test uses OPENAI_API_KEY when available.

## Structured Outputs Test Plan

1. Unit tests
   1. tests/unit/test_llm_request_args.py
      1. openai-chat response_format json_schema/json_object set when output_mode is enabled.
      2. anthropic/vertex output_config.format set when output_mode is enabled.
   2. tests/unit/test_llm_stream_handlers.py
      1. output_text is buffered and not printed when structured output is enabled.
      2. output_structured emitted once on block_end.
   3. tests/unit/test_llm_structured_output.py
      1. Non-SSE stdout contains JSON once and DONE marker.
      2. SSE includes output_structured event only.

Acceptance
1. Structured output tests pass without API keys.
