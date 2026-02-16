# API

1. Tool: call_subagent
   1. Signature: call_subagent(prompt: str, system_prompt: str | None = None, tool_subset: list[str] | None = None) -> str
   2. Behavior: spawns subprocess; parses SSE; returns concatenated output_text.
   3. Error behavior: returns "Error: <message>" string on validation or subprocess failure.
2. Subprocess input contract (stdin JSON)
   1. Fields: model, provider, url, api_key_name, prompt, system_prompt, tool_subset, think, temperature, max_tokens.
   2. Example payload:
      {
        "model": "<parent model>",
        "provider": "openai-chat|anthropic|vertexai-anthropic",
        "url": "<parent url>",
        "api_key_name": "<env var>",
        "prompt": "...",
        "system_prompt": "...",
        "tool_subset": ["search_web", "scrape_webpage"],
        "think": false,
        "temperature": 0.8,
        "max_tokens": 8192
      }
3. SSE parsing contract
   1. Only consume events with "event: output_text".
   2. Ignore tool_call, tool_result, reasoning_content, block_end, response_end.

Available tool names
1. search_web
2. scrape_webpage
3. bash_find
4. bash_ripgrep
5. bash_read
6. python_execute

## OpenAI Responses API Contract

1. Request args (make_openai_responses_request_args)
   1. data.model = opts.model
   2. data.input = [
      {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
      ]
   3. data.instructions = system_prompt when provided
   4. data.stream = True, data.store = False
   5. data.temperature = opts.temperature
   6. data.max_output_tokens = opts.max_tokens when provided
   7. data.reasoning = {"effort": "high", "summary": "detailed"} if opts.think else {"effort": "none"}
   8. if opts.tools: data.tools = opts.tools and data.tool_choice = "auto"

2. Streaming event mapping (handle_openai_responses_stream_response)
   1. response.created -> StreamResponse(type=response_start, id=response.id)
   2. response.output_text.delta -> StreamResponse(type=output_text, delta)
   3. response.reasoning_summary_text.delta -> StreamResponse(type=reasoning_content, delta)
   4. response.output_item.added (item.type=tool_call) -> create ToolCall(id=item.id, name=item.name)
   5. response.tool_call_arguments.delta -> append ToolCall.arguments; emit StreamResponse(type=tool_call, delta)
   6. response.output_item.done (item.type=tool_call) -> set ToolCall.arguments from item.arguments if present
   7. response.completed -> StreamResponse(type=block_end)

3. Tool response continuation (tool_call_response_to_openai_responses_messages)
   1. request_args.data["previous_response_id"] = last_openai_response_id
   2. request_args.data["input"] = [
      {"type": "tool_result", "tool_call_id": tc.id, "content": str(tc.result)}
      ] for each tool call
   3. tool_result items replace prior input for the follow-up request (no message history replay).

4. invoke_llm routing contract
   1. If request_args.data contains "messages", use that list and write back to "messages".
   2. Else if request_args.data contains "input", use that list and write back to "input".

## Structured Outputs API Contract

1. LLMOptions additions (internal)
   1. output_mode: "json_schema" | "json_object" | None
   2. output_schema: dict | None
2. OpenAI chat request args
   1. response_format json_schema:
      {"type":"json_schema","json_schema":{"name":"<name>","schema":<schema>,"strict":true}}
   2. response_format json_object:
      {"type":"json_object"}
3. Anthropic/Vertex request args
   1. output_config: {"format":{"type":"json_schema","schema":<schema>}}
4. Output to stdout
   1. Non-SSE prints JSON once at end, then DONE marker.
   2. SSE emits one output_structured event with JSON string.

Non-SSE example
```
{"capital":"Paris","country":"France"}

=== [ DONE ] ===
```

SSE example
```
event: output_structured
data: {"id":"","delta":"{\"capital\":\"Paris\",\"country\":\"France\"}","type":"output_structured"}

event: response_end
data: {"id":"","delta":"","type":"response_end"}
```
