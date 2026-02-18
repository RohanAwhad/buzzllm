# Data

1. Storage changes: none.
2. Migrations: none.
3. In-memory data:
   1. SSE output_text buffer in src/buzzllm/subagent.py.
   2. tool_subset list (validated against catalog).

Acceptance
1. No new files under data/ or migrations/.
2. No persistent state introduced.

## OpenAI Responses Data

1. Storage changes: none.
2. Migrations: none.
3. In-memory data:
   1. last_openai_response_id (string) for previous_response_id.
   2. request_args.data["input"] list reused between tool turns.

Acceptance
1. No new persistence is introduced for responses.
2. previous_response_id is only stored in-memory per invocation.

## Structured Outputs Data

1. Storage changes: none.
2. Migrations: none.
3. In-memory data:
   1. structured_output_buffer (string) for output_text accumulation.

Acceptance
1. No new persistence is introduced for structured outputs.
