# BuzzLLM Rust v2 — Diagram Manifest

| Diagram | Level | Type | Source Files | Last Verified |
|---------|-------|------|-------------|---------------|
| 001_context_flowchart_buzzllm.mmd | L1 Context | flowchart | all of rust/src/ | 2026-05-31 |
| 002_container_flowchart_buzzllm.mmd | L2 Container | flowchart | rust/src/main.rs, rust/src/tools/pythonexec.rs | 2026-05-31 |
| 003_component_flowchart_buzzllm.mmd | L3 Component | flowchart | all rust/src/*.rs and rust/src/**/*.rs | 2026-05-31 |
| 004_component_sequence_chat_flow.mmd | L3 Component | sequence | rust/src/main.rs, rust/src/llm.rs, rust/src/providers/mod.rs, rust/src/output.rs | 2026-05-31 |
| 005_component_sequence_tool_execution.mmd | L3 Component | sequence | rust/src/llm.rs, rust/src/tools/*.rs | 2026-05-31 |

## Cross-references

- 004 traverses: 003 → MAIN, PROMPTS, TMOD, LLM, PMOD, OUTPUT
- 005 traverses: 003 → LLM, TMOD, CS, WS, PE
