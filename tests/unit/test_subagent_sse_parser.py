from buzzllm.subagent import parse_sse_output_text


class TestSubagentSseParser:
    def test_parses_only_output_text(self):
        sse_lines = [
            "event: output_text",
            'data: {"id":"","delta":"Hello ","type":"output_text"}',
            "",
            "event: tool_call",
            'data: {"id":"","delta":"ignored","type":"tool_call"}',
            "",
            "event: output_text",
            'data: {"id":"","delta":"world","type":"output_text"}',
            "",
            "event: reasoning_content",
            'data: {"id":"","delta":"ignored","type":"reasoning_content"}',
        ]

        output = parse_sse_output_text(sse_lines)

        assert output == "Hello world"
