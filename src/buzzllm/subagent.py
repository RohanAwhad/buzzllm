import json
from typing import Iterable


def parse_sse_output_text(lines: Iterable[str]) -> str:
    output_chunks = []
    current_event = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event: "):
            current_event = line[len("event: ") :].strip()
            continue
        if line.startswith("data: ") and current_event == "output_text":
            data_content = line[len("data: ") :]
            payload = json.loads(data_content)
            delta = payload.get("delta", "")
            if delta:
                output_chunks.append(delta)

    return "".join(output_chunks)
