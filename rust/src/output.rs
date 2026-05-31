use crate::types::{StreamResponse, StreamResponseType};
use std::io::{self, Write};

pub fn print_to_stdout(data: &StreamResponse, sse: bool, brief: bool) {
    let stdout = io::stdout();
    print_to_writer(data, sse, brief, &mut stdout.lock());
}

pub fn print_to_writer(data: &StreamResponse, sse: bool, brief: bool, out: &mut impl Write) {
    if brief
        && matches!(
            data.response_type,
            StreamResponseType::ToolCall | StreamResponseType::ToolResult
        )
    {
        return;
    }

    if sse {
        let _ = writeln!(out, "event: {}", data.response_type);
        let _ = writeln!(out, "data: {}", data.to_json());
        let _ = writeln!(out);
        let _ = out.flush();
        return;
    }

    match data.response_type {
        StreamResponseType::ToolCall => {
            let _ = write!(out, "\x1b[96m{}\x1b[0m", data.delta);
        }
        StreamResponseType::ToolResult => {
            let _ = write!(out, "\x1b[92m{}\x1b[0m", data.delta);
        }
        StreamResponseType::ReasoningContent => {
            let _ = write!(out, "\x1b[93m{}\x1b[0m", data.delta);
        }
        StreamResponseType::BlockEnd => {
            let _ = writeln!(out);
        }
        StreamResponseType::ResponseEnd => {
            let _ = writeln!(out, "\n\n=== [ DONE ] ===");
        }
        _ => {
            let _ = write!(out, "{}", data.delta);
        }
    }
    let _ = out.flush();
}
