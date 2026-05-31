use crate::output::print_to_stdout;
use crate::providers::LlmClient;
use crate::tools::ToolRegistry;
use crate::types::{LlmOptions, StreamResponse, StreamResponseType, ToolCallData};
use futures::StreamExt;
use std::collections::HashMap;

pub async fn invoke_llm(
    opts: &LlmOptions,
    prompt: &str,
    system_prompt: &str,
    provider: &dyn LlmClient,
    registry: &ToolRegistry,
    sse: bool,
    brief: bool,
) {
    let request_args = match provider.build_request(opts, prompt, system_prompt) {
        Ok(args) => args,
        Err(e) => {
            tracing::error!("Failed to build request args: {}", e);
            let err_resp =
                StreamResponse::new("", format!("Error: {}", e), StreamResponseType::BlockEnd);
            print_to_stdout(&err_resp, sse, brief);
            print_to_stdout(
                &StreamResponse::new("", "", StreamResponseType::ResponseEnd),
                sse,
                brief,
            );
            return;
        }
    };

    let mut messages: Vec<serde_json::Value> = request_args
        .data
        .get("messages")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let mut data = request_args.data.clone();
    let mut tool_calls: HashMap<String, ToolCallData> = HashMap::new();
    let mut current_tool_call_id = String::new();

    let url = if !opts.url.is_empty() {
        opts.url.clone()
    } else {
        provider.default_api_url(&opts.model)
    };

    let client = reqwest::Client::new();

    let result: Result<(), anyhow::Error> = async {
        for _iter in 0..opts.max_infer_iters {
            let mut message_started = false;

            let mut req = client
                .post(&url)
                .timeout(std::time::Duration::from_secs(900));

            for (key, value) in &request_args.headers {
                req = req.header(key, value);
            }

            let response = req.json(&data).send().await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                tracing::error!("HTTP {} response: {}", status, body);
                return Err(anyhow::anyhow!("HTTP {}: {}", status, body));
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].to_string();
                    buffer = buffer[newline_pos + 1..].to_string();

                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let stream_responses = provider.parse_sse_line(
                        trimmed,
                        message_started,
                        &mut tool_calls,
                        &mut current_tool_call_id,
                    );

                    for sr in stream_responses {
                        if sr.response_type == StreamResponseType::ResponseStart {
                            message_started = true;
                        }
                        print_to_stdout(&sr, sse, brief);
                    }
                }
            }

            let trimmed = buffer.trim();
            if !trimmed.is_empty() {
                let stream_responses = provider.parse_sse_line(
                    trimmed,
                    message_started,
                    &mut tool_calls,
                    &mut current_tool_call_id,
                );
                for sr in stream_responses {
                    print_to_stdout(&sr, sse, brief);
                }
            }

            if tool_calls.is_empty() {
                return Ok(());
            }

            run_tools(&mut tool_calls, registry).await;

            for tc in tool_calls.values() {
                if tc.executed && tc.result.is_some() {
                    let result_response = StreamResponse::new(
                        &tc.id,
                        format!(
                            "\n\nTool Result ({}):\n{}\n",
                            tc.name,
                            tc.result.as_ref().unwrap()
                        ),
                        StreamResponseType::ToolResult,
                    );
                    print_to_stdout(&result_response, sse, brief);
                }
            }

            provider.assemble_tool_messages(&mut messages, &tool_calls);
            data["messages"] = serde_json::json!(messages);

            tool_calls.clear();
            current_tool_call_id.clear();
        }
        Ok(())
    }
    .await;

    if let Err(e) = result {
        tracing::error!("invoke_llm error: {}", e);
        let err_resp =
            StreamResponse::new("", format!("Error: {}", e), StreamResponseType::BlockEnd);
        print_to_stdout(&err_resp, sse, brief);
    }

    registry.cleanup().await;

    print_to_stdout(
        &StreamResponse::new("", "", StreamResponseType::ResponseEnd),
        sse,
        brief,
    );
}

async fn run_tools(tool_calls: &mut HashMap<String, ToolCallData>, registry: &ToolRegistry) {
    let mut futures = Vec::new();
    let ids: Vec<String> = tool_calls.keys().cloned().collect();

    for id in &ids {
        if let Some(tc) = tool_calls.get(id) {
            if tc.executed {
                continue;
            }
            let name = tc.name.clone();
            let arguments = tc.arguments.clone();
            let id = id.clone();

            if let Some(tool) = registry.get(&name) {
                let args: serde_json::Value = serde_json::from_str(&arguments)
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                futures.push(async move {
                    let result = tool.execute(args).await;
                    (id, result)
                });
            }
        }
    }

    let results = futures::future::join_all(futures).await;

    for (id, result) in results {
        if let Some(tc) = tool_calls.get_mut(&id) {
            tc.result = Some(result);
            tc.executed = true;
        }
    }
}
