use buzzllm::providers::{self, ToolSchemaFormat};
use buzzllm::tools::ToolRegistry;
use buzzllm::types::LlmOptions;
use buzzllm::{llm, prompts};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "buzzllm", about = "Invoke LLM with streaming response")]
struct Cli {
    /// LLM model name
    model: String,

    /// LLM API URL (pass empty string for provider default)
    url: String,

    /// User prompt
    prompt: String,

    /// System prompt name or custom text
    #[arg(long, default_value = "scream at mee for not setting your system prompt",
           long_help = format!("System prompt. Use a predefined prompt name or provide your own custom system prompt text. Available prompts: {}", crate::prompts::prompt_names().join(", ")))]
    system_prompt: String,

    /// LLM provider type
    #[arg(long, value_parser = ["openai-chat", "openai-responses", "anthropic", "vertexai-anthropic"])]
    provider: String,

    /// Environment variable name for API key
    #[arg(long)]
    api_key_name: String,

    /// Maximum tokens in response
    #[arg(long, default_value_t = 8192)]
    max_tokens: u32,

    /// Response temperature
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Enable thinking mode
    #[arg(long)]
    think: bool,

    /// Enable SSE mode for printing
    #[arg(short = 'S', long)]
    sse: bool,

    /// Only print final output, hide tool calls and results
    #[arg(short, long)]
    brief: bool,
}

fn init_logging() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let file_appender = tracing_appender::rolling::daily("/tmp", "buzzllm.logs");

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(file_appender).with_ansi(false))
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("buzzllm=debug")))
        .init();
}

async fn chat(args: Cli) {
    let client = match providers::create_client(&args.provider) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    let original_system_prompt = args.system_prompt.clone();

    let system_prompt = prompts::get_prompt(&args.system_prompt).unwrap_or(&args.system_prompt);

    let mut registry = ToolRegistry::new();
    let tools: Option<Vec<serde_json::Value>> = match original_system_prompt.as_str() {
        "websearch" => {
            registry.register(Box::new(buzzllm::tools::websearch::SearchWeb));
            registry.register(Box::new(buzzllm::tools::websearch::ScrapeWebpage));
            Some(match client.tool_schema_format() {
                ToolSchemaFormat::Anthropic => registry.anthropic_schemas(),
                ToolSchemaFormat::OpenAI => registry.openai_schemas(),
            })
        }
        "codesearch" => {
            registry.register(Box::new(buzzllm::tools::codesearch::BashFind));
            registry.register(Box::new(buzzllm::tools::codesearch::BashRipgrep));
            registry.register(Box::new(buzzllm::tools::codesearch::BashRead));
            Some(match client.tool_schema_format() {
                ToolSchemaFormat::Anthropic => registry.anthropic_schemas(),
                ToolSchemaFormat::OpenAI => registry.openai_schemas(),
            })
        }
        "coding" => {
            registry.register(Box::new(buzzllm::tools::codesearch::BashRead));
            registry.register(Box::new(buzzllm::tools::write_file::WriteFile));
            registry.register(Box::new(buzzllm::tools::bash::Bash));
            registry.register(Box::new(buzzllm::tools::websearch::SearchWeb));
            registry.register(Box::new(buzzllm::tools::websearch::ScrapeWebpage));
            Some(match client.tool_schema_format() {
                ToolSchemaFormat::Anthropic => registry.anthropic_schemas(),
                ToolSchemaFormat::OpenAI => registry.openai_schemas(),
            })
        }
        "pythonexec" => {
            registry.register(Box::new(buzzllm::tools::pythonexec::PythonExecute::new()));
            Some(match client.tool_schema_format() {
                ToolSchemaFormat::Anthropic => registry.anthropic_schemas(),
                ToolSchemaFormat::OpenAI => registry.openai_schemas(),
            })
        }
        _ => None,
    };

    let opts = LlmOptions {
        model: args.model,
        url: args.url,
        api_key_name: Some(args.api_key_name),
        max_tokens: Some(args.max_tokens),
        temperature: args.temperature,
        think: args.think,
        tools,
        max_infer_iters: 10,
    };

    llm::invoke_llm(
        &opts,
        &args.prompt,
        system_prompt,
        client.as_ref(),
        &registry,
        args.sse,
        args.brief,
    )
    .await;
}

#[tokio::main]
async fn main() {
    init_logging();
    let args = Cli::parse();
    chat(args).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_parse_minimal_args() {
        let args = Cli::try_parse_from([
            "buzzllm",
            "gpt-4",
            "",
            "hello",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
        ])
        .unwrap();
        assert_eq!(args.model, "gpt-4");
        assert_eq!(args.url, "");
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.provider, "openai-chat");
        assert_eq!(args.api_key_name, "KEY");
    }

    #[test]
    fn test_defaults() {
        let args = Cli::try_parse_from([
            "buzzllm",
            "gpt-4",
            "",
            "hello",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
        ])
        .unwrap();
        assert_eq!(args.max_tokens, 8192);
        assert_eq!(args.temperature, 0.8);
        assert!(!args.think);
        assert!(!args.sse);
        assert!(!args.brief);
        assert_eq!(args.url, "");
    }

    #[test]
    fn test_custom_max_tokens() {
        let args = Cli::try_parse_from([
            "buzzllm",
            "gpt-4",
            "",
            "hello",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "--max-tokens",
            "4096",
        ])
        .unwrap();
        assert_eq!(args.max_tokens, 4096);
    }

    #[test]
    fn test_custom_temperature() {
        let args = Cli::try_parse_from([
            "buzzllm",
            "gpt-4",
            "",
            "hello",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "--temperature",
            "0.5",
        ])
        .unwrap();
        assert_eq!(args.temperature, 0.5);
    }

    #[test]
    fn test_think_flag() {
        let args = Cli::try_parse_from([
            "buzzllm",
            "gpt-4",
            "",
            "hello",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "--think",
        ])
        .unwrap();
        assert!(args.think);
    }

    #[test]
    fn test_sse_short_flag() {
        let args = Cli::try_parse_from([
            "buzzllm",
            "gpt-4",
            "",
            "hello",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "-S",
        ])
        .unwrap();
        assert!(args.sse);
    }

    #[test]
    fn test_brief_flag() {
        let args = Cli::try_parse_from([
            "buzzllm",
            "gpt-4",
            "",
            "hello",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
            "-b",
        ])
        .unwrap();
        assert!(args.brief);
    }

    #[test]
    fn test_url_positional() {
        let args = Cli::try_parse_from([
            "buzzllm",
            "gpt-4",
            "https://custom.example.com/v1",
            "hello",
            "--provider",
            "openai-chat",
            "--api-key-name",
            "KEY",
        ])
        .unwrap();
        assert_eq!(args.url, "https://custom.example.com/v1");
    }

    #[test]
    fn test_all_providers_accepted() {
        for p in &[
            "openai-chat",
            "openai-responses",
            "anthropic",
            "vertexai-anthropic",
        ] {
            let args = Cli::try_parse_from([
                "buzzllm",
                "gpt-4",
                "",
                "hello",
                "--provider",
                p,
                "--api-key-name",
                "KEY",
            ])
            .unwrap();
            assert_eq!(args.provider, *p);
        }
    }

    #[test]
    fn test_invalid_provider_rejected() {
        let result = Cli::try_parse_from([
            "buzzllm",
            "gpt-4",
            "",
            "hello",
            "--provider",
            "invalid-provider",
            "--api-key-name",
            "KEY",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_provider_rejected() {
        let result = Cli::try_parse_from(["buzzllm", "gpt-4", "", "hello", "--api-key-name", "KEY"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_model_rejected() {
        let result = Cli::try_parse_from(["buzzllm", "--provider", "openai-chat"]);
        assert!(result.is_err());
    }
}
