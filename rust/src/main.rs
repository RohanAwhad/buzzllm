mod types;
mod providers;
mod tools;
mod prompts;
mod output;
mod llm;

use clap::Parser;
use providers::Provider;
use tools::ToolRegistry;
use types::LlmOptions;

#[derive(Parser, Debug)]
#[command(name = "buzzllm", about = "Invoke LLM with streaming response")]
struct Cli {
    /// LLM model name
    model: String,

    /// LLM API URL
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

    let file_appender = tracing_appender::rolling::never("/tmp", "buzzllm.logs");

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(file_appender)
                .with_ansi(false)
        )
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("buzzllm=debug"))
        )
        .init();
}

async fn chat(args: Cli) {
    let provider = match Provider::from_str(&args.provider) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    let original_system_prompt = args.system_prompt.clone();

    // Resolve system prompt: check if it's a known prompt name, otherwise use as literal text
    let system_prompt = prompts::get_prompt(&args.system_prompt)
        .unwrap_or(&args.system_prompt);

    // Register tools based on system prompt name
    let mut registry = ToolRegistry::new();
    let tools: Option<Vec<serde_json::Value>> = match original_system_prompt.as_str() {
        "websearch" => {
            registry.register(Box::new(tools::websearch::SearchWeb));
            registry.register(Box::new(tools::websearch::ScrapeWebpage));
            Some(if provider.is_anthropic_format() {
                registry.anthropic_schemas()
            } else {
                registry.openai_schemas()
            })
        }
        "codesearch" => {
            registry.register(Box::new(tools::codesearch::BashFind));
            registry.register(Box::new(tools::codesearch::BashRipgrep));
            registry.register(Box::new(tools::codesearch::BashRead));
            Some(if provider.is_anthropic_format() {
                registry.anthropic_schemas()
            } else {
                registry.openai_schemas()
            })
        }
        "pythonexec" => {
            registry.register(Box::new(tools::pythonexec::PythonExecute::new()));
            Some(if provider.is_anthropic_format() {
                registry.anthropic_schemas()
            } else {
                registry.openai_schemas()
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

    llm::invoke_llm(&opts, &args.prompt, system_prompt, &provider, &registry, args.sse, args.brief).await;
}

#[tokio::main]
async fn main() {
    init_logging();
    let args = Cli::parse();
    chat(args).await;
}
