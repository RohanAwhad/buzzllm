pub fn get_prompt(name: &str) -> Option<&'static str> {
    match name {
        "codesearch" => Some(include_str!("codesearch.txt")),
        "generate" => Some(include_str!("generate.txt")),
        "hackhub" => Some(include_str!("hackhub.txt")),
        "helpful" => Some(include_str!("helpful.txt")),
        "replace" => Some(include_str!("replace.txt")),
        "websearch" => Some(include_str!("websearch.txt")),
        _ => None,
    }
}

pub fn prompt_names() -> &'static [&'static str] {
    &["codesearch", "generate", "hackhub", "helpful", "replace", "websearch"]
}
