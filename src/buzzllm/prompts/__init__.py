from . import (
    codesearch_prompt,
    generate_prompt,
    helpful_prompt,
    replace_prompt,
    hackhub_prompt,
)

prompts = {
    "codesearch": codesearch_prompt.prompt,
    "generate": generate_prompt.prompt,
    "helpful": helpful_prompt.prompt,
    "replace": replace_prompt.prompt,
    "hackhub": hackhub_prompt.prompt,
}
