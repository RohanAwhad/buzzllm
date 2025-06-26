from pathlib import Path
from typing import Dict

# Get current file path and directory
current_file: Path = Path(__file__)
current_dir: Path = current_file.parent

# Find all *_prompt.txt files and read their content
prompts: Dict[str, str] = {}

for prompt_file in current_dir.glob("*_prompt.txt"):
    content: str = prompt_file.read_text(encoding='utf-8').strip()
    if content:
        filename: str = prompt_file.stem.replace("_prompt", "")
        prompts[filename] = content

# Make it available for importers
__all__ = ["prompts"]
