import subprocess
from pathlib import Path
from subprocess import TimeoutExpired

# Get current working directory as absolute path
CWD = Path.cwd().resolve()


def _validate_path(path_str: str) -> str:
    """Ensure path is within CWD and return relative path"""
    path = Path(path_str).resolve()
    if not str(path).startswith(str(CWD)):
        print("path outside cwd")
        raise ValueError(f"Path outside CWD not allowed: {path}")
    return str(path.relative_to(CWD))


def _paginate_results(results: list[str], limit: int = 20, offset: int = 0) -> dict:
    """Apply pagination to results and return metadata"""
    total = len(results)

    if limit is None or limit == 0:
        paginated = results[offset:]
        has_more = False
    else:
        end = offset + limit
        paginated = results[offset:end]
        has_more = end < total

    return {
        "results": paginated,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "returned": len(paginated),
    }


bash_find_tool_desc = """
This function uses `rg --files` to list files efficiently:

**Core command**: `rg --files [path]`
- Lists all files recursively
- Respects `.gitignore` automatically
- Much faster than `find` for file listing

**Key usage patterns in your function**:

```bash
# Basic file listing
rg --files

# With glob filter
rg --files --glob "*.py"

# In specific directory  
rg --files /path/to/dir
```

**Why ripgrep here**:
- Fast recursive file discovery
- Built-in `.gitignore` respect
- Clean output (one file per line)
- Good for large codebases

**Limitations in your implementation**:
- Falls back to `find` for directories (`type_filter="d"`)
- No built-in pagination (you handle it post-process)

**Alternative rg options you could add**:
- `--type py` (file type filters)
- `--hidden` (include hidden files)
- `--no-ignore` (ignore .gitignore)
""".strip()

bash_ripgrep_tool_desc = """
This function uses `rg` for text searching within files:

**Core command**: `rg "pattern" [path]`
- Searches for regex patterns in file contents
- Fast, recursive, respects `.gitignore`
- Shows filename:line:content by default

**Key usage patterns in your function**:

```bash
# Basic search
rg "function"

# Search in specific path
rg "TODO" src/

# With extra args
rg "error" --ignore-case --context 2
```

**Common `extra_args` options**:
- `--ignore-case` / `-i`: Case insensitive
- `--word-regexp` / `-w`: Match whole words
- `--context 3` / `-C3`: Show 3 lines around matches
- `--type py`: Search only Python files
- `--line-number` / `-n`: Show line numbers (default)
- `--no-heading`: Don't group by file
- `--color never`: Disable colors for parsing

**Output format**:
```
filename:line_number:matching_line_content
```

**Why ripgrep here**:
- Extremely fast text search
- Regex support built-in
- Clean, parseable output
- Handles large codebases well

**Your function strengths**:
- Post-processing pagination
- Error handling for no matches
- Path validation
- Timeout protection
""".strip()


def bash_find(
    path: str = ".",
    name: str = "",
    type_filter: str = "",
    extra_args: str = "",
    limit: int = 20,
    offset: int = 0,
):
    """
    Execute rg --files command (respects .gitignore) and return paginated output.

    Args:
      path (str): Directory path to search in, defaults to current directory
      name (str): Glob pattern to filter filenames
      type_filter (str): Filter by type ('d' for directories, None for files)
      extra_args (str): Additional command line arguments to pass
      limit (int): Maximum number of results to return. 0 to return all of them
      offset (int): Number of results to skip from the beginning

    Returns:
      dict: Contains 'results' list, 'total' count, 'returned' count, 'offset' value,
            or 'error' key with error message if command fails
    """
    validated_path = _validate_path(path)

    cmd = ["rg", "--files"]

    if name:
        cmd.extend(["--glob", name])

    if type_filter == "d":
        cmd = ["find", validated_path, "-type", "d"]
        if name:
            cmd.extend(["-name", name])
    else:
        if path != ".":
            cmd.append(validated_path)

    if extra_args:
        cmd.extend(extra_args.split(" "))

    try:
        print("running the following cmd:", cmd)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=CWD
        )
        if result.returncode != 0:
            print("error: no matches found or command failed:", result.stderr)
            return {"error": f"Command failed: {result.stderr}"}

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        paginated = _paginate_results(lines, limit, offset)

    except TimeoutExpired:
        return {"error": "Command timed out after 30 seconds"}

    print(f"Total files: {paginated['total']}, Returned: {paginated['returned']}")
    return paginated


def bash_ripgrep(
    pattern: str,
    files: str = ".",
    extra_args: str = "",
    limit: int = 20,
    offset: int = 0,
):
    """
    Execute ripgrep command and return paginated output.

    Args:
      pattern (str): Regular expression pattern to search for
      files (str): File or directory path to search in, defaults to current directory
      extra_args (List[str]): Additional command line arguments to pass to ripgrep
      limit (int): Maximum number of results to return. 0 to return all of them.
      offset (int): Number of results to skip from the beginning

    Returns:
      dict: Contains 'results' list, 'total' count, 'returned' count, 'offset' value,
            or 'error' key with error message if command fails or no matches found
    """
    validated_files = _validate_path(files)

    cmd = ["rg", pattern]

    if files != ".":
        cmd.append(validated_files)

    if extra_args:
        cmd.extend(extra_args.split(" "))

    try:
        print("running the following cmd:", cmd)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=CWD
        )
        if result.returncode != 0:
            print("error: no matches found or command failed:", result.stderr)
            return {"error": f"No matches found or command failed: {result.stderr}"}

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        paginated = _paginate_results(lines, limit, offset)

    except TimeoutExpired:
        return {"error": "Command timed out after 30 seconds"}

    print(f"Total matches: {paginated['total']}, Returned: {paginated['returned']}")
    return paginated


def bash_read(filepath: str, limit: int = 0, offset: int = 0):
    """
    Read file contents with pagination, restricted to current working directory.

    Args:
      filepath: Path to the file to read
      limit: Maximum number of lines to return
      offset: Number of lines to skip from the beginning

    Returns:
      dict: Contains 'results' list of lines, 'content' string with joined lines,
            'total' line count, 'returned' count, 'offset' value,
            or 'error' key with error message if file cannot be read
    """
    validated_path = _validate_path(filepath)

    try:
        with open(validated_path, "r") as f:
            lines = f.read().strip().split("\n")

        paginated = _paginate_results(lines, limit, offset)

        paginated["content"] = "\n".join(paginated["results"])

        print(
            f"File: {filepath}, Total lines: {paginated['total']}, Returned: {paginated['returned']}"
        )
        return paginated

    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}
