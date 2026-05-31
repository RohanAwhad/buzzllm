# Phase 5: Codesearch Tools

## Goal

Port the three codesearch tools (`bash_find`, `bash_ripgrep`, `bash_read`) to Rust. These run `rg` and `find` as subprocesses with pagination, path validation, and timeout.

## Source reference

- `src/buzzllm/tools/codesearch.py:6` — `CWD = Path.cwd().resolve()`
- `src/buzzllm/tools/codesearch.py:9-14` — `_validate_path()`
- `src/buzzllm/tools/codesearch.py:17-36` — `_paginate_results()`
- `src/buzzllm/tools/codesearch.py:125-179` — `bash_find()`
- `src/buzzllm/tools/codesearch.py:182-226` — `bash_ripgrep()`
- `src/buzzllm/tools/codesearch.py:229-256` — `bash_read()`

## Tools to implement

### Shared utilities

#### Path validation
Resolve the path and confirm it's within `CWD`. Return an error if the path escapes CWD (no `..` traversal out of workspace).

```rust
fn validate_path(path: &str, cwd: &Path) -> Result<PathBuf> {
    let resolved = cwd.join(path).canonicalize()?;
    if !resolved.starts_with(cwd) {
        return Err(anyhow!("path outside CWD not allowed: {}", resolved.display()));
    }
    Ok(resolved)
}
```

#### Pagination
Apply offset/limit to a `Vec<String>` and return a JSON object:

```json
{
  "results": ["line1", "line2"],
  "total": 150,
  "offset": 0,
  "limit": 20,
  "has_more": true,
  "returned": 20
}
```

`limit=0` means return all results from offset onward.

### Tool 1: `BashFind`

**Name**: `bash_find`

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | `"."` | Directory to search in |
| `name` | string | `""` | Glob pattern for filename filtering |
| `type_filter` | string | `""` | `"d"` for directories only |
| `extra_args` | string | `""` | Additional CLI args (space-split) |
| `limit` | integer | `20` | Max results (0 = all) |
| `offset` | integer | `0` | Skip N results |

**Behavior**:
- Default: run `rg --files [path]` — fast, respects `.gitignore`
- If `name` is set: `rg --files --glob "{name}"`
- If `type_filter == "d"`: fall back to `find {path} -type d [-name {name}]`
- `extra_args`: split on space, append to command
- Timeout: 30 seconds
- Paginate results

**Schema description**: Use the content from `bash_find_tool_desc` in the Python source (the multi-line string explaining rg --files usage). This becomes the `description` field in the tool schema.

### Tool 2: `BashRipgrep`

**Name**: `bash_ripgrep`

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `pattern` | string | (required) | Regex pattern to search for |
| `files` | string | `"."` | Path to search in |
| `extra_args` | string | `""` | Additional CLI args |
| `limit` | integer | `20` | Max results (0 = all) |
| `offset` | integer | `0` | Skip N results |

**Behavior**:
- Run `rg {pattern} [{files}]`
- `extra_args`: split on space, append
- Timeout: 30 seconds
- On returncode != 0: return `{"error": "No matches found or command failed: {stderr}"}`
- Paginate results

**Schema description**: Use `bash_ripgrep_tool_desc` content.

### Tool 3: `BashRead`

**Name**: `bash_read`

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `filepath` | string | (required) | Path to file to read |
| `limit` | integer | `0` | Max lines (0 = all) |
| `offset` | integer | `0` | Skip N lines |

**Behavior**:
- Read file contents as UTF-8
- Split into lines, paginate
- Return pagination object + `content` field (joined lines)

```json
{
  "results": ["line1", "line2"],
  "content": "line1\nline2",
  "total": 50,
  ...
}
```

### Implementation

All three implement the `Tool` trait from Phase 4:
- `execute()` receives JSON args, deserializes into a params struct, runs the subprocess/reads the file, returns JSON result
- Use `tokio::process::Command` for async subprocess execution
- CWD is captured once at startup (or passed in)

## Verification

1. `bash_find` with no args lists files in CWD (matches `rg --files` output)
2. `bash_find` with `name: "*.rs"` filters correctly
3. `bash_find` with `type_filter: "d"` falls back to `find`
4. `bash_ripgrep` finds a known pattern in the repo
5. `bash_ripgrep` with no matches returns error object
6. `bash_read` reads a known file, returns correct line count and content
7. Path validation rejects `../../../etc/passwd`
8. Pagination: 100 results with limit=20, offset=40 returns items 40-59 with `has_more: true`
9. Timeout: a pathological command (if constructible) returns timeout error
10. End-to-end: `cargo run -- "gpt-4o-mini" ... --system-prompt codesearch` with a codesearch question
