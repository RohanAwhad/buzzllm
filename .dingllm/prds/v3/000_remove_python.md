# v3: Remove Python Implementation, Promote Rust to Root

## Goal

Remove the Python implementation entirely and move the Rust implementation from `rust/` to the repository root. After this change, `Cargo.toml` lives at root, `src/` contains Rust source, and no Python packaging/test infrastructure remains.

## Motivation

The Rust port (v2) is complete and CI-tested. The Python implementation is no longer maintained — it has no CI, and the README already points users to the Rust build. Keeping both adds confusion for contributors and agents navigating the repo.

## Plan

### 1. Remove Python files

Delete these tracked paths:

| Path | What it is |
|------|-----------|
| `src/buzzllm/` | Python source (main.py, llm.py, prompts/, tools/) |
| `tests/` | Python test suite (conftest.py, unit/, integration/, e2e/) |
| `pyproject.toml` | Python packaging config |
| `.python-version` | Pins Python 3.10 for uv |
| `uv.lock` | uv lockfile |
| `python_runtime_docker/` | Docker build for Python exec tool |

Note: `.venv/`, `build/`, `__pycache__/` are gitignored and not tracked.

### 2. Move Rust to root

Move these from `rust/` to repo root:

| From | To |
|------|-----|
| `rust/Cargo.toml` | `Cargo.toml` |
| `rust/Cargo.lock` | `Cargo.lock` |
| `rust/src/` | `src/` |
| `rust/tests/` | `tests/` |
| `rust/.cargo/` | `.cargo/` |

Delete `rust/.gitignore` (its content — `target/` — merges into root `.gitignore`).

Use `git mv` for each to preserve history.

### 3. Update CI

`.github/workflows/ci.yml` — remove `working-directory: rust` from all 4 jobs (build, test, clippy, fmt) and the `workspaces: rust` from rust-cache config.

Before:
```yaml
- uses: Swatinem/rust-cache@v2
  with:
    workspaces: rust
- name: Build (debug)
  run: cargo build
  working-directory: rust
```

After:
```yaml
- uses: Swatinem/rust-cache@v2
- name: Build (debug)
  run: cargo build
```

### 4. Update `.gitignore`

Replace Python-centric patterns with Rust:

```gitignore
# Rust
target/

# Misc
.DS_Store
dev_scratch
.dingllm/files.txt
llm.md
```

### 5. Update `AGENTS.md`

- Remove "Two implementations" framing
- Remove Python quick-reference section
- Remove Python-specific gotchas (global state, conftest, asyncio_mode)
- Remove Python layout entries
- Keep Rust commands, gotchas, and layout (updated paths: `src/` not `rust/src/`)
- Keep "Adding a new provider" / "Adding a new tool" recipes (updated for root paths)

### 6. Update `CLAUDE.md`

Remove or replace. Current content is mostly Python-focused (CLI examples, Python architecture, Python data flow). Options:

- **(a) Delete** — the README and AGENTS.md already cover everything an agent needs.
- **(b) Rewrite** — trim to Rust-only architecture reference.

Recommendation: **(a) Delete**. The README has full CLI usage, AGENTS.md has dev workflow. A third file adds maintenance burden with no new information.

### 7. Update `README.md`

Minimal changes:
- Install section: `cd buzzllm/rust` becomes `cd buzzllm` 
- Architecture section: paths drop the `rust/` prefix (already shows `src/` without prefix)
- Python test suite reference section: remove entirely

## Files changed (summary)

| Action | Path |
|--------|------|
| DELETE | `src/buzzllm/` (Python source) |
| DELETE | `tests/` (Python tests) |
| DELETE | `pyproject.toml` |
| DELETE | `.python-version` |
| DELETE | `uv.lock` |
| DELETE | `python_runtime_docker/` |
| DELETE | `CLAUDE.md` |
| DELETE | `rust/` (after moving contents) |
| MOVE | `rust/Cargo.toml` -> `Cargo.toml` |
| MOVE | `rust/Cargo.lock` -> `Cargo.lock` |
| MOVE | `rust/src/` -> `src/` |
| MOVE | `rust/tests/` -> `tests/` |
| MOVE | `rust/.cargo/` -> `.cargo/` |
| EDIT | `.github/workflows/ci.yml` |
| EDIT | `.gitignore` |
| EDIT | `AGENTS.md` |
| EDIT | `README.md` |

## Acceptance criteria

1. `cargo build` works from repo root
2. `cargo test -- --test-threads=1` passes from repo root
3. `cargo clippy -- -D warnings` passes
4. `cargo fmt -- --check` passes
5. CI workflow runs successfully (no `working-directory` references to `rust/`)
6. No Python files remain in tracked tree (`*.py`, `pyproject.toml`, etc.)
7. `git log --follow src/main.rs` shows history through the move

## Execution order

```
git stash                          # save current uncommitted work
git checkout -b remove-python-implementation

# Step 1: remove Python
git rm -r src/buzzllm tests pyproject.toml .python-version uv.lock python_runtime_docker

# Step 2: move Rust to root (git mv preserves history)
git mv rust/Cargo.toml Cargo.toml
git mv rust/Cargo.lock Cargo.lock
git mv rust/src src
git mv rust/tests tests
git mv rust/.cargo .cargo
git rm rust/.gitignore
rmdir rust  # should be empty now

# Step 3-7: edit CI, .gitignore, AGENTS.md, README.md, delete CLAUDE.md
# ... (see sections above)

git add -A
git commit -m "refactor: remove Python implementation, promote Rust to repo root

src/buzzllm/, tests/, pyproject.toml, .python-version, uv.lock, python_runtime_docker/
- Remove entire Python implementation (source, tests, packaging, Docker build)
- Move rust/ contents to repo root (Cargo.toml, src/, tests/, .cargo/)
- Update CI to remove working-directory: rust
- Update .gitignore for Rust-only repo
- Update AGENTS.md and README.md for new layout
- Delete CLAUDE.md (redundant with README + AGENTS.md)"

git stash pop                      # restore uncommitted work
```

## Not in scope

- Adding new Rust features
- Changing the Rust code itself (only file moves)
- Modifying Cargo.toml dependencies or configuration
- Touching `.dingllm/specs/` diagrams (they already reference the Rust layout)
