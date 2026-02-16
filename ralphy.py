#!/usr/bin/env python3
"""
Ralph Wiggum dual-session loop:
- Each iteration spawns TWO *fresh* Claude Code headless runs:
  1) Implementer: applies plans -> edits repo -> updates devlogs.md -> commits
  2) Verifier: blind-to-implementer narrative; reviews artifacts -> returns structured verdict

Inputs:
- plans/ directory: PLAN.md, IMPLEMENTER.md, VERIFIER.md, TESTPLAN.md, etc.
- devlogs.md at repo root (or specify path)
- (optional) a deterministic verifier shell command you want the driver to run
  (verifier runs it directly via Bash tool; recommended).

Outputs:
- plans/FEEDBACK.md appended each iteration (verdict + logs summary)
- logs/ralph.jsonl (session ids, verdict, timings)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

from loguru import logger


MAX_PLAN_CHARS = 120_000
MAX_DIFF_CHARS = 80_000
MAX_DEVLOG_CHARS = 30_000
MAX_FEEDBACK_CHARS = 25_000


# -------------------------
# Utilities
# -------------------------


def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    cmd: list[str],
    cwd: Path,
    timeout_s: Optional[int] = None,
    env: Optional[dict] = None,
) -> Tuple[int, str, str]:
    if timeout_s is not None and timeout_s <= 0:
        timeout_s = None
    cmd_name = cmd[0] if cmd else "process"
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _reader(stream, lines: list[str], stream_name: str) -> None:
        for line in iter(stream.readline, ""):
            lines.append(line)
            logger.info("{} {} | {}", cmd_name, stream_name, line.rstrip("\n"))
        stream.close()

    t_out = threading.Thread(
        target=_reader,
        args=(p.stdout, stdout_lines, "stdout"),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_reader,
        args=(p.stderr, stderr_lines, "stderr"),
        daemon=True,
    )
    t_out.start()
    t_err.start()

    timed_out = threading.Event()
    timer = None
    if timeout_s is not None:

        def _timeout() -> None:
            timed_out.set()
            p.kill()

        timer = threading.Timer(timeout_s, _timeout)
        timer.start()

    rc = p.wait()
    if timer:
        timer.cancel()

    t_out.join()
    t_err.join()

    if timed_out.is_set():
        assert timeout_s is not None
        raise subprocess.TimeoutExpired(cmd, float(timeout_s))

    return rc, "".join(stdout_lines), "".join(stderr_lines)


def configure_logger(log_path: Path) -> None:
    level = os.getenv("LOGGING_LEVEL", "INFO").upper()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    )
    logger.add(
        log_path,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    )


def clamp(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n\n...[truncated {len(s) - max_chars} chars]..."


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def write_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_md(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(block)


def sha_short(sha: str) -> str:
    return sha.strip()[:10] if sha else ""


# -------------------------
# Git / Repo artifact capture
# -------------------------


@dataclass
class RepoArtifacts:
    branch: str
    head: str
    status_porcelain: str
    diff_stat: str
    diff: str
    recent_log: str
    devlog_tail: str


def git(repo: Path, args: list[str], timeout_s: int) -> Tuple[int, str, str]:
    return run(["git"] + args, cwd=repo, timeout_s=timeout_s)


def capture_repo_artifacts(
    repo: Path,
    devlog_path: Path,
    timeout_s: int,
    max_diff_chars: int,
    max_devlog_chars: int,
) -> RepoArtifacts:
    _, branch, _ = git(repo, ["rev-parse", "--abbrev-ref", "HEAD"], timeout_s)
    _, head, _ = git(repo, ["rev-parse", "HEAD"], timeout_s)
    _, status, _ = git(repo, ["status", "--porcelain"], timeout_s)
    _, diff_stat, _ = git(repo, ["diff", "--stat"], timeout_s)
    _, diff, _ = git(repo, ["diff"], timeout_s)
    _, recent_log, _ = git(
        repo, ["log", "-n", "25", "--oneline", "--decorate"], timeout_s
    )

    devlog_txt = read_text(devlog_path) if devlog_path.exists() else ""
    devlog_tail = clamp(devlog_txt[-max_devlog_chars:], max_devlog_chars)

    return RepoArtifacts(
        branch=branch.strip(),
        head=head.strip(),
        status_porcelain=status.rstrip(),
        diff_stat=diff_stat.rstrip(),
        diff=clamp(diff.rstrip(), max_diff_chars),
        recent_log=recent_log.rstrip(),
        devlog_tail=devlog_tail.rstrip(),
    )


# -------------------------
# Plans bundle
# -------------------------


def read_plan_bundle(plans_dir: Path, include_feedback: bool, max_chars: int) -> str:
    md_files = sorted(plans_dir.glob("*.md"))
    if not include_feedback:
        md_files = [p for p in md_files if p.name != "FEEDBACK.md"]

    parts = []
    for p in md_files:
        parts.append(f"# FILE: {p.name}\n\n{read_text(p).strip()}\n")
    bundle = "\n\n---\n\n".join(parts).strip() + "\n"
    return clamp(bundle, max_chars)


def tail_feedback(plans_dir: Path, max_chars: int) -> str:
    p = plans_dir / "FEEDBACK.md"
    if not p.exists():
        return ""
    t = read_text(p)
    return clamp(t[-max_chars:], max_chars)


# -------------------------
# Claude Code headless invocation
# -------------------------


def claude_headless(
    claude_bin: str,
    prompt: str,
    repo: Path,
    timeout_s: int,
    model: Optional[str],
) -> Dict[str, Any]:
    """
    Uses Claude Code headless: `claude -p "...prompt..."`
    Returns parsed JSON when possible; otherwise raw text in "result".
    """
    # cmd = [claude_bin, "-p", prompt]
    cmd = ["opencode", "run", prompt]
    if model:
        cmd += ["--model", model]

    rc, out, err = run(cmd, cwd=repo, timeout_s=timeout_s)
    data = {"result": out, "_claude_rc": rc, "_stderr": err}
    if rc != 0 and not out.strip():
        data["_driver_error"] = f"claude rc={rc}"
        data["_stdout"] = out
        return data

    try:
        parsed = json.loads(out)
    except Exception:
        return data
    if isinstance(parsed, dict):
        parsed.setdefault("result", out)
        parsed["_claude_rc"] = rc
        parsed["_stderr"] = err
        return parsed
    return data


# -------------------------
# Verifier schema + parsing
# -------------------------

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "ROLLBACK_RECOMMENDED"]},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "risky_changes": {"type": "array", "items": {"type": "string"}},
        "next_actions": {"type": "array", "items": {"type": "string"}},
        "rollback_target": {"type": ["string", "null"]},
        "notes": {"type": "string"},
    },
    "required": [
        "verdict",
        "blockers",
        "risky_changes",
        "next_actions",
        "rollback_target",
        "notes",
    ],
}


def extract_structured_output(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    s = text.strip()
    if s:
        try:
            obj, _ = decoder.raw_decode(s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# -------------------------
# Prompts
# -------------------------


def build_implementer_prompt(
    plan_bundle: str,
    feedback_tail: str,
    artifacts: RepoArtifacts,
    iteration: int,
    devlog_path: Path,
) -> str:
    return textwrap.dedent(f"""\
    You are the IMPLEMENTER.

    Your job: implement the feature(s) exactly as specified in the PLAN BUNDLE.
    You must follow the latest VERIFIER VERDICT in FEEDBACK (if present).

    Hard rules:
    - Update {devlog_path.name} at least once this iteration and after any verifier/test run you perform.
    - Create meaningful checkpoint commits regularly with message format: `ralph(iter {iteration}): <milestone>`.
    - Do not weaken tests to "make them pass". Fix the code.
    - Keep changes focused to the plan. Avoid unrelated refactors.

    What you have:
    - PLAN BUNDLE (authoritative spec)
    - FEEDBACK tail (verifier verdicts)
    - Repo artifacts (git status/diff/log + devlog tail)

    Iteration: {iteration}

    === FEEDBACK (tail) ===
    {feedback_tail or "(empty)"}

    === REPO ARTIFACTS ===
    Branch: {artifacts.branch}
    HEAD: {sha_short(artifacts.head)}

    git status --porcelain:
    {artifacts.status_porcelain or "(clean)"}

    git diff --stat:
    {artifacts.diff_stat or "(no diff)"}

    Recent commits:
    {artifacts.recent_log or "(none)"}

    devlogs tail:
    {artifacts.devlog_tail or "(missing/empty)"}

    === PLAN BUNDLE ===
    {plan_bundle}

    Now implement. Use the repo tools as needed. When you think you're done, ensure verifier criteria in plans/VERIFIER.md are satisfied.
    YOU HAVE MY FULL APPROVAL FOR ALL THE TOOLS YOU WANNA USE AND ALL THE THINGS THAT YOU WANT TO DO.
    """).strip()


def build_verifier_prompt(
    plan_bundle: str,
    artifacts: RepoArtifacts,
    driver_verify_cmd: Optional[str],
    iteration: int,
    bias_appendix: Optional[str] = None,
) -> str:
    # Verifier is explicitly "blind" to implementer narrative.
    appendix_block = ""
    if bias_appendix:
        appendix_block = (
            f"\n\n=== APPENDIX (may be biased; optional) ===\n{bias_appendix}\n"
        )

    verifier_block = "(no verifier command provided)"
    if driver_verify_cmd:
        verifier_block = textwrap.dedent(f"""\
        Command to run (verifier must execute via Bash tool):
        {driver_verify_cmd}
        """).strip()

    return textwrap.dedent(f"""\
    You are the VERIFIER.

    Goal: Decide whether the current repo satisfies the PLAN BUNDLE.
    IMPORTANT: Do NOT rely on the implementer's explanation. Base your judgment on artifacts:
    - plans/*
    - git diff / changed files
    - recent commits
    - devlogs
    - test/lint logs (run the command below if provided)

    If a verifier command is provided below, you MUST run it via the Bash tool and
    use its results in your verdict. Include the exit code and any failing output in notes.

    Output MUST be a single JSON object that matches the schema below.
    Do NOT wrap the JSON in code fences or add extra text.
    - verdict: PASS | FAIL | ROLLBACK_RECOMMENDED
    - blockers: bullet strings tied to specific plan sections or failing evidence
    - risky_changes: bullet strings of what needs careful review
    - next_actions: ordered list (max 5), concrete
    - rollback_target: commit hash or null
    - notes: brief (include how you decided)

    === OUTPUT JSON SCHEMA ===
    {json.dumps(VERDICT_SCHEMA, indent=2)}

    Iteration: {iteration}

    === VERIFIER COMMAND (run via Bash tool) ===
    {verifier_block}

    === REPO ARTIFACTS ===
    Branch: {artifacts.branch}
    HEAD: {sha_short(artifacts.head)}

    git status --porcelain:
    {artifacts.status_porcelain or "(clean)"}

    git diff --stat:
    {artifacts.diff_stat or "(no diff)"}

    git diff (truncated if large):
    {artifacts.diff or "(no diff)"}

    Recent commits:
    {artifacts.recent_log or "(none)"}

    devlogs tail:
    {artifacts.devlog_tail or "(missing/empty)"}

    === PLAN BUNDLE (authoritative) ===
    {plan_bundle}
    {appendix_block}


    YOU HAVE MY FULL APPROVAL FOR ALL THE TOOLS YOU WANNA USE AND ALL THE THINGS THAT YOU WANT TO DO.
    """).strip()


# -------------------------
# Feedback + loop control
# -------------------------


def feedback_block(
    iteration: int,
    impl_session: Optional[str],
    ver_session: Optional[str],
    verdict: Dict[str, Any],
) -> str:
    return (
        textwrap.dedent(f"""\
    \n\n## Iteration {iteration} ({utc_ts()})
    - implementer_session: {impl_session or "unknown"}
    - verifier_session: {ver_session or "unknown"}

    ### Verifier verdict (structured)
    ```json
    {json.dumps(verdict, indent=2, ensure_ascii=False)}
    ```
    """).rstrip()
        + "\n"
    )


def maybe_auto_rollback(
    repo: Path, rollback_target: Optional[str], timeout_s: int
) -> Tuple[bool, str]:
    if not rollback_target:
        return False, "no rollback_target"
    # Basic safety: accept hex-ish commit ids only
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", rollback_target.strip()):
        return False, f"invalid rollback_target format: {rollback_target}"
    rc, out, err = git(repo, ["reset", "--hard", rollback_target.strip()], timeout_s)
    ok = rc == 0
    msg = (out + "\n" + err).strip()
    return ok, msg or ("rollback ok" if ok else "rollback failed")


# -------------------------
# Main
# -------------------------


@dataclass
class Args:
    repo: Path
    plans: Path
    claude_bin: str
    model_impl: Optional[str]
    model_ver: Optional[str]
    max_iters: int
    timeout_s: int
    devlog_path: Path
    driver_verify_cmd: Optional[str]
    auto_rollback: bool
    logs_jsonl: Path
    sleep_s: float
    include_feedback_in_plan_bundle: bool
    bias_appendix_mode: str  # "none" | "append_implementer_result"


def parse_args() -> Args:
    ap = argparse.ArgumentParser(
        description="Ralph dual-session loop: implementer Claude + verifier Claude (fresh sessions)."
    )

    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--plans", type=Path, default=Path("plans"))
    ap.add_argument("--claude-bin", type=str, default="claude")

    ap.add_argument("--model-impl", type=str, default=None)
    ap.add_argument("--model-ver", type=str, default=None)

    ap.add_argument("--max-iters", type=int, default=25)
    ap.add_argument(
        "--timeout-s",
        type=int,
        default=1800,
        help="Command timeout in seconds (0 disables timeout)",
    )

    ap.add_argument("--devlog", type=Path, default=Path("devlogs.md"))

    ap.add_argument(
        "--driver-verify-cmd",
        type=str,
        default=None,
        help='Optional deterministic verifier command run by the VERIFIER. Example: "ruff check . && mypy . && pytest -q"',
    )

    ap.add_argument(
        "--auto-rollback",
        action="store_true",
        help="If set, script will `git reset --hard <rollback_target>` when verifier recommends rollback.",
    )

    ap.add_argument("--logs-jsonl", type=Path, default=Path("logs/ralph.jsonl"))

    ap.add_argument("--sleep-s", type=float, default=1.0)

    ap.add_argument(
        "--include-feedback-in-plan-bundle",
        action="store_true",
        help="If set, plan bundle includes plans/FEEDBACK.md. Usually keep false to avoid bloat.",
    )
    ap.add_argument(
        "--bias-appendix-mode",
        type=str,
        default="none",
        choices=["none", "append_implementer_result"],
        help="Whether to include implementer's natural language result as optional appendix to verifier.",
    )

    ns = ap.parse_args()

    repo = ns.repo.resolve()
    plans = (repo / ns.plans).resolve()
    devlog = (repo / ns.devlog).resolve()
    logs_jsonl = (repo / ns.logs_jsonl).resolve()

    return Args(
        repo=repo,
        plans=plans,
        claude_bin=ns.claude_bin,
        model_impl=ns.model_impl,
        model_ver=ns.model_ver,
        max_iters=ns.max_iters,
        timeout_s=ns.timeout_s,
        devlog_path=devlog,
        driver_verify_cmd=ns.driver_verify_cmd,
        auto_rollback=ns.auto_rollback,
        logs_jsonl=logs_jsonl,
        sleep_s=ns.sleep_s,
        include_feedback_in_plan_bundle=ns.include_feedback_in_plan_bundle,
        bias_appendix_mode=ns.bias_appendix_mode,
    )


def main() -> int:
    args = parse_args()
    log_path = args.logs_jsonl.with_suffix(".log")
    configure_logger(log_path)
    logger.info(
        "ralphy start repo={} plans={} max_iters={} timeout_s={} verifier_cmd={}",
        args.repo,
        args.plans,
        args.max_iters,
        args.timeout_s,
        args.driver_verify_cmd or "(none)",
    )

    if not args.plans.exists():
        logger.error("plans dir not found: {}", args.plans)
        raise SystemExit(f"plans dir not found: {args.plans} (use --plans)")

    # Ensure FEEDBACK.md exists
    feedback_file = args.plans / "FEEDBACK.md"
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    if not feedback_file.exists():
        feedback_file.write_text("# FEEDBACK\n\n", encoding="utf-8")
        logger.info("created feedback file {}", feedback_file)

    # Ensure devlogs.md exists
    args.devlog_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.devlog_path.exists():
        args.devlog_path.write_text("# devlogs\n\n", encoding="utf-8")
        logger.info("created devlog file {}", args.devlog_path)

    last_head = ""
    for i in range(1, args.max_iters + 1):
        logger.info("iter {} start", i)
        t_iter0 = utc_ts()

        # Capture artifacts BEFORE implementer
        pre_art = capture_repo_artifacts(
            repo=args.repo,
            devlog_path=args.devlog_path,
            timeout_s=args.timeout_s,
            max_diff_chars=MAX_DIFF_CHARS,
            max_devlog_chars=MAX_DEVLOG_CHARS,
        )
        pre_status_lines = (
            len(pre_art.status_porcelain.splitlines())
            if pre_art.status_porcelain
            else 0
        )
        logger.debug(
            "pre artifacts branch={} head={} status_lines={} diff_stat={}",
            pre_art.branch,
            sha_short(pre_art.head),
            pre_status_lines,
            pre_art.diff_stat or "(no diff)",
        )

        plan_bundle_impl = read_plan_bundle(
            args.plans,
            include_feedback=args.include_feedback_in_plan_bundle,
            max_chars=MAX_PLAN_CHARS,
        )
        feedback_tail_txt = tail_feedback(args.plans, max_chars=MAX_FEEDBACK_CHARS)
        logger.debug(
            "plan bundle chars={} feedback chars={}",
            len(plan_bundle_impl),
            len(feedback_tail_txt),
        )

        impl_prompt = build_implementer_prompt(
            plan_bundle=plan_bundle_impl,
            feedback_tail=feedback_tail_txt,
            artifacts=pre_art,
            iteration=i,
            devlog_path=args.devlog_path,
        )

        logger.info("implementer run start model={}", args.model_impl or "(default)")
        impl_json = claude_headless(
            claude_bin=args.claude_bin,
            prompt=impl_prompt,
            repo=args.repo,
            timeout_s=args.timeout_s,
            model=args.model_impl,
        )
        impl_session = impl_json.get("session_id") or impl_json.get("session") or None
        impl_result_text = impl_json.get("result", "")
        logger.info(
            "implementer run done rc={} session={} out_chars={}",
            impl_json.get("_claude_rc"),
            impl_session or "(none)",
            len(impl_result_text),
        )
        if impl_json.get("_claude_rc") != 0:
            logger.error(
                "implementer nonzero rc={} stderr_chars={}",
                impl_json.get("_claude_rc"),
                len(impl_json.get("_stderr", "")),
            )
        if impl_json.get("_driver_error"):
            logger.error("implementer driver error: {}", impl_json.get("_driver_error"))

        # Verifier runs deterministic command (optional)
        drv_rc = None
        if args.driver_verify_cmd:
            logger.info(
                "verifier command configured (verifier must run): {}",
                args.driver_verify_cmd,
            )
        else:
            logger.debug("verifier command not configured")

        # Capture artifacts AFTER implementer (this is what verifier will judge)
        post_art = capture_repo_artifacts(
            repo=args.repo,
            devlog_path=args.devlog_path,
            timeout_s=args.timeout_s,
            max_diff_chars=MAX_DIFF_CHARS,
            max_devlog_chars=MAX_DEVLOG_CHARS,
        )
        post_status_lines = (
            len(post_art.status_porcelain.splitlines())
            if post_art.status_porcelain
            else 0
        )
        logger.debug(
            "post artifacts branch={} head={} status_lines={} diff_stat={}",
            post_art.branch,
            sha_short(post_art.head),
            post_status_lines,
            post_art.diff_stat or "(no diff)",
        )

        # Build verifier prompt (blind to implementer narrative by default)
        plan_bundle_ver = read_plan_bundle(
            args.plans, include_feedback=False, max_chars=MAX_PLAN_CHARS
        )
        appendix = (
            impl_result_text
            if args.bias_appendix_mode == "append_implementer_result"
            else None
        )

        ver_prompt = build_verifier_prompt(
            plan_bundle=plan_bundle_ver,
            artifacts=post_art,
            driver_verify_cmd=args.driver_verify_cmd,
            iteration=i,
            bias_appendix=appendix,
        )

        logger.info("verifier run start model={}", args.model_ver or "(default)")
        ver_json = claude_headless(
            claude_bin=args.claude_bin,
            prompt=ver_prompt,
            repo=args.repo,
            timeout_s=args.timeout_s,
            model=args.model_ver,
        )
        ver_session = ver_json.get("session_id") or ver_json.get("session") or None
        logger.info(
            "verifier run done rc={} out_chars={} err_chars={}",
            ver_json.get("_claude_rc"),
            len(ver_json.get("result", "")),
            len(ver_json.get("_stderr", "")),
        )
        if ver_json.get("_claude_rc") != 0:
            logger.error("verifier nonzero rc={}", ver_json.get("_claude_rc"))

        verdict = extract_structured_output(ver_json.get("result", ""))
        if verdict is None:
            raw_error = (
                ver_json.get("result", "") or ver_json.get("_stderr", "")
            ).strip()
            error_text = clamp(raw_error, 500) if raw_error else ""
            if ver_json.get("_claude_rc") != 0 and error_text:
                logger.error("verifier run failed; using synthesized verdict")
                verdict = {
                    "verdict": "FAIL",
                    "blockers": [f"Verifier run failed: {error_text}"],
                    "risky_changes": [],
                    "next_actions": [
                        "Fix verifier runner/model configuration, then rerun."
                    ],
                    "rollback_target": None,
                    "notes": "Driver synthesized verdict from verifier error output.",
                }
            else:
                logger.error("verifier output invalid JSON; using fallback verdict")
                verdict = {
                    "verdict": "FAIL",
                    "blockers": [
                        "Verifier did not return valid JSON; check logs/ralph.jsonl"
                    ],
                    "risky_changes": [],
                    "next_actions": ["Fix verifier output format/schema, then rerun."],
                    "rollback_target": None,
                    "notes": "Driver fallback verdict.",
                }

        # Append verdict to plans/FEEDBACK.md (this is the ONLY narrative implementer should read next)
        append_md(feedback_file, feedback_block(i, impl_session, ver_session, verdict))
        logger.debug("feedback appended")

        # Optional auto rollback
        rollback_did = False
        rollback_msg = ""
        if args.auto_rollback and verdict.get("verdict") == "ROLLBACK_RECOMMENDED":
            logger.warning("auto rollback recommended")
            ok, msg = maybe_auto_rollback(
                args.repo, verdict.get("rollback_target"), args.timeout_s
            )
            rollback_did = ok
            rollback_msg = msg
            append_md(
                feedback_file, f"\n- auto_rollback: {rollback_did} ({rollback_msg})\n"
            )
            if rollback_did:
                logger.info("auto rollback succeeded: {}", rollback_msg)
            else:
                logger.error("auto rollback failed: {}", rollback_msg)

        # Decide done
        done_by_verdict = verdict.get("verdict") == "PASS"
        done = done_by_verdict

        t_iter1 = utc_ts()

        # Log jsonl
        head_now = post_art.head
        rec = {
            "ts_start": t_iter0,
            "ts_end": t_iter1,
            "iter": i,
            "branch": post_art.branch,
            "head": head_now,
            "impl_session": impl_session,
            "ver_session": ver_session,
            "impl_rc": impl_json.get("_claude_rc"),
            "ver_rc": ver_json.get("_claude_rc"),
            "driver_verify_cmd": args.driver_verify_cmd,
            "driver_verify_rc": drv_rc,
            "verdict": verdict,
            "auto_rollback": rollback_did,
            "auto_rollback_msg": rollback_msg,
        }
        write_jsonl(args.logs_jsonl, rec)
        logger.debug("wrote jsonl log record")
        logger.info(
            "iter {} verdict={} head={} driver_rc={} done={}",
            i,
            verdict.get("verdict"),
            sha_short(head_now),
            drv_rc,
            done,
        )

        if done:
            logger.info("done PASS stopping")
            return 0

        # If no repo progress for many iters, the verifier should recommend rollback,
        # but we can at least detect "stuck head".
        if last_head and head_now == last_head:
            logger.warning(
                "HEAD unchanged since last iteration ({}), might be stuck",
                sha_short(head_now),
            )
        last_head = head_now

        time.sleep(args.sleep_s)

    logger.warning("max iterations hit")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("unhandled exception")
        raise
