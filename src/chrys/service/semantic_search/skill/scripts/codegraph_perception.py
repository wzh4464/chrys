#!/usr/bin/env python3
"""Collect optional CodeGraph-backed repository perception for semantic-search."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from _common import (
    FORMAT_CODEGRAPH,
    FORMAT_INDEX,
    ScriptError,
    append_trace,
    bullet_lines,
    ensure_allowed_path,
    load_json,
    markdown_escape_line,
    now_iso,
    path_tokens,
    read_text,
    reject_benchmark_answer_path,
    resolve_path,
    sha1_path,
    stable_unique,
    tokenize,
    write_json,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Seeded workspace repo root.")
    parser.add_argument("--requirement", required=True, help="Original task prompt.")
    parser.add_argument("--index", required=True, help="Light semantic-search index from build_index.py.")
    parser.add_argument("--out", required=True, help="Output codegraph-perception.json path.")
    parser.add_argument("--markdown", help="Output codegraph-perception.md path. Defaults to out with .md suffix.")
    parser.add_argument("--artifact-dir", help="Semantic-search artifact directory. Defaults to output parent.")
    parser.add_argument(
        "--codegraph-cmd",
        default=os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_CMD", ""),
        help="CodeGraph CLI command. Can include arguments, e.g. 'uvx codegraph'. Defaults to auto-resolving codegraph in the current Python environment.",
    )
    parser.add_argument(
        "--install-codegraph",
        choices=("auto", "never", "force"),
        default=os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_INSTALL", "never"),
        help=(
            "Install CodeGraph when no CLI is available. Any value but 'never' downloads and runs a "
            "third-party installer, so it also requires a pinned --codegraph-install-sha256."
        ),
    )
    parser.add_argument(
        "--codegraph-install-url",
        default=os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_INSTALL_URL", "https://github.com/colbymchenry/codegraph"),
        help="Base GitHub repository URL recorded in diagnostics.",
    )
    parser.add_argument(
        "--codegraph-install-script-url",
        default=os.environ.get(
            "SEMANTIC_SEARCH_CODEGRAPH_INSTALL_SCRIPT_URL",
            "https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh",
        ),
        help="Official CodeGraph installer script URL.",
    )
    parser.add_argument(
        "--codegraph-install-sha256",
        default=os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_INSTALL_SHA256", ""),
        help="SHA-256 of the installer script. Required to install: an unpinned script is refused.",
    )
    parser.add_argument(
        "--codegraph-version",
        default=os.environ.get("CODEGRAPH_VERSION", ""),
        help="Optional CodeGraph release tag/version. Defaults to latest release.",
    )
    parser.add_argument(
        "--timeout", type=float, default=float(os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_TIMEOUT", "25"))
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=int(os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_MAX_OUTPUT_CHARS", "9000")),
    )
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--max-symbols", type=int, default=4)
    return parser.parse_args(argv)


def load_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path, str, dict[str, Any]]:
    repo = resolve_path(args.repo)
    if not repo.is_dir():
        raise ScriptError(f"repo path does not exist: {repo}")
    out = resolve_path(args.out)
    artifact_dir = resolve_path(args.artifact_dir or out.parent)
    requirement = ensure_allowed_path(
        args.requirement,
        allowed_roots=[artifact_dir],
        allowed_files=[resolve_path(args.requirement)],
        purpose="requirement",
    )
    reject_benchmark_answer_path(requirement, purpose="requirement")
    index_path = ensure_allowed_path(args.index, allowed_roots=[repo, artifact_dir], purpose="index")
    out = ensure_allowed_path(out, allowed_roots=[repo, artifact_dir, out.parent], purpose="output")
    markdown = resolve_path(args.markdown or out.with_suffix(".md"))
    markdown = ensure_allowed_path(
        markdown, allowed_roots=[repo, artifact_dir, markdown.parent], purpose="markdown-output"
    )
    requirement_text = read_text(requirement)
    index = load_json(index_path)
    if index.get("format") != FORMAT_INDEX:
        raise ScriptError(f"unsupported index format: {index.get('format')}")
    return repo, requirement, index_path, out, markdown, artifact_dir, requirement_text, index


def command_tokens(raw: str) -> list[str]:
    try:
        tokens = shlex.split(raw)
    except ValueError as err:
        raise ScriptError(f"invalid CodeGraph command {raw!r}: {err}") from err
    if not tokens:
        raise ScriptError("empty CodeGraph command")
    return tokens


def command_available(tokens: list[str]) -> bool:
    executable = tokens[0]
    if "/" in executable:
        return Path(executable).expanduser().exists()
    return shutil.which(executable) is not None


def current_python_tool_dirs() -> tuple[Path, Path]:
    executable = Path(os.path.abspath(sys.executable))
    if executable.parent.name == "bin":
        bin_dir = executable.parent
        install_dir = executable.parent.parent / "codegraph"
    else:
        bin_dir = executable.parent
        install_dir = executable.parent / "codegraph"
    return install_dir, bin_dir


def chrys_root() -> Path:
    return Path(__file__).resolve().parents[4]


def codegraph_download_dir() -> Path:
    override = os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_DOWNLOAD_DIR", "").strip()
    if override:
        return resolve_path(override)
    return chrys_root() / ".semantic-search-tools" / "codegraph" / "downloads"


def resolve_codegraph_command(args: argparse.Namespace, artifact_dir: Path) -> tuple[list[str], dict[str, Any]]:
    requested = args.codegraph_cmd.strip()
    install_dir, bin_dir = current_python_tool_dirs()
    download_dir = codegraph_download_dir()
    env_candidate = bin_dir / "codegraph"
    resolution: dict[str, Any] = {
        "requested_command": requested,
        "install_mode": args.install_codegraph,
        "python": sys.executable,
        "chrys_root": str(chrys_root()),
        "download_dir": str(download_dir),
        "install_dir": str(install_dir),
        "bin_dir": str(bin_dir),
        "attempts": [],
    }
    if requested:
        tokens = command_tokens(requested)
        if command_available(tokens):
            resolution.update({"source": "requested", "command": tokens, "available": True})
            return tokens, resolution
        if args.install_codegraph == "never":
            resolution.update(
                {
                    "source": "requested",
                    "command": tokens,
                    "available": False,
                    "note": "requested CodeGraph command is unavailable and installation is disabled",
                }
            )
            return tokens, resolution
    else:
        if env_candidate.is_file() and os.access(env_candidate, os.X_OK):
            tokens = [str(env_candidate)]
            resolution.update({"source": "python-env", "command": tokens, "available": True})
            return tokens, resolution
        if command_available(["codegraph"]):
            resolution.update({"source": "path", "command": ["codegraph"], "available": True})
            return ["codegraph"], resolution
        tokens = [str(env_candidate)]

    if args.install_codegraph in {"auto", "force"}:
        install_result = install_codegraph_bundle(
            install_dir=install_dir,
            bin_dir=bin_dir,
            download_dir=download_dir,
            repo_url=args.codegraph_install_url,
            install_script_url=args.codegraph_install_script_url,
            install_script_sha256=args.codegraph_install_sha256,
            version=args.codegraph_version,
            timeout=args.timeout,
        )
        resolution["attempts"].append(install_result)
        if env_candidate.is_file() and os.access(env_candidate, os.X_OK):
            tokens = [str(env_candidate)]
            resolution.update({"source": "installed-python-env", "command": tokens, "available": True})
            return tokens, resolution
        if command_available(["codegraph"]):
            resolution.update({"source": "installed-path", "command": ["codegraph"], "available": True})
            return ["codegraph"], resolution

    resolution.update({"source": "unavailable", "command": tokens, "available": False})
    return tokens, resolution


def install_codegraph_bundle(
    *,
    install_dir: Path,
    bin_dir: Path,
    download_dir: Path,
    repo_url: str,
    install_script_url: str,
    install_script_sha256: str,
    version: str,
    timeout: float,
) -> dict[str, Any]:
    """Download, verify, and run the official CodeGraph installer.

    The script is fetched to a file and checked against a caller-supplied
    SHA-256 before anything executes it. Piping a remote URL straight into a
    shell would make whoever controls that branch -- or whoever can set
    ``SEMANTIC_SEARCH_CODEGRAPH_INSTALL_SCRIPT_URL`` -- an arbitrary-code-
    execution vector, which is exactly what this repository's pinning rule
    exists to prevent, so an unpinned install is refused outright.
    """
    result: dict[str, Any] = {
        "method": "official-install-script",
        "repo_url": repo_url,
        "install_script_url": install_script_url,
        "requested_version": version,
        "download_dir": str(download_dir),
        "install_dir": str(install_dir),
        "bin_dir": str(bin_dir),
        "ok": False,
    }
    try:
        expected_digest = install_script_sha256.strip().lower()
        if not expected_digest:
            raise ScriptError(
                "refusing to run the CodeGraph installer without --codegraph-install-sha256: "
                "pin the script's digest, or install CodeGraph yourself and pass --codegraph-cmd"
            )
        curl = shutil.which("curl") or "/usr/bin/curl"
        shell = shutil.which("sh") or "/usr/bin/sh"
        if not Path(curl).exists() and "/" in curl:
            raise ScriptError("curl is required for the official CodeGraph installer")
        if not Path(shell).exists() and "/" in shell:
            raise ScriptError("sh is required for the official CodeGraph installer")
        download_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = download_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)
        install_dir.mkdir(parents=True, exist_ok=True)
        script_path = download_dir / "codegraph-install.sh"
        fetch = subprocess.run(  # noqa: S603 — argv is a list built from validated inputs
            [curl, "-fsSL", "-o", str(script_path), install_script_url],
            text=True,
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
        if fetch.returncode != 0:
            raise ScriptError(
                f"could not download the CodeGraph installer (rc={fetch.returncode}): "
                f"{(fetch.stderr or '').strip()[:500]}"
            )
        actual_digest = hashlib.sha256(script_path.read_bytes()).hexdigest()
        result["install_script_sha256"] = actual_digest
        if actual_digest != expected_digest:
            raise ScriptError(f"CodeGraph installer digest mismatch: expected {expected_digest}, got {actual_digest}")
        env = os.environ.copy()
        env.update(
            {
                "CODEGRAPH_INSTALL_DIR": str(install_dir),
                "CODEGRAPH_BIN_DIR": str(bin_dir),
                "TMPDIR": str(tmp_dir),
            }
        )
        if version:
            env["CODEGRAPH_VERSION"] = normalize_version(version)
        cmd = [shell, str(script_path)]
        proc = subprocess.run(  # noqa: S603 — argv is a list built from validated inputs
            cmd,
            text=True,
            capture_output=True,
            check=False,
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
        result.update(
            {
                "argv": cmd,
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[:4000],
                "stderr": (proc.stderr or "")[:4000],
                "tmp_dir": str(tmp_dir),
            }
        )
        launcher = bin_dir / "codegraph"
        if proc.returncode != 0:
            raise ScriptError(
                f"official CodeGraph installer failed with rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
            )
        if not launcher.is_file() and not launcher.is_symlink():
            raise ScriptError(f"official CodeGraph installer did not create {launcher}")
        launcher.chmod(launcher.stat().st_mode | 0o111)
        result.update({"ok": True, "launcher": str(launcher)})
        return result
    except (OSError, subprocess.SubprocessError, ScriptError) as err:
        result.update({"ok": False, "error": str(err)})
        return result


def normalize_version(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ScriptError("empty CodeGraph version")
    return cleaned if cleaned.startswith("v") else f"v{cleaned}"


def run_command(base: list[str], extra: list[str], *, cwd: Path, timeout: float, max_chars: int) -> dict[str, Any]:
    argv = [*base, *extra]
    try:
        proc = subprocess.run(  # noqa: S603 — argv is a list built from validated inputs
            argv,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        stdout = (proc.stdout or "")[:max_chars]
        stderr = (proc.stderr or "")[:max_chars]
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
        }
    except FileNotFoundError as err:
        return {"argv": argv, "returncode": 127, "ok": False, "stdout": "", "stderr": str(err)}
    except subprocess.TimeoutExpired as err:
        return {
            "argv": argv,
            "returncode": 124,
            "ok": False,
            "stdout": (err.stdout or "")[:max_chars] if isinstance(err.stdout, str) else "",
            "stderr": f"CodeGraph command timed out after {timeout}s",
        }


def extract_requirement_terms(text: str) -> list[str]:
    weighted: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            weighted.extend(tokenize(stripped))
    weighted.extend(tokenize(text))
    return stable_unique(weighted)[:80]


def query_strings(requirement_text: str, index: dict[str, Any], limit: int) -> list[str]:
    terms = extract_requirement_terms(requirement_text)
    queries: list[str] = []
    if terms:
        queries.append(" ".join(terms[:6]))
    for size in (4, 3, 2):
        for start in range(0, min(len(terms), 24), size):
            value = " ".join(terms[start : start + size])
            if len(value) >= 6:
                queries.append(value)
            if len(queries) >= limit:
                return stable_unique(queries)[:limit]
    for record in index.get("files", [])[:80]:
        path = record.get("path", "")
        if path and any(term in path.lower() for term in terms[:30]):
            queries.append(path)
        if len(queries) >= limit:
            break
    return stable_unique(queries)[:limit]


def symbol_queries(requirement_text: str, index: dict[str, Any], limit: int) -> list[str]:
    terms = set(extract_requirement_terms(requirement_text))
    candidates: list[tuple[int, str]] = []
    for symbol in index.get("symbols", []):
        name = str(symbol.get("name", ""))
        if not name:
            continue
        pieces = set(path_tokens(name))
        score = len(pieces & terms)
        path = str(symbol.get("file", ""))
        if any(term in path.lower() for term in terms):
            score += 1
        if score <= 0:
            continue
        candidates.append((score, name))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return stable_unique([name for _, name in candidates])[:limit]


def collect_codegraph(args: argparse.Namespace) -> dict[str, Any]:
    repo, requirement, index_path, out, markdown, artifact_dir, requirement_text, index = load_inputs(args)
    base, resolution = resolve_codegraph_command(args, artifact_dir)
    available = bool(resolution.get("available"))
    payload: dict[str, Any] = {
        "format": FORMAT_CODEGRAPH,
        "created_at": now_iso(),
        "backend": "codegraph-cli",
        "available": available,
        "inputs": {
            "repo": str(repo),
            "requirement": str(requirement),
            "requirement_sha1": sha1_path(requirement),
            "index": str(index_path),
            "index_sha1": sha1_path(index_path),
            "command": base,
            "command_resolution": resolution,
        },
        "command_resolution": resolution,
        "setup_attempts": [],
        "repository_queries": [],
        "symbol_relationships": [],
        "summary": {},
        "notes": [],
    }
    if not available:
        payload["notes"].append(
            "CodeGraph CLI was not found or could not be installed; repository perception will use builtin static evidence only."
        )
        payload["summary"] = {"status": "unavailable", "query_count": 0, "symbol_relationship_count": 0}
        write_outputs(out, markdown, payload)
        append_trace("codegraph-perception", {"out": str(out), "available": False})
        return payload

    setup_results = []
    status_before = run_command(base, ["status"], cwd=repo, timeout=args.timeout, max_chars=args.max_output_chars)
    setup_results.append(status_before)
    if not status_before["ok"]:
        init_result = run_command(base, ["init"], cwd=repo, timeout=args.timeout, max_chars=args.max_output_chars)
        setup_results.append(init_result)
    status_after = run_command(base, ["status"], cwd=repo, timeout=args.timeout, max_chars=args.max_output_chars)
    setup_results.append(status_after)
    files_result = run_command(base, ["files"], cwd=repo, timeout=args.timeout, max_chars=args.max_output_chars)
    setup_results.append(files_result)
    if not files_result["ok"]:
        setup_results.append(
            run_command(base, ["explore"], cwd=repo, timeout=args.timeout, max_chars=args.max_output_chars)
        )
    payload["setup_attempts"] = setup_results

    query_results = []
    for query in query_strings(requirement_text, index, args.max_queries):
        attempts, best = run_first_success(
            base,
            [["query", query, "--limit", "8"], ["query", query]],
            cwd=repo,
            timeout=args.timeout,
            max_chars=args.max_output_chars,
        )
        query_results.append({"query": query, "attempts": attempts, "best": best})
    payload["repository_queries"] = query_results

    relationships = []
    for symbol in symbol_queries(requirement_text, index, args.max_symbols):
        relationships.append(
            {
                "symbol": symbol,
                "node": run_first_success(
                    base,
                    [["node", symbol], ["query", symbol]],
                    cwd=repo,
                    timeout=args.timeout,
                    max_chars=args.max_output_chars,
                )[1],
                "callers": run_first_success(
                    base,
                    [["callers", symbol], ["query", f"callers {symbol}"]],
                    cwd=repo,
                    timeout=args.timeout,
                    max_chars=args.max_output_chars,
                )[1],
                "callees": run_first_success(
                    base,
                    [["callees", symbol], ["query", f"callees {symbol}"]],
                    cwd=repo,
                    timeout=args.timeout,
                    max_chars=args.max_output_chars,
                )[1],
                "impact": run_first_success(
                    base,
                    [["impact", symbol], ["query", f"impact {symbol}"]],
                    cwd=repo,
                    timeout=args.timeout,
                    max_chars=args.max_output_chars,
                )[1],
            }
        )
    payload["symbol_relationships"] = relationships
    payload["summary"] = {
        "status": "available",
        "setup_success_count": sum(1 for item in setup_results if item.get("ok")),
        "setup_attempt_count": len(setup_results),
        "query_count": len(query_results),
        "query_success_count": sum(1 for item in query_results if item.get("best", {}).get("ok")),
        "symbol_relationship_count": len(relationships),
        "relationship_success_count": sum(relationship_success_count(item) for item in relationships),
    }
    write_outputs(out, markdown, payload)
    append_trace(
        "codegraph-perception",
        {
            "out": str(out),
            "available": True,
            "queries": payload["summary"]["query_count"],
            "query_successes": payload["summary"]["query_success_count"],
        },
    )
    return payload


def first_success(results: list[dict[str, Any]]) -> dict[str, Any]:
    for result in results:
        if result.get("ok"):
            return result
    return results[-1] if results else {"ok": False, "stdout": "", "stderr": "no attempts"}


def run_first_success(
    base: list[str],
    alternatives: list[list[str]],
    *,
    cwd: Path,
    timeout: float,
    max_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for extra in alternatives:
        result = run_command(base, extra, cwd=cwd, timeout=timeout, max_chars=max_chars)
        results.append(result)
        if result.get("ok"):
            return results, result
    return results, first_success(results)


def relationship_success_count(item: dict[str, Any]) -> int:
    return sum(1 for key in ("node", "callers", "callees", "impact") if item.get(key, {}).get("ok"))


def write_outputs(out: Path, markdown: Path, payload: dict[str, Any]) -> None:
    write_json(out, payload)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CodeGraph Perception",
        "",
        "This document captures optional CodeGraph CLI evidence for repository understanding. It is evidence for requirement augmentation, not an edit mandate.",
        "",
        "## Status",
        "",
        f"- Backend: {payload.get('backend', '')}",
        f"- Available: {payload.get('available')}",
    ]
    summary = payload.get("summary", {})
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    notes = payload.get("notes", [])
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(bullet_lines(notes))
    lines.extend(["", "## Setup Evidence", ""])
    for item in payload.get("setup_attempts", [])[:8]:
        lines.append(f"- Command: `{shell_join(item.get('argv', []))}`")
        lines.append(f"  Return code: {item.get('returncode')}")
        if item.get("stdout"):
            lines.extend(["", "  Stdout:", "", "  ```text"])
            lines.extend(indent_text(item.get("stdout", ""), "  "))
            lines.append("  ```")
        if item.get("stderr"):
            lines.extend(["", "  Stderr:", "", "  ```text"])
            lines.extend(indent_text(item.get("stderr", ""), "  "))
            lines.append("  ```")
    lines.extend(["", "## Task Query Evidence", ""])
    for item in payload.get("repository_queries", [])[:12]:
        lines.append(f"### Query: `{markdown_escape_line(item.get('query', ''))}`")
        best = item.get("best", {})
        lines.append(f"- Return code: {best.get('returncode')}")
        lines.append(f"- OK: {best.get('ok')}")
        output = best.get("stdout") or best.get("stderr") or ""
        if output:
            lines.extend(["", "```text", output[:3000].rstrip(), "```", ""])
    lines.extend(["", "## Symbol Relationship Evidence", ""])
    for item in payload.get("symbol_relationships", [])[:12]:
        lines.append(f"### `{markdown_escape_line(item.get('symbol', ''))}`")
        for key in ("node", "callers", "callees", "impact"):
            result = item.get(key, {})
            lines.append(f"- {key}: ok={result.get('ok')} rc={result.get('returncode')}")
            output = result.get("stdout") or result.get("stderr") or ""
            if output:
                lines.extend(["", "```text", output[:2200].rstrip(), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def indent_text(text: str, prefix: str) -> list[str]:
    return [prefix + line for line in text.rstrip().splitlines()]


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        payload = collect_codegraph(args)
        print(f"Wrote CodeGraph perception: {resolve_path(args.out)}")
        print(payload.get("summary", {}))
        return 0
    except ScriptError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
