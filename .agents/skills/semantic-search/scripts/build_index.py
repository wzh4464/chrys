# Copyright (c) 2026 Chrys. All rights reserved.

"""Build a lightweight semantic-search repository index for requirement augmentation."""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Any

from _common import (
    FORMAT_INDEX,
    ScriptError,
    append_trace,
    classify_file,
    now_iso,
    path_tokens,
    read_text,
    resolve_output_path,
    resolve_path,
    sha1_path,
    should_skip_dir,
    should_skip_file,
    stable_unique,
    tokenize,
    write_json,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", required=True, help="Repo root, or name=/abs/repo. Repeatable.")
    parser.add_argument("--out", required=True, help="Output index path.")
    parser.add_argument("--max-file-bytes", type=int, default=350_000, help="Skip content scanning above this size.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional safety cap for indexed files.")
    return parser.parse_args(argv)


def parse_repo(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ScriptError(f"invalid repo spec: {raw}")
    else:
        value = raw
        name = resolve_path(value).name or "repo"
    root = resolve_path(value)
    if not root.is_dir():
        raise ScriptError(f"repo path does not exist or is not a directory: {root}")
    return name, root


def discover_files(root: Path, max_files: int) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            name for name in dirnames if not should_skip_dir(name) and not (current / name).is_symlink()
        ]
        for filename in filenames:
            path = current / filename
            if path.is_symlink() or should_skip_file(path):
                continue
            files.append(path)
            if max_files and len(files) >= max_files:
                return sorted(files)
    return sorted(files)


def is_probably_text(path: Path, sample_size: int = 4096) -> bool:
    try:
        sample = path.read_bytes()[:sample_size]
    except OSError:
        return False
    if b"\0" in sample:
        return False
    return True


def file_preview(path: Path, max_bytes: int) -> tuple[str, list[str]]:
    try:
        if path.stat().st_size > max_bytes or not is_probably_text(path):
            return "", []
    except OSError:
        return "", []
    text = read_text(path, max_chars=max_bytes)
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(stripped[:220])
        if len(lines) >= 24:
            break
    return "\n".join(lines)[:5000], tokenize(text[:80_000])


def extract_python_symbols(relative: str, text: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    symbols: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.append(
                {
                    "name": node.name,
                    "kind": "class",
                    "file": relative,
                    "line": node.lineno,
                    "signature": f"class {node.name}",
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = parent_class_name(tree, node)
            name = f"{parent}.{node.name}" if parent else node.name
            symbols.append(
                {
                    "name": name,
                    "kind": "method" if parent else "function",
                    "file": relative,
                    "line": node.lineno,
                    "signature": first_line(text, node.lineno),
                }
            )
    return symbols[:250]


def parent_class_name(tree: ast.AST, target: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and target in getattr(node, "body", []):
            return node.name
    return None


def first_line(text: str, line: int) -> str:
    lines = text.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()[:240]
    return ""


TEXT_SYMBOL_RE = re.compile(
    r"^\s*(?:public|protected|private|static|final|abstract|async|pub|override|def|class|trait|object|struct|enum|interface|fn|func|void|int|long|bool|char|double|float|PyObject|\w[\w:<>,\[\]\s*&]+)?\s*"
    r"(class|trait|object|struct|enum|interface|def|fn|func)?\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|\{|extends|implements|:)?"
)


def extract_text_symbols(relative: str, text: str, language: str) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#", "*", "/*")):
            continue
        match = TEXT_SYMBOL_RE.match(line)
        if not match:
            continue
        marker, name = match.groups()
        if name in {"if", "for", "while", "switch", "return", "catch", "else"}:
            continue
        kind = marker or "symbol"
        if language in {"c", "cpp"} and "(" in line and ";" not in line:
            kind = "function"
        symbols.append({"name": name, "kind": kind, "file": relative, "line": line_number, "signature": stripped[:240]})
        if len(symbols) >= 250:
            break
    return symbols


def extract_symbols(relative: str, path: Path, language: str, max_bytes: int) -> list[dict[str, Any]]:
    try:
        if path.stat().st_size > max_bytes:
            return []
    except OSError:
        return []
    text = read_text(path, max_chars=max_bytes)
    if language == "python":
        return extract_python_symbols(relative, text)
    if language:
        return extract_text_symbols(relative, text, language)
    return []


def build_index(repos: list[tuple[str, Path]], max_file_bytes: int, max_files: int) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    language_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for repo_name, root in repos:
        for path in discover_files(root, max_files):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            relative = path.relative_to(root).as_posix()
            classification = classify_file(path, relative)
            preview, content_terms = file_preview(path, max_file_bytes)
            file_symbols = extract_symbols(relative, path, classification["language"], max_file_bytes)
            symbol_terms = [symbol["name"] for symbol in file_symbols]
            record = {
                "repo": repo_name,
                "path": relative,
                "kind": classification["kind"],
                "language": classification["language"],
                "is_test": classification["is_test"],
                "is_generated": classification["is_generated"],
                "top_level": classification["top_level"],
                "size": size,
                "sha1": sha1_path(path) if size <= max_file_bytes else "",
                "terms": stable_unique([*path_tokens(relative), *content_terms[:160], *tokenize(" ".join(symbol_terms))])[:260],
                "symbols": file_symbols[:80],
                "preview": preview,
            }
            files.append(record)
            symbols.extend({**symbol, "repo": repo_name, "language": classification["language"]} for symbol in file_symbols)
            language_counts[classification["language"] or "text"] = language_counts.get(classification["language"] or "text", 0) + 1
            kind_counts[classification["kind"]] = kind_counts.get(classification["kind"], 0) + 1
    return {
        "format": FORMAT_INDEX,
        "created_at": now_iso(),
        "repositories": [{"name": name, "root": str(root)} for name, root in repos],
        "files": files,
        "symbols": symbols,
        "stats": {
            "repo_count": len(repos),
            "file_count": len(files),
            "symbol_count": len(symbols),
            "language_counts": language_counts,
            "kind_counts": kind_counts,
        },
    }


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        repos = [parse_repo(raw) for raw in args.repo]
        out = resolve_output_path(args.out)
        index = build_index(repos, args.max_file_bytes, args.max_files)
        write_json(out, index)
        append_trace("build-index", {"out": str(out), "stats": index["stats"]})
        print(f"Wrote semantic-search light index: {out}")
        print(index["stats"])
        return 0
    except ScriptError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
