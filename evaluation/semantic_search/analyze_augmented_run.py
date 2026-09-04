#!/usr/bin/env python3
"""Offline summary for semantic-search Augmented Requirement artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import ScriptError, load_json, read_text, resolve_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augmented-requirement", required=True)
    parser.add_argument("--routes", required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        augmented = resolve_path(args.augmented_requirement)
        routes_path = resolve_path(args.routes)
        text = read_text(augmented)
        routes = load_json(routes_path)
        generation = routes.get("generation", {})
        print(f"Augmented Requirement: {augmented}")
        print(f"Length: {len(text.splitlines())} lines, {len(text)} chars")
        print(f"Routes: {len(routes.get('routes', []))}")
        if generation:
            print(f"Generation: {generation.get('mode', 'unknown')} - {generation.get('note', '')}")
        for route in routes.get("routes", []):
            path = Path(route["path"])
            status = "present" if path.exists() else "missing"
            print(
                f"- {route['priority']}: {route['topic']} -> {path} [{status}] summary={route.get('summary_line_count', 0)}"
            )
        return 0
    except ScriptError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
