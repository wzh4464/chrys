# Copyright (c) 2026 Chrys. All rights reserved.

"""Read one immutable Git requirement from a uv lockfile."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def locked_git_requirement(
    lock_path: Path,
    *,
    package_name: str,
    expected_repository: str,
) -> str:
    """Return a PEP 508 requirement after validating an immutable lock source."""
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)

    matches = [package for package in lock.get("package", []) if package.get("name") == package_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one locked {package_name!r} package, found {len(matches)}")

    source = matches[0].get("source")
    if not isinstance(source, dict) or set(source) != {"git"} or not isinstance(source["git"], str):
        raise ValueError(f"locked {package_name!r} source is not a Git source")

    parsed = urlsplit(source["git"])
    repository = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    if repository != expected_repository:
        raise ValueError(f"locked {package_name!r} repository is {repository!r}, expected {expected_repository!r}")

    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    commit = parsed.fragment
    if set(query) != {"rev"} or query["rev"] != [commit] or _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError(
            f"locked {package_name!r} source must have identical 40-character lowercase rev and commit fragment"
        )

    return f"{package_name} @ git+{expected_repository}@{commit}"


def main() -> int:
    """Print the validated locked requirement for build-script consumption."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    try:
        requirement = locked_git_requirement(
            args.lock,
            package_name=args.package,
            expected_repository=args.repository,
        )
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(f"{requirement}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
