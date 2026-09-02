# Copyright (c) 2026 Chrys. All rights reserved.

"""Validate and locate the pinned pact-core wheel shipped with this checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

_SCHEMA = "chrys/vendored-wheel/v1"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FIELDS = {
    "schema",
    "package",
    "version",
    "source_repository",
    "source_commit",
    "wheel",
    "sha256",
}


def validate_vendored_wheel(project_root: Path) -> Path:
    """Validate provenance, package metadata, and digest; return the wheel path."""
    root = project_root.resolve(strict=True)
    provenance_path = root / "vendor" / "pact-core.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or set(provenance) != _FIELDS:
        raise ValueError("pact-core provenance must contain exactly the v1 fields")
    if provenance["schema"] != _SCHEMA:
        raise ValueError(f"unsupported pact-core provenance schema: {provenance['schema']!r}")
    if provenance["package"] != "pact-core":
        raise ValueError("vendored package must be pact-core")
    if provenance["source_repository"] != "https://github.com/SELab-Leibniz/pact.git":
        raise ValueError("unexpected pact-core source repository")
    if _COMMIT_PATTERN.fullmatch(provenance["source_commit"]) is None:
        raise ValueError("pact-core source commit must be a lowercase 40-character Git commit")
    if _SHA256_PATTERN.fullmatch(provenance["sha256"]) is None:
        raise ValueError("pact-core wheel SHA-256 must be lowercase hexadecimal")

    wheel_path = (root / provenance["wheel"]).resolve(strict=True)
    wheel_path.relative_to(root)
    if not wheel_path.is_file():
        raise ValueError("vendored pact-core wheel is not a regular file")
    with wheel_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != provenance["sha256"]:
        raise ValueError(f"vendored pact-core wheel SHA-256 mismatch: {digest}")

    with zipfile.ZipFile(wheel_path) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("vendored pact-core wheel must contain exactly one METADATA file")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    if metadata["Name"] != provenance["package"] or metadata["Version"] != provenance["version"]:
        raise ValueError("vendored pact-core wheel metadata does not match provenance")
    return wheel_path


def main() -> int:
    """Print the validated absolute wheel path for build-script consumption."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        wheel_path = validate_vendored_wheel(args.project_root)
    except (OSError, json.JSONDecodeError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    sys.stdout.write(f"{wheel_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
