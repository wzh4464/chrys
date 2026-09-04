#!/usr/bin/env python3
"""What a PACT campaign runs to accept a mission on a DeepSWE task, by language.

The long-horizon track delegates to a campaign only when the workspace has a
deterministic verification command (``pact.verify_command``); DeepSWE tasks carry
none, so without this every run ended at the repaired baseline and PACT never saw
the goal contract and plan that clarification had produced for it. The hidden
DeepSWE tests are not in the task image; these run the repository's own suite at
the base commit.
"""

from __future__ import annotations

import re
from pathlib import Path

VERIFY_BY_LANGUAGE: dict[str, str] = {
    "go": "go test ./...",
    "python": "python -m pytest -q -x -p no:cacheprovider",
    "typescript": "npm test --silent",
    "javascript": "npm test --silent",
    "rust": "cargo test -q",
}


def verify_command_for(language: str | None) -> str:
    """The verify command for a task language, or an empty string when there is none."""
    return VERIFY_BY_LANGUAGE.get(str(language or "").strip().lower(), "")


# --- the task's own verifier, minus the hidden tests -------------------------------
#
# Every DeepSWE task ships tests/test.sh, the script its Harbor verifier runs: a
# "base" run of the repository's pass-to-pass tests and a "new" run of the hidden
# fail-to-pass tests, each tuned to the repository (the right runner, the right
# package of a monorepo, the excluded flaky suites, the environment variables). The
# language defaults above fail at the base commit for half of the tasks (missing
# optional dependencies, go.work layouts, snapshot suites), which blocks the campaign
# before it starts; the base run of test.sh is what the task itself considers green.


_RUNNER_HINTS = (
    "pytest",
    "vitest",
    "jest",
    "mocha",
    "go test",
    "go build",
    "go vet",
    "cargo ",
    "pnpm ",
    "npm ",
    "npx ",
    "yarn ",
    "bun ",
    "deno ",
    "python -m",
    "python3 -m",
    "node ",
    "make ",
    "tox ",
    "nox ",
    "bash ",
)
_SKIP_PREFIXES = (
    "log ",
    "require_cmd",
    "cp ",
    "rm ",
    "mv ",
    "mkdir",
    "cat ",
    "echo ",
    "sleep ",
    "if ",
    "fi",
    "for ",
    "done",
    "then",
    "else",
    "elif",
    "[ ",
    "[[",
    "case ",
    "esac",
    "export ",
    "convert_to_ctrf",
    "ctrf_check",
    "junit-to-ctrf",
    "go-ctrf-json-reporter",
    ":",
    "set ",
    "trap ",
    "cd ",
    "python3 /tests/",
    "python /tests/",
    "chmod ",
    "touch ",
    "ls ",
    "find ",
    "wc ",
    "grep ",
    "sed ",
    "awk ",
    "tee ",
    "exit",
    "return",
    "}",
    "{",
    "while ",
    "until ",
    "local ",
    "unset ",
)
_EXCLUDE_OPTIONS = (
    "--exclude",
    "--ignore",
    "--testPathIgnorePatterns",
    "--deselect",
    "--skip",
    "-t ",
    "--testNamePattern",
)
_REPORT_TOKENS = [
    re.compile(r"""\s--reporters?=(?:junit|"?\$CTRF_REPORTER"?|\S*ctrf\S*)"""),
    re.compile(r"""\s--reporters?\s+(?:junit|"?\$CTRF_REPORTER"?)"""),
    re.compile(r"""\s--outputFile(?:=|\s+)[^\s"')]+"""),
    re.compile(r"""\s--junit-?xml(?:=|\s+)[^\s"')]+"""),
    re.compile(r"""\s--junit-path(?:=|\s+)[^\s"')]+"""),
    re.compile(r"""\s--log-junit(?:=|\s+)[^\s"')]+"""),
    re.compile(r"""\s--profile\s+junit\b"""),
    re.compile(r"""\s-json\b"""),
    re.compile(r"""\s--json\b"""),
]
_REDIRECTS = [
    re.compile(r"""\s[12]?>>?\s*(?:"[^"]*"|'[^']*'|[^\s"']+)"""),
    re.compile(r"""\s[12]?>&[12]"""),
    re.compile(r"""\s&>\s*\S+"""),
]


def _set_e_block(script: str) -> str:
    blocks = re.findall(r"^set \+e\s*$(.*?)^set -e\s*$", script, re.DOTALL | re.MULTILINE)
    return "\n".join(blocks) if blocks else script


def _join_continuations(block: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw in block.splitlines():
        if raw.rstrip().endswith("\\"):
            pending += raw.rstrip()[:-1] + " "
            continue
        lines.append(pending + raw)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def _cut_pipeline(line: str) -> str:
    quote = ""
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = ""
        elif char in ("'", '"'):
            quote = char
        elif char == "|":
            return line[:index]
    return line


def hidden_test_paths(test_patch: str) -> list[str]:
    """Paths the hidden test patch adds or touches (``+++ b/<path>`` lines)."""
    return [m.group(1).strip() for m in re.finditer(r"^\+\+\+ b/(\S+)", test_patch, re.MULTILINE)]


def _mentions_hidden_test(line: str, hidden: list[str]) -> bool:
    tokens = [t for t in line.split() if not t.startswith(_EXCLUDE_OPTIONS)]
    # A -t/--exclude style option may carry its pattern in the next token.
    kept: list[str] = []
    skip_next = False
    for token in line.split():
        if skip_next:
            skip_next = False
            continue
        if token in ("-t", "--exclude", "--ignore", "--deselect", "--testPathIgnorePatterns", "--testNamePattern"):
            skip_next = True
            continue
        if token.startswith(_EXCLUDE_OPTIONS):
            continue
        kept.append(token)
    rest = " ".join(kept or tokens)
    names = set()
    for path in hidden:
        names.add(path)
        names.add(Path(path).name)
    return any(name in rest for name in names)


def _clean(line: str) -> str:
    line = _cut_pipeline(line)
    for pattern in _REPORT_TOKENS:
        line = pattern.sub("", line)
    for pattern in _REDIRECTS:
        line = pattern.sub("", line)
    line = line.replace("/app/", "./")
    line = re.sub(r"\s+", " ", line).strip().rstrip(";&").strip()
    return line


def verify_from_test_sh(script: str, test_patch: str, language: str | None) -> str:
    """The base run of a task's verifier as a verify command, or the language default."""
    hidden = hidden_test_paths(test_patch)
    commands: list[str] = []
    for raw in _join_continuations(_set_e_block(script)):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("run_log ")
        if line.startswith(_SKIP_PREFIXES) or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$", line):
            continue
        if not any(hint in line for hint in _RUNNER_HINTS):
            continue
        if _mentions_hidden_test(line, hidden):
            continue
        cleaned = _clean(line)
        if cleaned and "$" not in cleaned:
            commands.append(cleaned)
    return " && ".join(commands) if commands else verify_command_for(language)


def verify_command_for_task(task_dir: Path, language: str | None) -> str:
    """Derive the verify command from ``<task>/tests/test.sh``; the language default otherwise."""
    script = task_dir / "tests" / "test.sh"
    if not script.is_file():
        return verify_command_for(language)
    patch = task_dir / "tests" / "test.patch"
    return verify_from_test_sh(
        script.read_text(encoding="utf-8", errors="replace"),
        patch.read_text(encoding="utf-8", errors="replace") if patch.is_file() else "",
        language,
    )


# --- the task's own runner, in base mode ------------------------------------------
#
# The hidden test patch of every task adds ``test.sh`` at the repository root: the
# runner the verifier calls in ``base`` and ``new`` mode. Its base branch is the
# repository's own regression suite, tuned per task; shipped to the agent image with
# the lines that name the hidden tests blanked, invoked as ``bash <script> base``,
# it is the most faithful verify command a campaign can have.

VERIFY_SCRIPT_PATH = "/opt/deepswe_verify.sh"
VERIFY_COMMAND = f"bash {VERIFY_SCRIPT_PATH} base"


def hidden_runner_script(test_patch: str) -> str | None:
    """The ``test.sh`` the hidden test patch adds at the repository root, or None."""
    match = re.search(r"^\+\+\+ b/test\.sh\n(.*?)(?=^diff --git |\Z)", test_patch, re.DOTALL | re.MULTILINE)
    if match is None:
        return None
    body = []
    for line in match.group(1).splitlines():
        if line.startswith(("@@", "+++", "\\")):
            continue
        if line.startswith("+"):
            body.append(line[1:])
    text = "\n".join(body).strip("\n")
    return text + "\n" if text else None


def sanitize_runner(script: str, hidden: list[str]) -> str:
    """Point the runner's references to hidden tests at paths that do not exist.

    The base branch may exclude the hidden test by name (``--ignore=``, ``--exclude``);
    an exclusion of a missing path is harmless, so the shell structure and the base
    command survive intact while the hidden test's name does not reach the image.
    """
    replacements: list[tuple[str, str]] = []
    for path in hidden:
        if path == "test.sh":
            continue
        name = Path(path).name
        suffix = "".join(Path(name).suffixes)
        placeholder = f"__hidden__{suffix}"
        replacements.append((path, str(Path(path).with_name(placeholder))))
        replacements.append((name, placeholder))
    # Longest first so a full path is rewritten before its basename.
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)
    for old, new in replacements:
        script = script.replace(old, new)
    # The runner lived at the repository root and finds it through its own location;
    # installed under /opt it must use the working directory instead.
    script = _DIRNAME_OF_SELF.sub(".", script)
    return script if script.endswith("\n") else script + "\n"


_DIRNAME_OF_SELF = re.compile(r"""\$\(\s*dirname\s+"?(?:\$0|\$\{BASH_SOURCE\[0\]\}|\$BASH_SOURCE)"?\s*\)""")
_NEW_MODE_HINT = re.compile(r"new", re.IGNORECASE)


def runner_base_needs_hidden_file(sanitized: str) -> bool:
    """True when a line outside the runner's new-mode branch still relies on a hidden file.

    An exclusion of a placeholder path is harmless; running or importing one is not
    (mashumaro's runner is ``python3 test.py base`` and test.py is itself hidden).
    """
    lines = sanitized.splitlines()
    for index, line in enumerate(lines):
        if "__hidden__" not in line:
            continue
        # The new-mode branch names the hidden test by design; its marker (`new)`,
        # `= "new"`, `NEW_TEST=`) sits on the line or a few lines above it.
        if any(_NEW_MODE_HINT.search(previous) for previous in lines[max(0, index - 3) : index + 1]):
            continue
        tokens = [token for token in line.split() if "__hidden__" in token]
        if all(
            token.startswith(_EXCLUDE_OPTIONS) or token.lstrip("\"'").startswith(("**/", "'**/")) for token in tokens
        ):
            continue
        return True
    return False


def verify_plan_for_task(task_dir: Path, language: str | None) -> tuple[str, str]:
    """(verify command, script body) for a task; both empty when the task cannot be verified.

    The command is always ``bash /opt/deepswe_verify.sh base``; the script is the
    task's sanitized runner, else a one-line script around the verifier-derived or
    language-default command.
    """
    patch = task_dir / "tests" / "test.patch"
    hidden: list[str] = []
    if patch.is_file():
        patch_text = patch.read_text(encoding="utf-8", errors="replace")
        hidden = hidden_test_paths(patch_text)
        runner = hidden_runner_script(patch_text)
        if runner:
            sanitized = sanitize_runner(runner, hidden)
            if not runner_base_needs_hidden_file(sanitized):
                return VERIFY_COMMAND, sanitized
    command = verify_command_for_task(task_dir, language)
    if not command:
        return "", ""
    return VERIFY_COMMAND, f"#!/usr/bin/env bash\n# derived from the task's verifier (base run)\nset -e\n{command}\n"
