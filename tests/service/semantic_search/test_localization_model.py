# Copyright (c) 2026 Chrys. All rights reserved.

"""The localization search runs in-process, through a Chrys model client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chrys.foundation.config.settings import Settings
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.semantic_search.config import SemanticSearchConfig
from chrys.service.semantic_search.localization_model import (
    ChrysLocalizationModel,
    resolve_localization_model_profile,
)


@pytest.fixture
def indexed_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny repository plus the index the deterministic stage would have written."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "parser.py").write_text(
        "def parse_value(value):\n    return value\n\n\nclass Parser:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "src/parser.py",
                        "kind": "source",
                        "language": "python",
                        "is_test": False,
                        "units": [
                            {
                                "name": "parse_value",
                                "kind": "function",
                                "start_line": 1,
                                "end_line": 2,
                                "path": "src/parser.py",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return repo, index


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[Any] = []


def _profile() -> ModelProfile:
    return ModelProfile(id="cheap", name="Cheap", stream=False)


async def test_a_model_that_names_an_unvisited_file_yields_nothing(
    indexed_repo: tuple[Path, Path],
) -> None:
    """A location the search never visited is a guess, not a finding."""
    repo, index = indexed_repo
    agent = MagicMock()
    agent.run = _async_return(_Response('[{"file": "src/parser.py", "symbol": "parse_value"}]'))

    with patch("chrys.service.semantic_search.localization_model.Agent", return_value=agent):
        run = await ChrysLocalizationModel(_profile(), client=MagicMock()).localize(
            "fix parse_value",
            repo=repo,
            index_path=index,
            codegraph_path=None,
            config=SemanticSearchConfig(),
        )

    assert run is None


async def test_the_five_search_tools_are_offered_to_the_agent(
    indexed_repo: tuple[Path, Path],
) -> None:
    repo, index = indexed_repo
    captured: dict[str, Any] = {}
    agent = MagicMock()
    agent.run = _async_return(_Response("[]"))

    def _agent(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return agent

    with patch("chrys.service.semantic_search.localization_model.Agent", _agent):
        await ChrysLocalizationModel(_profile(), client=MagicMock()).localize(
            "fix parse_value",
            repo=repo,
            index_path=index,
            codegraph_path=None,
            config=SemanticSearchConfig(),
        )

    assert [tool.name for tool in captured["tools"]] == [
        "find_file",
        "find_code_definition",
        "find_code_content",
        "find_child_unit",
        "finish_search",
    ]


async def test_an_injected_client_is_reused_rather_than_created(
    indexed_repo: tuple[Path, Path],
) -> None:
    repo, index = indexed_repo
    client = MagicMock()
    agent = MagicMock()
    agent.run = _async_return(_Response("[]"))

    with (
        patch("chrys.service.semantic_search.localization_model.Agent", return_value=agent),
        patch("chrys.service.semantic_search.localization_model.create_client") as create_client,
    ):
        await ChrysLocalizationModel(_profile(), client=client).localize(
            "fix parse_value",
            repo=repo,
            index_path=index,
            codegraph_path=None,
            config=SemanticSearchConfig(),
        )

    create_client.assert_not_called()


async def test_the_search_is_bounded_by_the_config(indexed_repo: tuple[Path, Path]) -> None:
    """An unbounded search would turn a preflight into an autonomous session."""
    repo, index = indexed_repo
    client = MagicMock()
    agent = MagicMock()
    agent.run = _async_return(_Response("[]"))

    with patch("chrys.service.semantic_search.localization_model.Agent", return_value=agent):
        await ChrysLocalizationModel(_profile(), client=client).localize(
            "fix parse_value",
            repo=repo,
            index_path=index,
            codegraph_path=None,
            config=SemanticSearchConfig(max_iterations=7),
        )

    assert client.max_iterations == 7
    assert client.max_function_calls == 14


async def test_trace_events_are_reported_to_the_caller(indexed_repo: tuple[Path, Path]) -> None:
    repo, index = indexed_repo
    events: list[str] = []
    agent = MagicMock()
    agent.run = _async_return(_Response("[]"))

    with patch("chrys.service.semantic_search.localization_model.Agent", return_value=agent):
        await ChrysLocalizationModel(
            _profile(), client=MagicMock(), on_trace=lambda event, _data: events.append(event)
        ).localize(
            "fix parse_value",
            repo=repo,
            index_path=index,
            codegraph_path=None,
            config=SemanticSearchConfig(),
        )

    assert events[0] == "agent-start"
    assert "agent-complete" in events


# --------------------------------------------------------------------------
# model selection
# --------------------------------------------------------------------------


def test_the_setting_selects_a_cheaper_profile() -> None:
    cheap = ModelProfile(id="cheap", name="Cheap")
    active = ModelProfile(id="active", name="Active")
    registry = MagicMock()

    with patch("chrys.service.profiles.models.resolver.resolve_profile_selector", return_value=cheap):
        resolved = resolve_localization_model_profile(Settings(semantic_search_model_profile="cheap"), registry, active)

    assert resolved is cheap


def test_an_empty_setting_uses_the_active_model() -> None:
    active = ModelProfile(id="active", name="Active")

    resolved = resolve_localization_model_profile(Settings(), MagicMock(), active)

    assert resolved is active


def test_an_unknown_profile_falls_back_to_the_active_model() -> None:
    """A typo must degrade to the working model, not to no localization."""
    active = ModelProfile(id="active", name="Active")

    with patch("chrys.service.profiles.models.resolver.resolve_profile_selector", return_value=None):
        resolved = resolve_localization_model_profile(
            Settings(semantic_search_model_profile="typo"), MagicMock(), active
        )

    assert resolved is active


def _async_return(value: Any):
    async def _run(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return _run


def test_a_dangling_symlink_does_not_take_the_index_down(tmp_path: Path) -> None:
    """`chrys locate` on a repo with a broken symlink used to die on `stat`.

    Benchmark checkouts and vendored trees carry them routinely, and the whole
    pipeline runs through this script.
    """
    import subprocess
    import sys

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "dangling.py").symlink_to(tmp_path / "nonexistent")
    script = Path(__file__).resolve().parents[3] / "src/chrys/service/semantic_search/skill/scripts/build_index.py"
    out = tmp_path / "index.json"

    completed = subprocess.run(
        [sys.executable, str(script), "--repo", str(repo), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    indexed = json.loads(out.read_text(encoding="utf-8"))
    assert [entry["path"] for entry in indexed["files"]] == ["a.py"]


def test_the_fingerprint_ignores_the_run_own_artifacts(tmp_path: Path) -> None:
    """`chrys locate --artifact-dir` documents putting it inside the repository.

    The manifest is written last, so the next call's fingerprint sees a file
    the stored one could not — leaving the cache permanently invalid and
    re-running the full index and LLM search every time.
    """
    from chrys.service.semantic_search.output import repo_fingerprint

    repo = tmp_path / "repo"
    artifacts = repo / "artifacts"
    artifacts.mkdir(parents=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")

    before = repo_fingerprint(repo, exclude=artifacts)
    (artifacts / "manifest.json").write_text('{"mode": "llm"}', encoding="utf-8")

    assert repo_fingerprint(repo, exclude=artifacts) == before

    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")

    assert repo_fingerprint(repo, exclude=artifacts) != before
