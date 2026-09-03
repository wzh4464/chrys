# Copyright (c) 2026 Chrys. All rights reserved.

"""Profile-level ``routing`` configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.service.profiles.agents.loader import AgentProfileLoadError, load_profile_from_yaml


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "Sample.yaml"
    path.write_text(f"name: Sample\ndescription: Sample\n{body}", encoding="utf-8")
    return path


def test_routing_defaults_to_off(tmp_path: Path) -> None:
    profile = load_profile_from_yaml(_write(tmp_path, ""))

    routing = profile.routing
    assert routing.mode == "off"
    assert routing.target_profile == ""
    assert routing.classifier == "both"
    assert routing.min_confidence == pytest.approx(0.7)
    assert routing.inherit is True
    assert routing.stale_after_seconds == pytest.approx(1800.0)
    assert routing.long_horizon.localization is True
    assert routing.long_horizon.clarification is True
    assert routing.long_horizon.pact_tool == "chrys_pact"
    assert routing.long_horizon.require_pact is False


def test_routing_reads_every_field(tmp_path: Path) -> None:
    profile = load_profile_from_yaml(
        _write(
            tmp_path,
            """routing:
  mode: auto
  target_profile: LongHorizon
  classifier: heuristic
  min_confidence: 0.85
  inherit: false
  stale_after_seconds: 600
  long_horizon:
    localization: false
    clarification: false
    pact_tool: other_tool
    require_pact: true
""",
        )
    )

    routing = profile.routing
    assert routing.mode == "auto"
    assert routing.target_profile == "LongHorizon"
    assert routing.classifier == "heuristic"
    assert routing.min_confidence == pytest.approx(0.85)
    assert routing.inherit is False
    assert routing.stale_after_seconds == pytest.approx(600.0)
    assert routing.long_horizon.localization is False
    assert routing.long_horizon.clarification is False
    assert routing.long_horizon.pact_tool == "other_tool"
    assert routing.long_horizon.require_pact is True


@pytest.mark.parametrize(
    "body",
    [
        "routing:\n  mode: sometimes\n",
        "routing:\n  target_profile: 123\n",
        "routing:\n  classifier: magic\n",
        "routing:\n  min_confidence: 1.5\n",
        "routing:\n  min_confidence: -0.1\n",
        "routing:\n  inherit: yes-please\n",
        "routing:\n  stale_after_seconds: -1\n",
        "routing:\n  stale_after_seconds: 0\n",
        "routing:\n  unknown_key: 1\n",
        "routing:\n  long_horizon:\n    unknown_key: 1\n",
        "routing:\n  long_horizon:\n    pact_tool: 5\n",
        "routing:\n  long_horizon:\n    localization: maybe\n",
        "routing: not-a-mapping\n",
    ],
)
def test_invalid_routing_is_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(AgentProfileLoadError):
        load_profile_from_yaml(_write(tmp_path, body))


def test_an_empty_pact_tool_is_rejected(tmp_path: Path) -> None:
    """An empty tool name would make the readiness veto silently unsatisfiable."""
    with pytest.raises(AgentProfileLoadError):
        load_profile_from_yaml(_write(tmp_path, 'routing:\n  long_horizon:\n    pact_tool: ""\n'))


@pytest.mark.parametrize("spelling", ["off", "OFF", "Off", '"off"'])
def test_the_yaml_boolean_off_trap_is_handled(tmp_path: Path, spelling: str) -> None:
    """YAML 1.1 resolves every casing of a bare ``off`` to False.

    The documented spelling is bare, so all of them have to mean the mode.
    """
    profile = load_profile_from_yaml(_write(tmp_path, f"routing:\n  mode: {spelling}\n"))

    assert profile.routing.mode == "off"


def test_a_boolean_true_mode_is_still_rejected(tmp_path: Path) -> None:
    with pytest.raises(AgentProfileLoadError):
        load_profile_from_yaml(_write(tmp_path, "routing:\n  mode: on\n"))


def test_an_acp_profile_cannot_enable_routing(tmp_path: Path) -> None:
    """An external ACP sub-agent never opens a turn, so it can never route one."""
    path = tmp_path / "Acp.yaml"
    path.write_text(
        "name: Acp\ndescription: Acp\nsub_agent_only: true\nacp:\n  command: chrys\n"
        "  args: [pact-agent]\nrouting:\n  mode: auto\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentProfileLoadError):
        load_profile_from_yaml(path)


def test_an_acp_profile_may_declare_routing_off(tmp_path: Path) -> None:
    path = tmp_path / "Acp.yaml"
    path.write_text(
        "name: Acp\ndescription: Acp\nsub_agent_only: true\nacp:\n  command: chrys\n"
        "  args: [pact-agent]\nrouting:\n  mode: off\n",
        encoding="utf-8",
    )

    assert load_profile_from_yaml(path).routing.mode == "off"
