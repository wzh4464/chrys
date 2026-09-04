# Copyright (c) 2026 Chrys. All rights reserved.

"""Agent-profile activation contract for requirement clarification."""

from pathlib import Path

import pytest

from chrys.service.profiles.agents.loader import AgentProfileLoadError, load_profile_from_yaml
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    RequirementClarificationConfig,
    RequirementEnrichmentConfig,
)
from chrys.service.profiles.agents.serializer import profile_to_dict, save_profile


def test_requirement_clarification_defaults_disabled(tmp_path: Path) -> None:
    path = tmp_path / "default.yaml"
    path.write_text("name: default\n", encoding="utf-8")

    profile = load_profile_from_yaml(path)

    assert profile.requirement_clarification.enabled is False
    assert "requirement_clarification" not in profile_to_dict(profile)


def test_requirement_clarification_enabled_round_trip(tmp_path: Path) -> None:
    profile = AgentProfile(
        name="clarifying",
        requirement_clarification=RequirementClarificationConfig(enabled=True),
    )

    restored = load_profile_from_yaml(save_profile(profile, target_dir=tmp_path))

    assert profile_to_dict(profile)["requirement_clarification"] == {"enabled": True}
    assert restored.requirement_clarification.enabled is True


def test_requirement_clarification_imported_p0_round_trip(tmp_path: Path) -> None:
    profile = AgentProfile(
        name="clarifying-imported-p0",
        requirement_clarification=RequirementClarificationConfig(
            enabled=True,
            reuse_workspace_as_p0=True,
        ),
    )

    restored = load_profile_from_yaml(save_profile(profile, target_dir=tmp_path))

    assert restored.requirement_clarification.reuse_workspace_as_p0 is True


@pytest.mark.parametrize("body", ['enabled: "yes"', "enabled: 1", "proposal_count: 3"])
def test_requirement_clarification_rejects_non_contract_fields(tmp_path: Path, body: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(f"name: invalid\nrequirement_clarification:\n  {body}\n", encoding="utf-8")

    with pytest.raises(AgentProfileLoadError, match="requirement_clarification"):
        load_profile_from_yaml(path)


def test_acp_profile_cannot_enable_requirement_clarification(tmp_path: Path) -> None:
    path = tmp_path / "external.yaml"
    path.write_text(
        "name: external\nacp:\n  command: agent\nrequirement_clarification:\n  enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentProfileLoadError, match="cannot enable requirement_clarification"):
        load_profile_from_yaml(path)


def test_requirement_enrichment_enabled_round_trip(tmp_path: Path) -> None:
    profile = AgentProfile(
        name="enriching",
        requirement_enrichment=RequirementEnrichmentConfig(enabled=True),
    )

    restored = load_profile_from_yaml(save_profile(profile, target_dir=tmp_path))

    assert profile_to_dict(profile)["requirement_enrichment"] == {"enabled": True}
    assert restored.requirement_enrichment.enabled is True


def test_acp_profile_cannot_enable_requirement_enrichment(tmp_path: Path) -> None:
    path = tmp_path / "external.yaml"
    path.write_text(
        "name: external\nacp:\n  command: agent\nrequirement_enrichment:\n  enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentProfileLoadError, match="cannot enable requirement_enrichment"):
        load_profile_from_yaml(path)
