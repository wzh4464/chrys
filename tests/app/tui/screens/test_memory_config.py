# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the Memory configuration panel."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Checkbox, Input, Label, Static

from chrys.app.tui.screens.agents.panels.memory import MemoryConfigPanel, MemoryFileCard, MemoryFolderCard
from chrys.service.profiles.agents.schema import MemoryConfig


class _MemoryPanelApp(App):
    def compose(self) -> ComposeResult:
        yield Static("placeholder")


def _set_fake_platform(monkeypatch: pytest.MonkeyPatch, os_name: str) -> None:
    fake_platform = type(
        "P",
        (),
        {
            "is_macos": os_name == "macos",
            "is_windows": os_name == "windows",
            "is_linux": os_name == "linux",
        },
    )()
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)


async def _mount_panel(panel: MemoryConfigPanel, pilot) -> list[MemoryFileCard]:
    await pilot.app.mount(panel)
    await pilot.pause()
    return list(panel.query(MemoryFileCard))


async def _wait_for_memory_cards(
    panel: MemoryConfigPanel,
    pilot,
    *,
    files: int,
    folders: int,
) -> tuple[list[MemoryFileCard], list[MemoryFolderCard]]:
    for _ in range(20):
        file_cards = list(panel.query(MemoryFileCard))
        folder_cards = list(panel.query(MemoryFolderCard))
        if len(file_cards) < files or len(folder_cards) < folders:
            await pilot.pause(0.05)
            continue
        file_inputs_ready = all(list(file_cards[index].query(f"#mem-file-path-{index}")) for index in range(files))
        folder_inputs_ready = all(
            list(folder_cards[index].query(f"#mem-folder-path-{index}")) for index in range(folders)
        )
        if file_inputs_ready and folder_inputs_ready:
            await pilot.pause()
            return list(panel.query(MemoryFileCard)), list(panel.query(MemoryFolderCard))
        await pilot.pause(0.05)
    raise AssertionError(f"Expected {files} memory file cards and {folders} memory folder cards to be mounted")


@pytest.mark.parametrize(
    (
        "os_name",
        "absolute_file_placeholder",
        "relative_file_placeholder",
        "absolute_folder_placeholder",
        "relative_folder_placeholder",
    ),
    [
        (
            "windows",
            r"C:\path\to\file.md",
            r"relative\path\to\file.md",
            r"C:\path\to\folder",
            r"relative\path\to\folder",
        ),
        ("macos", "/Users/you/file.md", "relative/path/to/file.md", "/Users/you/folder", "relative/path/to/folder"),
        ("linux", "/home/you/file.md", "relative/path/to/file.md", "/home/you/folder", "relative/path/to/folder"),
    ],
)
@pytest.mark.asyncio
async def test_memory_path_placeholders_are_platform_specific(
    os_name: str,
    absolute_file_placeholder: str,
    relative_file_placeholder: str,
    absolute_folder_placeholder: str,
    relative_folder_placeholder: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_fake_platform(monkeypatch, os_name)
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(files=[""], folders=[""]))
        await _mount_panel(panel, pilot)

        file_input = panel.query_one("#mem-file-path-0", Input)
        checkbox = panel.query_one("#mem-file-rel-0", Checkbox)
        folder_input = panel.query_one("#mem-folder-path-0", Input)
        folder_checkbox = panel.query_one("#mem-folder-rel-0", Checkbox)

        # Empty rows infer absolute mode for files and folders alike.
        assert file_input.placeholder == absolute_file_placeholder
        assert folder_input.placeholder == absolute_folder_placeholder

        checkbox.toggle()
        folder_checkbox.toggle()
        await pilot.pause()

        assert file_input.placeholder == relative_file_placeholder
        assert folder_input.placeholder == relative_folder_placeholder


@pytest.mark.asyncio
async def test_add_memory_paths_insert_new_cards_before_existing_paths() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(files=["docs/a.md"], folders=["notes"]))
        await _mount_panel(panel, pilot)

        panel.query_one("#mem-add-file", Button).press()
        panel.query_one("#mem-add-folder", Button).press()

        file_cards, folder_cards = await _wait_for_memory_cards(panel, pilot, files=2, folders=2)
        assert file_cards[0].query_one("#mem-file-path-0", Input).value == ""
        assert file_cards[1].query_one("#mem-file-path-1", Input).value == "docs/a.md"
        assert folder_cards[0].query_one("#mem-folder-path-0", Input).value == ""
        assert folder_cards[1].query_one("#mem-folder-path-1", Input).value == "notes"
        assert panel.get_config().files == ["docs/a.md"]
        assert panel.get_config().folders == ["notes"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("notes.md", True),
        ("./notes.md", True),
        ("../shared/notes.md", True),
        ("/abs/notes.md", False),
        ("~/notes.md", False),
        (r"C:\notes.md", False),
    ],
)
@pytest.mark.asyncio
async def test_memory_file_relative_checkbox_inferred_from_saved_path(path: str, expected: bool) -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        cards = await _mount_panel(MemoryConfigPanel(MemoryConfig(files=[path])), pilot)

        assert cards[0].query_one("#mem-file-rel-0", Checkbox).value is expected


@pytest.mark.asyncio
async def test_memory_file_relative_toggle_hides_and_restores_browse() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        cards = await _mount_panel(MemoryConfigPanel(MemoryConfig(files=["/tmp/notes.md"])), pilot)
        checkbox = cards[0].query_one("#mem-file-rel-0", Checkbox)
        browse = cards[0].query_one("#mem-file-browse-0", Button)

        assert browse.display is True

        checkbox.toggle()
        await pilot.pause()
        assert browse.display is False

        checkbox.toggle()
        await pilot.pause()
        assert browse.display is True


@pytest.mark.asyncio
async def test_memory_file_relative_checkbox_is_below_path_row_and_folders_match() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(files=["/tmp/notes.md"], folders=["docs"]))
        cards = await _mount_panel(panel, pilot)
        card = cards[0]
        path_input = card.query_one("#mem-file-path-0", Input)
        browse = card.query_one("#mem-file-browse-0", Button)
        checkbox = card.query_one("#mem-file-rel-0", Checkbox)

        assert isinstance(path_input.parent, Horizontal)
        assert browse.parent is path_input.parent
        assert checkbox.parent is not path_input.parent
        # Folder rows carry the same path-mode chrome as file rows.
        folder_checkbox = panel.query_one("#mem-folder-rel-0", Checkbox)
        assert folder_checkbox.value is True
        # Relative mode hides Browse (same as file rows).
        assert panel.query_one("#mem-folder-browse-0", Button).display is False


@pytest.mark.asyncio
async def test_memory_file_validate_rejects_absolute_path_when_relative_checked() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(files=["/abs/notes.md"]), workspace_cwd="/repo")
        cards = await _mount_panel(panel, pilot)
        cards[0].query_one("#mem-file-rel-0", Checkbox).toggle()
        await pilot.pause()

        errors = panel.validate()

    assert "Memory File 1: '/abs/notes.md' is absolute; turn off Workspace relative." in errors


@pytest.mark.asyncio
async def test_memory_file_validate_rejects_relative_path_when_unchecked() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(files=["notes.md"]), workspace_cwd="/repo")
        cards = await _mount_panel(panel, pilot)
        cards[0].query_one("#mem-file-rel-0", Checkbox).toggle()
        await pilot.pause()

        errors = panel.validate()

    assert "Memory File 1: 'notes.md' is relative; enable Workspace relative." in errors


@pytest.mark.asyncio
async def test_memory_folder_validate_rejects_absolute_path_when_relative_checked() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(folders=["/abs/notes"]), workspace_cwd="/repo")
        await pilot.app.mount(panel)
        await _wait_for_memory_cards(panel, pilot, files=0, folders=1)
        panel.query_one("#mem-folder-rel-0", Checkbox).toggle()
        await pilot.pause()

        errors = panel.validate()

    assert "Memory Folder 1: '/abs/notes' is absolute; turn off Workspace relative." in errors


@pytest.mark.asyncio
async def test_memory_folder_validate_rejects_relative_path_when_unchecked() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(folders=["notes"]), workspace_cwd="/repo")
        await pilot.app.mount(panel)
        await _wait_for_memory_cards(panel, pilot, files=0, folders=1)
        panel.query_one("#mem-folder-rel-0", Checkbox).toggle()
        await pilot.pause()

        errors = panel.validate()

    assert "Memory Folder 1: 'notes' is relative; enable Workspace relative." in errors


@pytest.mark.asyncio
async def test_memory_folder_validate_allows_absolute_path_when_unchecked() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(folders=["/abs/notes"]), workspace_cwd="/repo")
        await pilot.app.mount(panel)
        await _wait_for_memory_cards(panel, pilot, files=0, folders=1)

        errors = panel.validate()

    assert errors == []


@pytest.mark.asyncio
async def test_memory_file_validate_allows_absolute_path_when_unchecked() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(files=["/abs/notes.md"]), workspace_cwd="/repo")
        await _mount_panel(panel, pilot)

        errors = panel.validate()

    assert errors == []


@pytest.mark.asyncio
async def test_memory_file_preview_handles_resolve_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(files=["notes.md"]), workspace_cwd="/repo")
        cards = await _mount_panel(panel, pilot)

        def fail_resolve(self: Path) -> Path:
            raise OSError("resolve failed")

        monkeypatch.setattr(Path, "resolve", fail_resolve)
        cards[0]._refresh_preview()
        await pilot.pause()

        preview = cards[0].query_one("#mem-file-preview-0", Label)
        expected_path = str(Path("/repo") / "notes.md")
        assert f"Current: {expected_path}" in str(preview.content)
        assert "Missing in current workspace" in str(preview.content)


@pytest.mark.asyncio
async def test_memory_file_foreign_absolute_preview_reports_missing() -> None:
    app = _MemoryPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        cards = await _mount_panel(MemoryConfigPanel(MemoryConfig(files=[r"C:\notes.md"])), pilot)
        preview = cards[0].query_one("#mem-file-preview-0", Label)

        assert "File does not exist" in str(preview.content)
        assert "Missing in current workspace" not in str(preview.content)


@pytest.mark.asyncio
async def test_memory_file_browse_initial_path_prefers_workspace_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _MemoryPanelApp()
    pushed: dict[str, object] = {}

    async with app.run_test(size=(120, 40)) as pilot:
        panel = MemoryConfigPanel(MemoryConfig(files=[""]), workspace_cwd=str(tmp_path))
        cards = await _mount_panel(panel, pilot)

        def fake_push_screen(screen: object, callback: object | None = None) -> None:
            pushed["screen"] = screen
            pushed["callback"] = callback

        monkeypatch.setattr(app, "push_screen", fake_push_screen)
        cards[0].query_one("#mem-file-browse-0", Button).press()
        await pilot.pause()

    assert pushed["screen"]._initial_path == str(tmp_path)
