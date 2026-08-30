# Copyright (c) 2026 Chrys. All rights reserved.

"""Regression tests for file-edit tool renderers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rich.color import Color, ColorSystem
from textual.app import App, ComposeResult
from textual.content import Content
from textual.geometry import Offset, Region
from textual.selection import Selection
from textual.widgets import Static

from chrys.app.tui.support.gc_freeze import GcAbsorbReason, GcAbsorbRequested
from chrys.app.tui.theme import (
    ANSI_THEMES,
    CHRYS_ANSI_THEME,
    CHRYS_THEME,
    CHRYS_THEMES,
    TUI_VARIABLE_DEFAULTS,
)
from chrys.app.tui.widgets.chat.file_snapshot import (
    FileSnapshotRef,
    file_snapshot_inline_char_limit,
    should_externalize_snapshot,
)
from chrys.app.tui.widgets.chat.renderers.file_edit import (
    _EDIT_FILE_INLINE_MAX_DISPLAY_LINES,
    _FILE_COPY_FULL_DIFF_MAX_CHARS,
    _WRITE_FILE_LINE_COUNT_SCAN_CHARS,
    EditFileToolCall,
    WriteFileToolCall,
)
from chrys.app.tui.widgets.chat.tool_call import BaseToolCard, ToolCardHeader
from chrys.app.tui.widgets.diff_view import DiffView
from chrys.app.tui.widgets.diff_view.annotations import AnnotationsColumn
from chrys.app.tui.widgets.diff_view.code import ELLIPSIS_SENTINEL, CodeColumn
from chrys.app.tui.widgets.diff_view.compute import (
    compute_grouped_opcodes,
    compute_highlighted_code_lines,
    flatten_unified_diff,
)
from chrys.app.tui.widgets.diff_view.inline import InlineUnifiedDiffLines
from chrys.app.tui.widgets.diff_view.styles import line_style_for_color_system
from chrys.app.tui.widgets.diff_view.unified import UnifiedDiffLines
from chrys.app.tui.widgets.file_preview import WRITE_FILE_PREVIEW_MAX_LINE_CHARS
from chrys.foundation.config.process_settings import reset_process_settings
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY
from chrys.service.mutations.store import SnapshotStore
from tests.support.paths import SRC_ROOT

_CHRYS_CSS = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"


def _content_bg_rgb(content: Content) -> tuple[int, int, int] | None:
    for _text, style, _control in content.render_segments():
        if style is None or style.bgcolor is None or style.bgcolor.triplet is None:
            continue
        triplet = style.bgcolor.triplet
        return (triplet.red, triplet.green, triplet.blue)
    return None


def _render_code_row(code_column: CodeColumn, row: int):
    code_column._invalidate_render_cache()
    code_column._frame_scroll_x = 0
    code_column._frame_scroll_y = 0
    code_column._frame_width = code_column._get_render_width()
    code_column._frame_selection = None
    return code_column.render_line(row)


async def _wait_for_inline_diff(pilot, *, attempts: int = 200) -> InlineUnifiedDiffLines:
    for _ in range(attempts):
        await pilot.pause()
        matches = list(pilot.app.query(InlineUnifiedDiffLines))
        if matches:
            await pilot.pause()
            return matches[0]
    raise AssertionError("InlineUnifiedDiffLines was not mounted")


class _WriteFileToolApp(App):
    def compose(self) -> ComposeResult:
        yield WriteFileToolCall("c1", "write_file", args={"path": "dist/index.html"})


class _EditFileToolApp(App):
    def compose(self) -> ComposeResult:
        yield EditFileToolCall("c1", "edit_file", args={"path": "src/app.py"})


class _GcEditFileToolApp(_EditFileToolApp):
    def __init__(self) -> None:
        self.gc_messages: list[GcAbsorbRequested] = []
        self.prewarmed_at_request = False
        super().__init__()

    def on_gc_absorb_requested(self, message: GcAbsorbRequested) -> None:
        diffs = list(self.query(InlineUnifiedDiffLines))
        self.prewarmed_at_request = bool(diffs and diffs[0]._strip_cache)
        self.gc_messages.append(message)


class _FileToolInstanceApp(App):
    def __init__(self, tool: EditFileToolCall | WriteFileToolCall) -> None:
        super().__init__()
        self._tool = tool

    def compose(self) -> ComposeResult:
        yield self._tool


class _ThemeSwitchDiffApp(App):
    CSS_PATH = str(_CHRYS_CSS)

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return dict(TUI_VARIABLE_DEFAULTS)

    def __init__(self, *, before: str = "same\n", after: str = "same\n", theme_name: str = "chrys") -> None:
        super().__init__()
        self._before = before
        self._after = after
        self.register_theme(CHRYS_THEME)
        self.register_theme(CHRYS_ANSI_THEME)
        self.theme = theme_name

    def _watch_theme(self, theme_name: str) -> None:
        super()._watch_theme(theme_name)
        is_ansi = theme_name in ANSI_THEMES
        self.set_class(theme_name in CHRYS_THEMES, "-chrys")
        self.set_class(is_ansi and theme_name in CHRYS_THEMES, "-chrys-ansi")

    def compose(self) -> ComposeResult:
        diff = DiffView("same.py", "same.py", self._before, self._after)
        diff.split = False
        yield diff


class _LargeOneLineHtml(str):
    """Sentinel for minified/generated HTML where full line splitting is unsafe."""

    def splitlines(self, keepends: bool = False) -> list[str]:
        raise AssertionError("write_file renderer split the full HTML snapshot")

    def count(self, sub: str, start: int | None = None, end: int | None = None) -> int:
        if sub == "\n" and (start is None or end is None or end > _WRITE_FILE_LINE_COUNT_SCAN_CHARS):
            raise AssertionError("write_file renderer counted the full HTML snapshot")
        return super().count(sub, 0 if start is None else start, len(self) if end is None else end)


def test_diff_view_changed_gutters_survive_256_color_downgrade() -> None:
    expected_gutter_256_color = {"+": 22, "-": 52}
    for annotation in ("+", "-"):
        gutter_bg = Color.parse(DiffView.NUMBER_STYLES[annotation].rsplit(" on ", 1)[1])
        assert gutter_bg.downgrade(ColorSystem.EIGHT_BIT).number == expected_gutter_256_color[annotation]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 128 * 1024), ("4096", 4096), ("-1", 0), ("not-an-int", 128 * 1024)],
)
def test_snapshot_inline_limit_parse_is_fail_soft(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
    expected: int,
) -> None:
    """Same fail-soft grammar as the module-level parser this replaced."""
    if raw is None:
        monkeypatch.delenv("CHRYS_TUI_FILE_SNAPSHOT_INLINE_CHARS", raising=False)
    else:
        monkeypatch.setenv("CHRYS_TUI_FILE_SNAPSHOT_INLINE_CHARS", raw)
    reset_process_settings()

    assert file_snapshot_inline_char_limit() == expected


def test_live_snapshot_externalization_uses_utf8_byte_size() -> None:
    text = "你" * (file_snapshot_inline_char_limit() // len("你".encode()) + 1)

    assert len(text) < file_snapshot_inline_char_limit()
    assert len(text.encode("utf-8")) > file_snapshot_inline_char_limit()
    assert should_externalize_snapshot(("", text)) is True


def test_diff_view_tcss_theme_variables_do_not_render_as_syntax_errors() -> None:
    diff = DiffView(
        "theme.tcss",
        "theme.tcss",
        "",
        "border: round $primary;\nborder-subtitle-color: $text-muted;\n",
    )

    _lines_before, lines_after = diff.highlighted_code_lines
    assert "$primary" in lines_after[0].plain
    assert "$text-muted" in lines_after[1].plain
    assert all("error-muted" not in str(span.style) for line in lines_after for span in line.spans)


def test_diff_highlighting_uses_path_language_without_content_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known file paths should avoid expensive content-based lexer guessing."""
    from chrys.app.tui.widgets.diff_view import compute as compute_module

    def fail_guess_language(_code: str, _path: str | None) -> str:
        raise AssertionError("diff highlighting should not call content language guessing for .py paths")

    monkeypatch.setattr(compute_module.highlight, "guess_language", fail_guess_language)

    before = "def old() -> int:\n    return 1\n"
    after = "def new() -> int:\n    return 2\n"
    lines_before, lines_after = compute_highlighted_code_lines(
        before,
        after,
        "src/example.py",
        "src/example.py",
        compute_grouped_opcodes(before, after),
    )

    assert [line.plain for line in lines_before] == before.splitlines()
    assert [line.plain for line in lines_after] == after.splitlines()


def test_diff_highlighting_fast_path_includes_csharp(monkeypatch: pytest.MonkeyPatch) -> None:
    """C# files should use the path fast path instead of content guessing."""
    from chrys.app.tui.widgets.diff_view import compute as compute_module

    def fail_guess_language(_code: str, _path: str | None) -> str:
        raise AssertionError("diff highlighting should not call content language guessing for .cs paths")

    monkeypatch.setattr(compute_module.highlight, "guess_language", fail_guess_language)

    before = "public sealed class Example { }\n"
    after = "public sealed class Example { public int Value => 1; }\n"
    _lines_before, lines_after = compute_highlighted_code_lines(
        before,
        after,
        "src/Example.cs",
        "src/Example.cs",
        compute_grouped_opcodes(before, after),
    )

    assert lines_after[0].plain == after.rstrip("\n")


def test_diff_highlighting_unknown_same_path_guesses_after_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown same-path diffs should not reuse an empty before-side content guess."""
    from chrys.app.tui.widgets.diff_view import compute as compute_module

    guess_calls: list[tuple[str, str | None]] = []

    def fake_guess_language(code: str, path: str | None) -> str:
        guess_calls.append((code, path))
        return "text" if not code else "python"

    monkeypatch.setattr(compute_module.highlight, "guess_language", fake_guess_language)

    before = ""
    after = "def hello() -> int:\n    return 1\n"
    lines_before, lines_after = compute_highlighted_code_lines(
        before,
        after,
        "notes.unknown",
        "notes.unknown",
        compute_grouped_opcodes(before, after),
    )

    assert lines_before == []
    assert [line.plain for line in lines_after] == after.splitlines()
    assert guess_calls == [(before, "notes.unknown"), (after, "notes.unknown")]


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "args"),
    [
        (WriteFileToolCall, "write_file", {"path": "dist/index.html"}),
        (EditFileToolCall, "edit_file", {"path": "src/app.py"}),
    ],
)
def test_file_tool_renderers_use_base_tool_card_contract(
    tool_cls: type[EditFileToolCall] | type[WriteFileToolCall],
    tool_name: str,
    args: dict[str, str],
) -> None:
    tool = tool_cls("c1", tool_name, args=args)

    assert isinstance(tool, BaseToolCard)
    assert tool.call_id == "c1"
    assert tool.tool_name == tool_name
    assert tool.status == "running"
    assert tool.result_text == ""
    assert tool.duration_ms == 0
    assert tool.args == args


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("src/main.cxx", "cpp"),
        ("include/widget.hh", "cpp"),
        ("src/App.mm", "objective-c++"),
        ("src/Script.csx", "csharp"),
        ("src/App.csproj", "xml"),
        ("src/Library.fs", "fsharp"),
        ("src/Library.fsi", "fsharp"),
        ("src/Program.fsx", "fsharp"),
        ("src/App.vb", "vb.net"),
        ("src/Main.scala", "scala"),
        ("src/app.dart", "dart"),
        ("src/pages/Index.ets", "typescript"),
        ("src/pages/Index.arkts", "typescript"),
        ("src/index.mjs", "javascript"),
        ("src/App.vue", "vue"),
        ("src/App.svelte", "html"),
        ("tsconfig.jsonc", "javascript"),
        ("src/config.jsonl", "json"),
        ("setup.cfg", "ini"),
        (".env", "text"),
        (".gitignore", "text"),
        ("src/icon.svg", "xml"),
        ("src/App.xaml", "xml"),
        ("Directory.Build.props", "xml"),
        ("src/script.ps1", "powershell"),
        ("scripts/login.ksh", "bash"),
        ("scripts/build.bat", "batch"),
        ("shader.frag", "glsl"),
        ("shader.vert", "glsl"),
        ("schema.gql", "graphql"),
        ("README.mdx", "markdown"),
        ("CMakeLists.txt", "cmake"),
        ("go.mod", "text"),
        ("Pipfile", "toml"),
        ("Gemfile", "ruby"),
        ("Dockerfile.dev", "docker"),
        ("Makefile.am", "make"),
    ],
)
def test_diff_language_fast_path_covers_common_project_files(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    language: str,
) -> None:
    """Common project file names should avoid content-based language guessing."""
    from chrys.app.tui.widgets.diff_view import compute as compute_module

    def fail_guess_language(_code: str, _path: str | None) -> str:
        raise AssertionError(f"diff highlighting should not content-guess {path}")

    monkeypatch.setattr(compute_module.highlight, "guess_language", fail_guess_language)

    assert compute_module._guess_diff_language("placeholder\n", path) == language


def test_grouped_diff_highlighting_preserves_multiline_lexer_state() -> None:
    """Sparse highlighting should keep lexer state for hunks inside multi-line strings."""
    before = 'x = """\n' + "\n".join(f"line {index}" for index in range(80)) + '\n"""\n'
    after = 'x = """\n' + "\n".join("changed" if index == 40 else f"line {index}" for index in range(80)) + '\n"""\n'
    opcodes = compute_grouped_opcodes(before, after)
    _tag, _i1, _i2, changed_start, _changed_end = next(
        opcode for group in opcodes for opcode in group if opcode[0] == "replace"
    )

    _full_before, full_after = compute_highlighted_code_lines(
        before,
        after,
        "src/example.py",
        "src/example.py",
        opcodes,
        grouped_only=False,
    )
    _grouped_before, grouped_after = compute_highlighted_code_lines(
        before,
        after,
        "src/example.py",
        "src/example.py",
        opcodes,
        grouped_only=True,
    )

    def syntax_spans(content: Content) -> list[tuple[int, int, str]]:
        return [(span.start, span.end, str(span.style)) for span in content.spans if " on " not in str(span.style)]

    assert syntax_spans(grouped_after[changed_start]) == syntax_spans(full_after[changed_start])


def test_grouped_diff_highlighting_preserves_trailing_context_blank_line() -> None:
    """Sparse highlighting must not shrink ranges that end on interior blank lines."""
    before = "a = 1\nb = 2\nc = 3\n\nd = 4\ne = 5\nf = 6\ng = 7\n"
    after = "a = 10\nb = 2\nc = 3\n\nd = 4\ne = 5\nf = 6\ng = 7\n"
    opcodes = compute_grouped_opcodes(before, after)
    lines_before, lines_after = compute_highlighted_code_lines(
        before,
        after,
        "src/example.py",
        "src/example.py",
        opcodes,
        grouped_only=True,
    )
    flat = flatten_unified_diff(
        opcodes,
        lines_before,
        lines_after,
        {" ": "", "+": "", "-": ""},
    )

    assert [line.plain for line in lines_before] == before.splitlines()
    assert [line.plain for line in lines_after] == after.splitlines()
    assert [
        line.plain for annotation, line in zip(flat.annotations, flat.code_lines, strict=True) if annotation == " "
    ] == ["b = 2", "c = 3", ""]


@pytest.mark.asyncio
async def test_inline_diff_prepare_highlights_only_grouped_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inline previews should not syntax-highlight unchanged offscreen file regions."""
    from chrys.app.tui.widgets.diff_view import compute as compute_module

    original_highlight_code_lines = compute_module._highlight_code_lines
    highlighted_line_counts: list[int] = []

    def highlight_spy(code: str, path: str, language: str) -> list[Content]:
        highlighted_line_counts.append(len(code.splitlines()))
        return original_highlight_code_lines(code, path, language)

    monkeypatch.setattr(compute_module, "_highlight_code_lines", highlight_spy)

    before_lines = [f"VALUE_{index} = {index}" for index in range(300)]
    after_lines = before_lines.copy()
    after_lines[50] = "VALUE_50 = 5000"
    diff = InlineUnifiedDiffLines(
        "src/example.py",
        "src/example.py",
        "\n".join(before_lines) + "\n",
        "\n".join(after_lines) + "\n",
        auto_height=True,
    )

    await diff.prepare()

    assert highlighted_line_counts
    assert sum(highlighted_line_counts) < len(before_lines)


@pytest.mark.asyncio
async def test_diff_view_uses_explicit_full_line_backgrounds_in_256_color_mode() -> None:
    async with _ThemeSwitchDiffApp(before="", after="added\n").run_test() as pilot:
        await pilot.pause()
        pilot.app.console._color_system = ColorSystem.EIGHT_BIT
        code_column = pilot.app.query_one(CodeColumn)

        assert line_style_for_color_system(DiffView.DARK_LINE_STYLES["+"], "256") == "on #005F00"
        assert line_style_for_color_system(DiffView.DARK_LINE_STYLES["-"], "256") == "on #5F0000"

        diff = pilot.app.query_one(DiffView)
        assert _content_bg_rgb(diff._make_annotation_factory_unified(["+"])(0)) == (0, 95, 0)
        assert _content_bg_rgb(diff._make_annotation_factory_unified(["-"])(0)) == (95, 0, 0)
        assert _content_bg_rgb(diff._make_annotation_factory_split(["+"], "+")(0)) == (0, 95, 0)
        assert _content_bg_rgb(diff._make_annotation_factory_split(["-"], "-")(0)) == (95, 0, 0)

        strip = _render_code_row(code_column, 0)
        line_bg = Color.parse("#005F00").triplet
        assert line_bg is not None
        has_256_line_bg = False
        for _text, style, _control in strip._segments:
            if style is None or style.bgcolor is None:
                continue
            bgcolor = style.bgcolor.triplet
            has_256_line_bg = has_256_line_bg or (
                bgcolor is not None
                and (bgcolor.red, bgcolor.green, bgcolor.blue)
                == (
                    line_bg.red,
                    line_bg.green,
                    line_bg.blue,
                )
            )
        assert has_256_line_bg is True


@pytest.mark.asyncio
async def test_diff_view_preserves_intraline_highlights_in_256_color_mode() -> None:
    async with _ThemeSwitchDiffApp(before="spam\n", after="span\n").run_test() as pilot:
        await pilot.pause()
        pilot.app.console._color_system = ColorSystem.EIGHT_BIT
        code_column = pilot.app.query_one(CodeColumn)

        strip = _render_code_row(code_column, 0)
        row_bg = Color.parse("#5F0000").triplet
        assert row_bg is not None
        intraline_bg = None
        for text, style, _control in strip._segments:
            if text != "m" or style is None or style.bgcolor is None:
                continue
            intraline_bg = style.bgcolor.triplet
            break

        assert intraline_bg is not None
        assert (intraline_bg.red, intraline_bg.green, intraline_bg.blue) != (
            row_bg.red,
            row_bg.green,
            row_bg.blue,
        )


@pytest.mark.asyncio
async def test_diff_view_uses_light_palette_for_light_themes() -> None:
    async with _ThemeSwitchDiffApp(before="", after="added\n", theme_name="ansi-light").run_test() as pilot:
        await pilot.pause()
        code_column = pilot.app.query_one(CodeColumn)
        assert code_column.line_styles[0] == DiffView.LIGHT_LINE_STYLES["+"]


@pytest.mark.asyncio
async def test_diff_view_recomposes_when_theme_brightness_switches() -> None:
    async with _ThemeSwitchDiffApp(before="", after="added\n").run_test() as pilot:
        await pilot.pause()
        code_column = pilot.app.query_one(CodeColumn)
        assert code_column.line_styles[0] == DiffView.DARK_LINE_STYLES["+"]

        pilot.app.theme = "ansi-light"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while pilot.app.query_one(CodeColumn).line_styles[0] != DiffView.LIGHT_LINE_STYLES["+"]:
            if loop.time() > deadline:
                raise AssertionError("diff view did not switch to the light palette")
            await pilot.pause(0.05)

        code_column = pilot.app.query_one(CodeColumn)
        assert code_column.line_styles[0] == DiffView.LIGHT_LINE_STYLES["+"]


@pytest.mark.asyncio
async def test_inline_unified_diff_refreshes_palette_when_theme_brightness_switches() -> None:
    async with App().run_test(size=(80, 12)) as pilot:
        diff = InlineUnifiedDiffLines("added.py", "added.py", "", "added\n", auto_height=True)
        await diff.prepare()
        await pilot.app.mount(diff)
        await pilot.pause()

        assert diff._flat_styles[0] == InlineUnifiedDiffLines.DARK_LINE_STYLES["+"]

        pilot.app.theme = "ansi-light"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while diff._flat_styles[0] != InlineUnifiedDiffLines.LIGHT_LINE_STYLES["+"]:
            if loop.time() > deadline:
                raise AssertionError("inline diff did not switch to the light palette")
            await pilot.pause(0.05)
        diff.render_lines(Region(0, 0, diff.size.width, 1))

        assert diff._flat_styles[0] == InlineUnifiedDiffLines.LIGHT_LINE_STYLES["+"]


@pytest.mark.asyncio
async def test_inline_unified_diff_render_lines_after_remove_returns_blank() -> None:
    async with App().run_test(size=(80, 12)) as pilot:
        diff = InlineUnifiedDiffLines("added.py", "added.py", "", "added\n", auto_height=True)
        await diff.prepare()
        await pilot.app.mount(diff)
        await pilot.pause()

        await diff.remove()
        await pilot.pause()

        rendered = diff.render_lines(Region(0, 0, 20, 2))

    assert [strip.cell_length for strip in rendered] == [20, 20]


@pytest.mark.asyncio
async def test_code_column_render_lines_after_remove_returns_blank() -> None:
    async with _ThemeSwitchDiffApp(before="", after="added\n").run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        code_column = pilot.app.query_one(CodeColumn)

        await code_column.remove()
        await pilot.pause()

        rendered = code_column.render_lines(Region(0, 0, 20, 2))

    assert [strip.cell_length for strip in rendered] == [20, 20]


def test_annotation_column_render_lines_unattached_returns_blank() -> None:
    code_column = CodeColumn([Content("added")], ["+"], 5)
    annotation = AnnotationsColumn(1, 3, lambda _index: Content(" + "), code_column)

    rendered = annotation.render_lines(Region(0, 0, 20, 2))

    assert [strip.cell_length for strip in rendered] == [20, 20]


@pytest.mark.asyncio
async def test_diff_view_invalidates_render_cache_when_theme_switches() -> None:
    async with _ThemeSwitchDiffApp().run_test() as pilot:
        await pilot.pause()
        code_column = pilot.app.query_one(CodeColumn)
        crop = Region(0, 0, code_column.size.width, 1)

        old_lines = code_column.render_lines(crop)
        assert code_column._lines_cache_result is old_lines

        pilot.app.theme = "chrys-ansi"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while code_column._lines_cache_result is old_lines:
            if loop.time() > deadline:
                raise AssertionError("diff view render cache was not invalidated")
            await pilot.pause(0.05)

        assert code_column._lines_cache_result is not old_lines


@pytest.mark.asyncio
async def test_diff_view_annotation_cache_tracks_color_system() -> None:
    async with _ThemeSwitchDiffApp(before="", after="added\n").run_test() as pilot:
        await pilot.pause()
        annotation = list(pilot.app.query(AnnotationsColumn))[-1]

        # Force TRUECOLOR explicitly so the assertion is deterministic across
        # platforms (Windows CI defaults to a downsampled color system).
        pilot.app.console._color_system = ColorSystem.TRUECOLOR
        annotation.render_lines(Region(0, 0, annotation.size.width, 1))
        assert _content_bg_rgb(annotation._get_entry(0)) == (36, 63, 48)

        pilot.app.console._color_system = ColorSystem.EIGHT_BIT
        annotation.render_lines(Region(0, 0, annotation.size.width, 1))

        assert _content_bg_rgb(annotation._get_entry(0)) == (0, 95, 0)


@pytest.mark.asyncio
async def test_unified_diff_without_scrollbars_uses_flat_renderer() -> None:
    class _InlineUnifiedDiffApp(App):
        def compose(self) -> ComposeResult:
            diff = DiffView("added.py", "added.py", "", "added\n")
            diff.split = False
            diff.auto_height = True
            diff.show_scrollbars = False
            yield diff

    async with _InlineUnifiedDiffApp().run_test() as pilot:
        await pilot.pause()
        flat = pilot.app.query_one(UnifiedDiffLines)

        assert not list(pilot.app.query(CodeColumn))
        rendered = flat.render_lines(Region(0, 0, flat.size.width, 1))
        assert "+ added" in "".join(segment.text for segment in rendered[0]._segments)
        assert flat.ALLOW_SELECT
        selected, _separator = flat.get_selection(Selection(Offset(9, 0), Offset(14, 0)))
        assert selected == "added"
        full_row_selected, _separator = flat.get_selection(Selection(Offset(0, 0), Offset(flat.size.width, 0)))
        assert full_row_selected.rstrip() == "added"

        pilot.app.console._color_system = ColorSystem.EIGHT_BIT
        flat.render_lines(Region(0, 0, flat.size.width, 1))
        rendered = flat.render_line(0)
        line_bg = Color.parse("#005F00").triplet
        assert line_bg is not None
        assert any(
            style is not None
            and style.bgcolor is not None
            and style.bgcolor.triplet is not None
            and (style.bgcolor.triplet.red, style.bgcolor.triplet.green, style.bgcolor.triplet.blue)
            == (line_bg.red, line_bg.green, line_bg.blue)
            for _text, style, _control in rendered._segments
        )


@pytest.mark.asyncio
async def test_code_column_render_line_picks_up_color_system_after_theme_change() -> None:
    """When ``_color_system`` flips, ``render_line`` must re-resolve line styles.

    Regression test for the Tier 3 hoist: ``CodeColumn`` caches a per-frame
    ``_frame_color_system`` so the hot path doesn't traverse ``self.app.console``
    per row.  If the cache is not invalidated on theme / color-system change,
    the downgrade map silently mis-applies.  Tests using the public
    ``render_lines`` API exercise this end-to-end.
    """
    async with _ThemeSwitchDiffApp(before="", after="added\n").run_test() as pilot:
        await pilot.pause()
        code_column = pilot.app.query_one(CodeColumn)

        # First frame in TRUECOLOR: no downgrade applied.
        pilot.app.console._color_system = ColorSystem.TRUECOLOR
        code_column._invalidate_render_cache()
        code_column.render_lines(Region(0, 0, code_column.size.width, 1))
        assert code_column._frame_color_system == "truecolor"

        # Theme swap to a 256-color terminal: downgrade map must engage.
        pilot.app.console._color_system = ColorSystem.EIGHT_BIT
        code_column._invalidate_render_cache()
        code_column.render_lines(Region(0, 0, code_column.size.width, 1))
        assert code_column._frame_color_system == "256"

        strip = _render_code_row(code_column, 0)
        downgraded = Color.parse("#005F00").triplet
        assert downgraded is not None
        assert any(
            style is not None and style.bgcolor is not None and style.bgcolor.triplet == downgraded
            for _text, style, _control in strip._segments
        )


@pytest.mark.asyncio
async def test_flat_unified_diff_annotations_toggle_updates_after_mount() -> None:
    class _InlineUnifiedDiffApp(App):
        def compose(self) -> ComposeResult:
            diff = DiffView("added.py", "added.py", "", "added\n")
            diff.split = False
            diff.auto_height = True
            diff.show_scrollbars = False
            yield diff

    async with _InlineUnifiedDiffApp().run_test() as pilot:
        await pilot.pause()
        diff = pilot.app.query_one(DiffView)
        flat = pilot.app.query_one(UnifiedDiffLines)
        assert flat._annotation_width == 3

        rendered = flat.render_lines(Region(0, 0, flat.size.width, 1))
        plain = "".join(segment.text for segment in rendered[0]._segments)

        assert "+ added" in plain

        diff.annotations = False
        await pilot.pause()
        assert flat._annotation_width == 1
        rendered = flat.render_lines(Region(0, 0, flat.size.width, 1))
        plain = "".join(segment.text for segment in rendered[0]._segments)

        assert "added" in plain
        assert "+ added" not in plain

        diff.annotations = True
        await pilot.pause()
        assert flat._annotation_width == 3
        rendered = flat.render_lines(Region(0, 0, flat.size.width, 1))
        plain = "".join(segment.text for segment in rendered[0]._segments)

        assert "+ added" in plain


@pytest.mark.asyncio
async def test_flat_unified_diff_multi_row_selection_copies_code_only() -> None:
    class _InlineUnifiedDiffApp(App):
        def compose(self) -> ComposeResult:
            diff = DiffView("added.py", "added.py", "", "first\nsecond\n")
            diff.split = False
            diff.auto_height = True
            diff.show_scrollbars = False
            yield diff

    async with _InlineUnifiedDiffApp().run_test() as pilot:
        await pilot.pause()
        flat = pilot.app.query_one(UnifiedDiffLines)

        selected, _separator = flat.get_selection(Selection(Offset(0, 0), Offset(flat.size.width, 1)))

        assert selected == "first\nsecond"


@pytest.mark.asyncio
async def test_flat_unified_diff_selection_preserves_blank_context_break_row() -> None:
    before = "\n".join(f"line {index}" for index in range(30)) + "\n"
    after_lines = [f"line {index}" for index in range(30)]
    after_lines[0] = "changed 0"
    after_lines[29] = "changed 29"
    after = "\n".join(after_lines) + "\n"

    class _InlineUnifiedDiffApp(App):
        def compose(self) -> ComposeResult:
            diff = DiffView("changed.py", "changed.py", before, after)
            diff.split = False
            diff.auto_height = True
            diff.show_scrollbars = False
            yield diff

    async with _InlineUnifiedDiffApp().run_test() as pilot:
        await pilot.pause()
        flat = pilot.app.query_one(UnifiedDiffLines)
        ellipsis_index = flat._code_lines.index(ELLIPSIS_SENTINEL)

        selected, _separator = flat.get_selection(
            Selection(Offset(0, ellipsis_index - 1), Offset(flat.size.width, ellipsis_index + 1))
        )

        assert selected == "line 3\n\nline 26"


@pytest.mark.asyncio
async def test_write_file_large_one_line_html_result_does_not_scan_full_snapshot() -> None:
    """Reproduce the frozen `Running: write_file` path with minified HTML.

    The live ToolCallResult handler cannot reset the status bar until
    WriteFileToolCall.set_complete() returns.  For generated HTML, line-count
    and preview code must be bounded; calling splitlines() on the full snapshot
    can stall or corrupt the TUI without an exception.
    """
    html = _LargeOneLineHtml("<!doctype html><html><body>" + ("x" * 1_000_000) + "</body></html>")

    async with _WriteFileToolApp().run_test() as pilot:
        tool = pilot.app.query_one(WriteFileToolCall)
        header = tool.query_one(ToolCardHeader)
        assert header.actions_visible is False

        tool.set_complete(
            "Written 1000039 chars (1 lines) to /repo/dist/index.html.",
            25,
            file_snapshot=("", html),
        )
        assert header.actions_visible is True
        diff = await _wait_for_inline_diff(pilot)

        assert tool.status == "complete"
        assert diff.code_after.endswith("...")
        assert len(diff.code_after) <= WRITE_FILE_PREVIEW_MAX_LINE_CHARS + len("...")


@pytest.mark.asyncio
async def test_write_file_line_count_fallback_is_bounded() -> None:
    html = _LargeOneLineHtml("<!doctype html><html><body>" + ("x" * 1_000_000) + "</body></html>")

    async with _WriteFileToolApp().run_test() as pilot:
        tool = pilot.app.query_one(WriteFileToolCall)

        tool.set_complete(
            "Successfully wrote dist/index.html.",
            25,
            file_snapshot=("", html),
        )
        await pilot.pause()

        panel = tool.query_one("#ft-panel")
        assert panel.border_subtitle == "1+ lines written"


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "args"),
    [
        (WriteFileToolCall, "write_file", {"path": "dist/index.html"}),
        (EditFileToolCall, "edit_file", {"path": "src/app.py"}),
    ],
)
@pytest.mark.asyncio
async def test_file_tool_renderers_record_rejected_completion_approval(
    tool_cls: type[EditFileToolCall] | type[WriteFileToolCall],
    tool_name: str,
    args: dict[str, str],
) -> None:
    tool = tool_cls("c1", tool_name, args=args)

    async with _FileToolInstanceApp(tool).run_test() as pilot:
        mounted = pilot.app.query_one(tool_cls)

        mounted.set_complete("Error: rejected", duration_ms=10, approval="user_rejected")
        await pilot.pause()

        assert mounted.approval == "user_rejected"
        assert mounted.has_class("-rejected")
        assert mounted.query_one("#ft-panel").border_subtitle == "Rejected"
        assert mounted.query_one("#ft-content").query_one(Static).render().plain == "Error: rejected"


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "args"),
    [
        (WriteFileToolCall, "write_file", {"path": "dist/index.html"}),
        (EditFileToolCall, "edit_file", {"path": "src/app.py"}),
    ],
)
@pytest.mark.asyncio
async def test_file_tool_renderers_use_structured_failure_metadata(
    tool_cls: type[EditFileToolCall] | type[WriteFileToolCall],
    tool_name: str,
    args: dict[str, str],
) -> None:
    tool = tool_cls("c1", tool_name, args=args)

    async with _FileToolInstanceApp(tool).run_test() as pilot:
        mounted = pilot.app.query_one(tool_cls)

        mounted.set_complete(
            "Error: literal file contents",
            duration_ms=10,
            metadata={TOOL_FAILED_METADATA_KEY: False},
            file_snapshot=("before\n", "after\n"),
        )
        await pilot.pause()

        assert mounted.has_class("-success")
        assert not mounted.has_class("-error")


@pytest.mark.parametrize(
    ("tool_cls", "tool_name", "args"),
    [
        (WriteFileToolCall, "write_file", {"path": "dist/index.html"}),
        (EditFileToolCall, "edit_file", {"path": "src/app.py"}),
    ],
)
@pytest.mark.asyncio
async def test_file_tool_renderers_set_error_status_for_failed_results(
    tool_cls: type[EditFileToolCall] | type[WriteFileToolCall],
    tool_name: str,
    args: dict[str, str],
) -> None:
    tool = tool_cls("c1", tool_name, args=args)

    async with _FileToolInstanceApp(tool).run_test() as pilot:
        mounted = pilot.app.query_one(tool_cls)

        mounted.set_complete("Error: failed", duration_ms=10, metadata={TOOL_FAILED_METADATA_KEY: True})
        await pilot.pause()

        assert mounted.status == "error"
        assert mounted.has_class("-error")


@pytest.mark.asyncio
async def test_disk_backed_live_snapshot_defers_inline_diff_until_explicit_mount(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    before_hash = store.save_data_as_blob(b"old\n").content_hash
    after_hash = store.save_data_as_blob(b"new\n").content_hash
    ref = FileSnapshotRef(store.mutations_dir, before_hash, after_hash)

    async with _EditFileToolApp().run_test() as pilot:
        tool = pilot.app.query_one(EditFileToolCall)

        tool.set_complete("Edited src/app.py", 25, file_snapshot=ref)
        await pilot.pause(0.35)

        assert tool._after_content is None
        assert tool._diff_pending is True
        assert not list(pilot.app.query(DiffView))
        assert not list(pilot.app.query(InlineUnifiedDiffLines))

        await tool.mount_diff_if_pending()
        await pilot.pause()

        assert tool._get_after_content() == "new\n"
        assert not list(pilot.app.query(DiffView))
        assert list(pilot.app.query(InlineUnifiedDiffLines))


@pytest.mark.asyncio
async def test_deferred_live_diff_posts_absorb_after_visible_rows_are_prewarmed() -> None:
    app = _GcEditFileToolApp()
    async with app.run_test(size=(80, 12)) as pilot:
        tool = app.query_one(EditFileToolCall)
        tool.set_complete(
            "Edited src/app.py",
            25,
            file_snapshot=("old\n", "new\n"),
        )

        await _wait_for_inline_diff(pilot)
        for _ in range(5):
            await pilot.pause()
            if app.gc_messages:
                break

        assert len(app.gc_messages) == 1
        assert app.gc_messages[0].reason is GcAbsorbReason.STABLE_CONTENT_MOUNTED
        assert app.gc_messages[0].terminal_boundary is False
        assert app.prewarmed_at_request is True


@pytest.mark.asyncio
async def test_write_file_preview_truncation_keeps_raw_snapshot_for_copy() -> None:
    raw_line = "<!doctype html><html><body>" + ("x" * (WRITE_FILE_PREVIEW_MAX_LINE_CHARS + 20))
    raw_html = raw_line + "</body></html>"

    async with _WriteFileToolApp().run_test() as pilot:
        tool = pilot.app.query_one(WriteFileToolCall)

        tool.set_complete(
            f"Written {len(raw_html)} chars (1 lines) to /repo/dist/index.html.",
            25,
            file_snapshot=("", raw_html),
        )

        diff = await _wait_for_inline_diff(pilot)
        assert raw_html not in diff.code_after
        assert diff.code_after.endswith("...")
        assert tool._get_after_content() == raw_html
        copy_sections = tool._tool_copy_sections()
        assert any(raw_html in text for _, _, text in copy_sections)


@pytest.mark.asyncio
async def test_edit_file_inline_diff_is_height_capped() -> None:
    before = "\n".join(f"old {index}" for index in range(80)) + "\n"
    after = "\n".join(f"new {index}" for index in range(80)) + "\n"

    async with _EditFileToolApp().run_test() as pilot:
        tool = pilot.app.query_one(EditFileToolCall)

        tool.set_complete("Edited src/app.py", 25, file_snapshot=(before, after))

        diff = await _wait_for_inline_diff(pilot)
        assert diff.max_display_lines == _EDIT_FILE_INLINE_MAX_DISPLAY_LINES

        flat = pilot.app.query_one(UnifiedDiffLines)
        assert flat.styles.height.value == _EDIT_FILE_INLINE_MAX_DISPLAY_LINES


@pytest.mark.asyncio
async def test_file_tool_mounts_text_fallback_when_inline_diff_prepare_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_prepare(self: InlineUnifiedDiffLines) -> None:
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(InlineUnifiedDiffLines, "prepare", fail_prepare)

    async with _WriteFileToolApp().run_test() as pilot:
        tool = pilot.app.query_one(WriteFileToolCall)
        result = "Written 4 chars (1 lines) to /repo/dist/index.html."

        tool.set_complete(result, 25, file_snapshot=("", "new\n"))

        for _ in range(200):
            await pilot.pause()
            if tool.query_one("#ft-content").children:
                break

        assert not list(pilot.app.query(InlineUnifiedDiffLines))
        fallback = tool.query_one("#ft-content").query_one(Static)
        assert fallback.render().plain == result


@pytest.mark.asyncio
async def test_write_file_large_snapshot_copy_diff_is_bounded() -> None:
    html = _LargeOneLineHtml("<!doctype html><html><body>" + ("x" * 1_000_000) + "</body></html>")

    async with _WriteFileToolApp().run_test() as pilot:
        tool = pilot.app.query_one(WriteFileToolCall)

        tool.set_complete(
            "Written 1000039 chars (1 lines) to /repo/dist/index.html.",
            25,
            file_snapshot=("", html),
        )

        copy_sections = tool._tool_copy_sections()
        preview = next(text for title, _, text in copy_sections if title == "Diff Preview")
        assert "Full diff omitted" in preview
        assert len(preview) < _FILE_COPY_FULL_DIFF_MAX_CHARS
        assert "x" * 1000 not in preview


def test_large_edit_file_copy_preview_uses_real_unified_diff() -> None:
    before_lines = [f"line {i}" for i in range(80)]
    after_lines = before_lines.copy()
    after_lines[50] = "line fifty changed"
    before = "\n".join(before_lines) + "\n" + ("tail" * 20_000)
    after = "\n".join(after_lines) + "\n" + ("tail" * 20_000)
    tool = EditFileToolCall("c1", "edit_file", args={"path": "src/app.py"})
    tool.result_text = "Edited src/app.py"
    tool._before_content = before
    tool._after_content = after

    copy_sections = tool._tool_copy_sections()
    preview = next(text for title, _, text in copy_sections if title == "Diff Preview")

    assert "showing bounded diff around first change" in preview
    assert "-line 50" in preview
    assert "+line fifty changed" in preview
    assert "-line 0" not in preview
    assert "+line 0" not in preview
    assert "Bounded preview contains no changed lines" not in preview


def test_large_edit_file_copy_preview_keeps_dash_plus_prefixed_changes() -> None:
    before_lines = [f"line {i}" for i in range(80)]
    after_lines = before_lines.copy()
    before_lines[50] = "-- removed flag"
    after_lines[50] = "++ added flag"
    before = "\n".join(before_lines) + "\n" + ("tail" * 20_000)
    after = "\n".join(after_lines) + "\n" + ("tail" * 20_000)
    tool = EditFileToolCall("c1", "edit_file", args={"path": "src/app.py"})
    tool.result_text = "Edited src/app.py"
    tool._before_content = before
    tool._after_content = after

    copy_sections = tool._tool_copy_sections()
    preview = next(text for title, _, text in copy_sections if title == "Diff Preview")

    assert "--- removed flag" in preview
    assert "+++ added flag" in preview


def test_file_tool_copy_full_diff_includes_final_newline_change() -> None:
    tool = EditFileToolCall("c1", "edit_file", args={"path": "src/app.py"})
    tool.result_text = "Edited src/app.py"
    tool._before_content = "line"
    tool._after_content = "line\n"

    copy_sections = tool._tool_copy_sections()
    diff = next(text for title, _, text in copy_sections if title == "Diff")

    assert "-line [no newline]" in diff
    assert "+line [EOL: LF]" in diff


def test_file_tool_copy_full_diff_includes_line_ending_change() -> None:
    tool = EditFileToolCall("c1", "edit_file", args={"path": "src/app.py"})
    tool.result_text = "Edited src/app.py"
    tool._before_content = "line\r\n"
    tool._after_content = "line\n"

    copy_sections = tool._tool_copy_sections()
    diff = next(text for title, _, text in copy_sections if title == "Diff")

    assert "-line [EOL: CRLF]" in diff
    assert "+line [EOL: LF]" in diff


def test_large_file_copy_preview_includes_line_ending_change() -> None:
    before = ("line\r\n" * 14_000) + "tail\r\n"
    after = ("line\n" * 14_000) + "tail\n"
    tool = EditFileToolCall("c1", "edit_file", args={"path": "src/app.py"})
    tool.result_text = "Edited src/app.py"
    tool._before_content = before
    tool._after_content = after

    copy_sections = tool._tool_copy_sections()
    diff = next(text for title, _, text in copy_sections if title == "Diff Preview")

    assert "-line [EOL: CRLF]" in diff
    assert "+line [EOL: LF]" in diff
