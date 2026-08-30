# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the image compression progress dialog."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.dialogs.image_compression import ImageCompressionDialog
from chrys.foundation.config.settings import Settings


def test_image_compression_default_title_keeps_english_and_localizes_chinese() -> None:
    assert ImageCompressionDialog()._title == "Preparing Image"
    controller = LocaleController(Settings(locale="zh-Hans"))
    assert ImageCompressionDialog(locale_controller=controller)._title == "正在准备图像"


@pytest.mark.asyncio
async def test_image_compression_finish_dismisses_once() -> None:
    class _Harness(App):
        def __init__(self) -> None:
            super().__init__()
            self.dialog = ImageCompressionDialog()
            self.results: list[None] = []

        def compose(self) -> ComposeResult:
            yield Static("base")

        def on_mount(self) -> None:
            self.push_screen(self.dialog, self.results.append)

    async with _Harness().run_test() as pilot:
        await pilot.pause()

        pilot.app.dialog.finish()
        pilot.app.dialog.finish()
        await pilot.pause()

    assert pilot.app.results == [None]
