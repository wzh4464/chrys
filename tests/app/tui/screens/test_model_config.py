# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the model configuration screen."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Checkbox, Input, Label, OptionList, Select, Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.main.config_actions import RuntimeConfigController
from chrys.app.tui.screens.main.session_handlers import SessionCallbacks, SessionHandler
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.screens.models.screen import (
    _API_STYLE_CHAT_COMPLETIONS,
    _API_STYLE_RESPONSES,
    _BASE_URL,
    _CHAT_OPTIONS,
    _HEADER_NAME_PLACEHOLDER,
    _HEADER_VALUE_PLACEHOLDER,
    _OPTION_NAME_PLACEHOLDER,
    _OPTION_VALUE_PLACEHOLDER,
    _PROVIDER_BASE_URL,
    _VISION,
    ModelConfigScreen,
)
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentRuntimeDetails,
    ProfileSwitched,
    RuntimeModelDetails,
    SettingsReload,
)
from chrys.foundation.i18n import Localizer
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.util.chrys_headers import MODEL_ID_HEADER, X_SESSION_ID_HEADER
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.waiting import wait_for, wait_until


def test_model_labels_follow_localization_contract() -> None:
    assert format_message(_API_STYLE_CHAT_COMPLETIONS.bind()) == "Chat Completions"
    assert format_message(_API_STYLE_RESPONSES.bind()) == "Responses"
    assert format_message(_VISION.bind()) == "Vision Model"
    chinese = Localizer("zh-Hans")
    assert chinese.render(_API_STYLE_CHAT_COMPLETIONS.bind()) == "Chat Completions"
    assert chinese.render(_API_STYLE_RESPONSES.bind()) == "Responses"
    assert chinese.render(_BASE_URL.bind()) == "服务地址"
    assert chinese.render(_PROVIDER_BASE_URL.bind(provider="OpenAI")) == "OpenAI 服务地址"
    assert chinese.render(_CHAT_OPTIONS.bind()) == "Chat 选项（额外请求字段）"  # noqa: RUF001
    assert chinese.render(_VISION.bind()) == "视觉模型"


def test_model_option_and_header_placeholders_show_usable_examples() -> None:
    assert format_message(_HEADER_NAME_PLACEHOLDER.bind()) == "e.g. X-Auth-Token"
    assert format_message(_HEADER_VALUE_PLACEHOLDER.bind()) == "e.g. {{AUTH_TOKEN}}"
    assert format_message(_OPTION_NAME_PLACEHOLDER.bind()) == "e.g. extra_body, temperature"
    assert format_message(_OPTION_VALUE_PLACEHOLDER.bind()) == "e.g. 0.7, true, or {...}"

    chinese = Localizer("zh-Hans")
    assert chinese.render(_HEADER_NAME_PLACEHOLDER.bind()) == "例如 X-Auth-Token"
    assert chinese.render(_HEADER_VALUE_PLACEHOLDER.bind()) == "例如 {{AUTH_TOKEN}}"
    assert chinese.render(_OPTION_NAME_PLACEHOLDER.bind()) == "例如 extra_body、temperature"
    assert chinese.render(_OPTION_VALUE_PLACEHOLDER.bind()) == "例如 0.7、true 或 {...}"


class _ModelConfigApp(App):
    locale_controller = LocaleController(Settings(locale="en"))

    def compose(self) -> ComposeResult:
        yield Static("placeholder")


@pytest.fixture(autouse=True)
def _isolate_chrys_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep profile serializer paths inside pytest temp dirs."""
    fake_platform = type("P", (), {"config_dir": tmp_path})()
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "")
    monkeypatch.delenv("CHRYS_MODEL_PROFILE")


async def _wait_for_kv_rows(container, pilot, count: int) -> None:
    """Poll until async key-value row mounts (and their inputs) are ready."""

    # Wait for both the row container AND its key/value Input children. On
    # slow runners (e.g. Windows CI) the row mounts a tick before its
    # children, so querying inputs immediately after row count is satisfied
    # races. 40 * 0.05s ≈ 2s budget — plenty for any realistic runner.
    for _ in range(40):
        rows = list(container.query(".mc-kv-item-row"))
        if len(rows) >= count and all(
            row.query(".mc-kv-key-input") and row.query(".mc-kv-value-input") for row in rows
        ):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Expected at least {count} key-value rows with inputs to be mounted")


def _confirmation_session_handler(state: MainScreenState, services: MainScreenServices) -> SessionHandler:
    """Build a SessionHandler wired to sync the process-effective cache on confirmation."""

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    def _set_active_model_profile_id(value: str) -> None:
        services.active_model_profile_id = value

    handler = SessionHandler(
        state=state,
        services=services,
        view=cast(Any, SimpleNamespace()),
        callbacks=SessionCallbacks(
            set_agent_loading=_noop,
            set_has_messages=_noop,
            set_creating_new_session=_noop,
            set_restoring_session=_noop,
            set_profile_display=_noop,
            set_active_model_profile_id=_set_active_model_profile_id,
            set_workspace_cwd=_noop,
            set_workspace_original_cwd=_noop,
            update_subtitle=_noop,
            update_toc=_noop,
            clear_suggestion_file_cache=_noop,
            start_session_restore=_noop,
            post_gc_message=_noop,
            debug=_noop,
            refresh_model_indicator=_noop,
        ),
        agent_load=cast(
            Any,
            SimpleNamespace(
                begin_session_restore_load=_noop,
                cancel_agent_load=_noop,
                finish_agent_load=_noop,
            ),
        ),
    )
    handler._presenter = cast(
        Any,
        SimpleNamespace(set_tool_info=_noop, clear_status=_noop, flash_status=_noop),
    )
    return handler


async def _model_config_result_events(result: str, registry: ModelProfileRegistry) -> list[SettingsReload]:
    bus = EventBus()
    published: list[SettingsReload] = []

    async def capture(event: SettingsReload) -> None:
        published.append(event)

    await bus.subscribe(SettingsReload, capture)
    controller = RuntimeConfigController(
        state=MainScreenState(),
        services=MainScreenServices(bus=bus, model_registry=registry),
        view=cast(Any, SimpleNamespace(notify=lambda *_args, **_kwargs: None)),
        callbacks=cast(Any, SimpleNamespace(debug=lambda _key, _message="": None)),
    )
    await controller.on_model_config_result(result)
    return published


def test_provider_editor_tables_cover_every_provider() -> None:
    """The editor's label and default-base-url maps have silent ``.title()``/empty
    fallbacks, so a provider missing from either ships a visibly broken editor."""
    from chrys.app.tui.screens.models.screen import _PROVIDER_DEFAULT_BASE_URLS, _PROVIDER_LABELS, _PROVIDERS

    provider_ids = {provider_id for _label, provider_id in _PROVIDERS}
    assert provider_ids <= set(_PROVIDER_LABELS)
    assert provider_ids <= set(_PROVIDER_DEFAULT_BASE_URLS)
    assert _PROVIDER_LABELS["glm-openai"] == "GLM (OpenAI)"
    assert _PROVIDER_DEFAULT_BASE_URLS["glm-openai"] == "https://open.bigmodel.cn/api/paas/v4"


@pytest.mark.asyncio
async def test_model_config_container_default_size_limits() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(160, 80)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        container = screen.query_one("#mc-container", Vertical)
        assert str(container.styles.max_width) == "120"
        assert str(container.styles.max_height) == "60"


@pytest.mark.asyncio
async def test_model_config_uses_default_token_limits() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#mc-max-tokens", Input).value == "200000"
        assert screen.query_one("#mc-max-output-tokens", Input).value == "32000"


@pytest.mark.asyncio
async def test_model_config_input_ctrl_a_selects_and_deletes_text() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="deepseek-v4-flash")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        model_input = screen.query_one("#mc-model", Input)
        model_input.focus()
        await pilot.pause()

        await pilot.press("ctrl+a")
        await pilot.pause()

        assert model_input.selected_text == "deepseek-v4-flash"

        await pilot.press("backspace")
        await pilot.pause()

        assert model_input.value == ""


@pytest.mark.asyncio
async def test_model_config_clone_saves_copy_with_new_id(
    tmp_path: Path,
) -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        provider="anthropic",
        model_id="claude-test",
        max_context_tokens=200000,
        base_url="https://example.test",
        api_key="{{MODEL_KEY}}",
        http_connect_timeout=5.0,
        http_read_timeout=60.0,
        http_max_retries=4,
        verify_ssl=False,
        bypass_proxy=True,
        http_headers=json.dumps({"X-Team": "platform"}),
        chat_options=json.dumps({"temperature": 0.7}),
        stream=True,
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#mc-clone", Button).press()
        # The clone handler hops to a thread for the save; poll for its
        # selection switch instead of betting on a single pause.
        await wait_for(
            lambda: screen._selected_profile_id != profile.id,
            pilot=pilot,
            description="clone selected",
        )

        copied = registry.get(screen._selected_profile_id)

    assert copied is not None
    assert copied.id != profile.id
    assert copied.name == "Model A Copy"
    assert copied.provider == profile.provider
    assert copied.model_id == profile.model_id
    assert copied.max_context_tokens == profile.max_context_tokens
    assert copied.base_url == profile.base_url
    assert copied.api_key == profile.api_key
    assert copied.http_connect_timeout == profile.http_connect_timeout
    assert copied.http_read_timeout == profile.http_read_timeout
    assert copied.http_max_retries == profile.http_max_retries
    assert copied.verify_ssl == profile.verify_ssl
    assert copied.bypass_proxy == profile.bypass_proxy
    assert copied.http_headers == profile.http_headers
    assert copied.chat_options == profile.chat_options
    assert copied.stream == profile.stream
    assert (tmp_path / "models" / f"{copied.id}.yaml").is_file()


@pytest.mark.asyncio
async def test_model_config_clone_increments_existing_clone_suffix() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        # Each clone must land before the next press: the handler saves on a
        # thread, and the following clone reads the selection it installs.
        for expected in range(2, 5):
            screen.query_one("#mc-clone", Button).press()
            await wait_for(
                lambda expected=expected: len(registry.list_profiles()) == expected,
                pilot=pilot,
                description="clone registered",
            )

        selected = registry.get(screen._selected_profile_id)

    assert selected is not None
    assert selected.name == "Model A Copy 3"
    assert {p.name for p in registry.list_profiles()} == {
        "Model A",
        "Model A Copy",
        "Model A Copy 2",
        "Model A Copy 3",
    }


@pytest.mark.asyncio
async def test_model_config_clone_increments_clone_suffix_without_root_profile() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-copy", name="Model A copy 2", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#mc-clone", Button).press()
        await wait_for(
            lambda: screen._selected_profile_id != profile.id,
            pilot=pilot,
            description="clone selected",
        )

        selected = registry.get(screen._selected_profile_id)

    assert selected is not None
    assert selected.name == "Model A Copy 3"


@pytest.mark.asyncio
async def test_model_config_footer_buttons_stay_inside_modal_after_clone() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(90, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#mc-clone", Button).press()
        await wait_for(
            lambda: len(registry.list_profiles()) == 2,
            pilot=pilot,
            description="clone registered",
        )

        container = screen.query_one("#mc-container", Vertical)
        container_left = container.region.x
        container_right = container.region.x + container.region.width
        buttons = [
            screen.query_one(f"#{button_id}", Button)
            for button_id in ("mc-new", "mc-clone", "mc-delete", "mc-save", "mc-cancel")
        ]
        sidebar = screen.query_one("#mc-list", OptionList)

        assert all(container_left <= button.region.x for button in buttons)
        assert all(button.region.x + button.region.width <= container_right for button in buttons)
        assert list(screen.query("#mc-activate")) == []
        assert all("(Active)" not in sidebar.get_option(item.id).prompt.plain for item in registry.list_profiles())

    css_path = Path(__file__).parents[4] / "src/chrys/app/tui/screens/models/screen.tcss"
    assert "#mc-activate" not in css_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_model_config_save_global_default_is_file_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.service.profiles.models.env_bridge import (
        get_global_default_profile_id,
        set_global_default_profile_id,
    )
    from chrys.service.profiles.models.serializer import save_profile

    io_threads: list[int] = []

    def recording_save(profile: ModelProfile) -> Path:
        io_threads.append(threading.get_ident())
        return save_profile(profile)

    def recording_default(profile_id: str) -> None:
        io_threads.append(threading.get_ident())
        set_global_default_profile_id(profile_id)

    monkeypatch.setattr("chrys.service.profiles.models.serializer.save_profile", recording_save)
    monkeypatch.setattr(
        "chrys.service.profiles.models.env_bridge.set_global_default_profile_id",
        recording_default,
    )

    registry = ModelProfileRegistry()
    global_default = ModelProfile(id="model-a", name="Model A", model_id="global-wire")
    runtime_effective = ModelProfile(id="model-b", name="Model B", model_id="runtime-wire")
    registry.register(global_default)
    registry.register(runtime_effective)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", runtime_effective.id)
    # The dotenv starts without a pointer so the assertion below proves the
    # save path itself performed the file write.
    assert get_global_default_profile_id() == ""

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=global_default.id)
        await app.push_screen(screen)
        await pilot.pause()

        saved = await screen._save_only()
        result = screen._cancel_result()

        assert saved is not None
        assert get_global_default_profile_id() == global_default.id
        assert os.environ["CHRYS_MODEL_PROFILE"] == runtime_effective.id
        assert result == "updated"
        assert len(io_threads) == 2
        assert threading.get_ident() not in io_threads

    published = await _model_config_result_events(result, registry)
    assert len(published) == 1


@pytest.mark.asyncio
async def test_first_save_with_no_global_default_promotes_and_close_adopts() -> None:
    """A first configured model becomes the default at save and active at close."""
    from chrys.service.profiles.models.env_bridge import get_global_default_profile_id

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-a", name="Model A", model_id="first-wire"))
    assert get_global_default_profile_id() == ""

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id="")
        await app.push_screen(screen)
        await pilot.pause()

        saved = await screen._save_only()
        result = screen._cancel_result()

        assert saved is not None
        # The save claims the empty global default, file-only: the process
        # pointer stays unset until the screen closes.
        assert get_global_default_profile_id() == "model-a"
        assert "CHRYS_MODEL_PROFILE" not in os.environ
        assert result == "updated"

    published = await _model_config_result_events(result, registry)
    assert len(published) == 1
    assert os.environ["CHRYS_MODEL_PROFILE"] == "model-a"


@pytest.mark.asyncio
async def test_save_does_not_override_existing_global_default() -> None:
    from chrys.service.profiles.models.env_bridge import (
        get_global_default_profile_id,
        set_global_default_profile_id,
    )

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-a", name="Model A", model_id="first-wire"))
    registry.register(ModelProfile(id="model-b", name="Model B", model_id="default-wire"))
    set_global_default_profile_id("model-b")

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id="model-b")
        await app.push_screen(screen)
        await pilot.pause()
        screen._load_profile("model-a")
        await pilot.pause()

        saved = await screen._save_only()

        assert saved is not None
        assert saved.id == "model-a"
        assert get_global_default_profile_id() == "model-b"


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_default", ["ghost-id", "hollow-id"])
async def test_save_reclaims_unresolvable_global_default(stale_default: str) -> None:
    """A stored default whose profile vanished (models directory replaced,
    file deleted externally) or was never filled in is no default at all;
    the next successful save claims the pointer like the empty case."""
    from chrys.service.profiles.models.env_bridge import (
        get_global_default_profile_id,
        set_global_default_profile_id,
    )

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-a", name="Model A", model_id="first-wire"))
    registry.register(ModelProfile(id="hollow-id", name="Hollow"))
    set_global_default_profile_id(stale_default)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=stale_default)
        await app.push_screen(screen)
        await pilot.pause()
        screen._load_profile("model-a")
        await pilot.pause()

        saved = await screen._save_only()

        assert saved is not None
        assert saved.id == "model-a"
        assert get_global_default_profile_id() == "model-a"


@pytest.mark.asyncio
async def test_rename_normalizes_name_based_process_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Renaming the runtime-effective profile must not strand a name-based
    process pointer: it is normalized to the profile id, or modal close
    would misread the runtime as inactive and adopt the global default —
    silently switching the live model."""
    from chrys.service.profiles.models.env_bridge import (
        get_global_default_profile_id,
        set_global_default_profile_id,
    )

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-a", name="Old Name", model_id="runtime-wire"))
    registry.register(ModelProfile(id="model-b", name="Model B", model_id="default-wire"))
    set_global_default_profile_id("model-b")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "Old Name")

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id="model-b")
        await app.push_screen(screen)
        await pilot.pause()
        screen._load_profile("model-a")
        await pilot.pause()
        screen.query_one("#mc-name", Input).value = "New Name"

        saved = await screen._save_only()

        assert saved is not None
        assert saved.name == "New Name"
        assert os.environ["CHRYS_MODEL_PROFILE"] == "model-a"
        assert get_global_default_profile_id() == "model-b"


@pytest.mark.asyncio
async def test_rename_leaves_other_profiles_name_pointer_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.service.profiles.models.env_bridge import set_global_default_profile_id

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-a", name="Old Name", model_id="runtime-wire"))
    registry.register(ModelProfile(id="model-b", name="Model B", model_id="default-wire"))
    set_global_default_profile_id("model-b")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "Model B")

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id="model-b")
        await app.push_screen(screen)
        await pilot.pause()
        screen._load_profile("model-a")
        await pilot.pause()
        screen.query_one("#mc-name", Input).value = "New Name"

        await screen._save_only()

        assert os.environ["CHRYS_MODEL_PROFILE"] == "Model B"


@pytest.mark.asyncio
async def test_close_keeps_existing_process_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.service.profiles.models.env_bridge import set_global_default_profile_id

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-a", name="Model A", model_id="first-wire"))
    registry.register(ModelProfile(id="model-b", name="Model B", model_id="active-wire"))
    set_global_default_profile_id("model-a")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "model-b")

    published = await _model_config_result_events("updated", registry)

    assert len(published) == 1
    assert os.environ["CHRYS_MODEL_PROFILE"] == "model-b"


@pytest.mark.asyncio
async def test_close_repoints_dangling_process_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process pointer whose profile no longer exists preserves nothing;
    closing the modal adopts the valid file default instead of leaving the
    runtime stuck on the built-in placeholder."""
    from chrys.service.profiles.models.env_bridge import set_global_default_profile_id

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-a", name="Model A", model_id="first-wire"))
    set_global_default_profile_id("model-a")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "ghost-id")

    published = await _model_config_result_events("updated", registry)

    assert len(published) == 1
    assert os.environ["CHRYS_MODEL_PROFILE"] == "model-a"


@pytest.mark.asyncio
async def test_close_does_not_adopt_unresolvable_file_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the file default is itself hollow, adoption stands down rather
    than swapping one unusable pointer for another."""
    from chrys.service.profiles.models.env_bridge import set_global_default_profile_id

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="hollow-id", name="Hollow"))
    set_global_default_profile_id("hollow-id")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "ghost-id")

    published = await _model_config_result_events("updated", registry)

    assert len(published) == 1
    assert os.environ["CHRYS_MODEL_PROFILE"] == "ghost-id"


@pytest.mark.asyncio
async def test_model_config_delete_runtime_effective_reloads_from_global_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.service.profiles.models.env_bridge import (
        get_global_default_profile_id,
        set_global_default_profile_id,
    )

    registry = ModelProfileRegistry()
    global_default = ModelProfile(id="model-a", name="Model A", model_id="global-wire")
    runtime_effective = ModelProfile(id="model-b", name="Model B", model_id="runtime-wire")
    registry.register(global_default)
    registry.register(runtime_effective)
    set_global_default_profile_id(global_default.id)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", runtime_effective.id)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=global_default.id)
        await app.push_screen(screen)
        await pilot.pause()

        await screen._do_delete(runtime_effective.id)
        result = screen._cancel_result()

        assert registry.get(runtime_effective.id) is None
        assert get_global_default_profile_id() == global_default.id
        assert os.environ["CHRYS_MODEL_PROFILE"] == global_default.id
        assert result == "switched"

    published = await _model_config_result_events(result, registry)
    assert len(published) == 1

    # The reload's backend confirmation must land runtime details and the
    # process-effective cache on the promoted profile.
    state = MainScreenState()
    services = MainScreenServices(
        bus=EventBus(),
        model_registry=registry,
        active_model_profile_id=runtime_effective.id,
    )
    handler = _confirmation_session_handler(state, services)
    await handler.on_profile_switched(
        ProfileSwitched(
            from_profile="Code",
            to_profile="Code",
            runtime_details=AgentRuntimeDetails(
                model=RuntimeModelDetails(
                    profile_id=global_default.id,
                    name=global_default.name,
                    selection_source="active",
                )
            ),
        )
    )
    assert services.active_model_profile_id == global_default.id
    assert state.runtime.details_confirmed is True
    assert state.runtime.details.model.profile_id == global_default.id


@pytest.mark.asyncio
async def test_model_config_delete_global_default_leaves_other_runtime_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.service.profiles.models.env_bridge import (
        get_global_default_profile_id,
        set_global_default_profile_id,
    )

    registry = ModelProfileRegistry()
    global_default = ModelProfile(id="model-a", name="Model A", model_id="global-wire")
    promoted_default = ModelProfile(id="model-c", name="Model C", model_id="promoted-wire")
    runtime_effective = ModelProfile(id="model-b", name="Model B", model_id="runtime-wire")
    registry.register(global_default)
    registry.register(promoted_default)
    registry.register(runtime_effective)
    set_global_default_profile_id(global_default.id)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", runtime_effective.id)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=global_default.id)
        await app.push_screen(screen)
        await pilot.pause()

        await screen._do_delete(global_default.id)
        result = screen._cancel_result()

        assert registry.get(global_default.id) is None
        assert get_global_default_profile_id() == promoted_default.id
        assert os.environ["CHRYS_MODEL_PROFILE"] == runtime_effective.id
        assert result == ""

    published = await _model_config_result_events(result, registry)
    assert published == []


@pytest.mark.asyncio
async def test_model_config_delete_recognizes_name_based_process_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime-effective check resolves the pointer by id OR unique name;
    a name pointer at the deleted profile must resync the environment and
    request a reload instead of keeping the dead executor running."""
    from chrys.service.profiles.models.env_bridge import set_global_default_profile_id

    registry = ModelProfileRegistry()
    global_default = ModelProfile(id="model-a", name="Model A", model_id="global-wire")
    runtime_effective = ModelProfile(id="model-b", name="Model B", model_id="runtime-wire")
    registry.register(global_default)
    registry.register(runtime_effective)
    set_global_default_profile_id(global_default.id)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", runtime_effective.name)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=global_default.id)
        await app.push_screen(screen)
        await pilot.pause()

        await screen._do_delete(runtime_effective.id)
        result = screen._cancel_result()

        assert os.environ["CHRYS_MODEL_PROFILE"] == global_default.id
        assert result == "switched"


@pytest.mark.asyncio
async def test_model_config_delete_global_default_promotes_first_selectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promotion skips hollow profiles: a never-filled auto-seeded profile can
    precede valid ones in registry order, and promoting it would point both
    pointers at an unusable model while real ones exist."""
    from chrys.service.profiles.models.env_bridge import (
        get_global_default_profile_id,
        set_global_default_profile_id,
    )

    registry = ModelProfileRegistry()
    hollow = ModelProfile(id="model-hollow", name="Profile 1")
    global_default = ModelProfile(id="model-a", name="Model A", model_id="global-wire")
    selectable = ModelProfile(id="model-b", name="Model B", model_id="promoted-wire")
    registry.register(hollow)
    registry.register(global_default)
    registry.register(selectable)
    set_global_default_profile_id(global_default.id)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", global_default.id)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=global_default.id)
        await app.push_screen(screen)
        await pilot.pause()

        await screen._do_delete(global_default.id)

        assert get_global_default_profile_id() == selectable.id
        assert os.environ["CHRYS_MODEL_PROFILE"] == selectable.id


@pytest.mark.asyncio
@pytest.mark.parametrize("file_default_pointer", ["model-hollow", "ghost-id"])
async def test_model_config_delete_runtime_effective_skips_unresolvable_file_default(
    file_default_pointer: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting the runtime-effective profile must land the process pointer on
    a selectable profile even when the file default is hollow or dangling;
    adopting that default verbatim would strand the session on the
    placeholder while a usable profile exists."""
    from chrys.service.profiles.models.env_bridge import (
        get_global_default_profile_id,
        set_global_default_profile_id,
    )

    registry = ModelProfileRegistry()
    hollow = ModelProfile(id="model-hollow", name="Profile 1")
    runtime_effective = ModelProfile(id="model-a", name="Model A", model_id="runtime-wire")
    selectable = ModelProfile(id="model-b", name="Model B", model_id="fallback-wire")
    registry.register(hollow)
    registry.register(runtime_effective)
    registry.register(selectable)
    set_global_default_profile_id(file_default_pointer)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", runtime_effective.id)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=file_default_pointer)
        await app.push_screen(screen)
        await pilot.pause()

        await screen._do_delete(runtime_effective.id)
        result = screen._cancel_result()

        # The file pointer is repaired only when the default itself is
        # deleted; the process pointer still lands on a usable profile.
        assert get_global_default_profile_id() == file_default_pointer
        assert os.environ["CHRYS_MODEL_PROFILE"] == selectable.id
        assert result == "switched"


@pytest.mark.asyncio
async def test_model_config_delete_runtime_effective_clears_pointer_when_nothing_selectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.service.profiles.models.env_bridge import set_global_default_profile_id

    registry = ModelProfileRegistry()
    hollow = ModelProfile(id="model-hollow", name="Profile 1")
    runtime_effective = ModelProfile(id="model-a", name="Model A", model_id="runtime-wire")
    registry.register(hollow)
    registry.register(runtime_effective)
    set_global_default_profile_id(hollow.id)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", runtime_effective.id)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=hollow.id)
        await app.push_screen(screen)
        await pilot.pause()

        await screen._do_delete(runtime_effective.id)

        assert "CHRYS_MODEL_PROFILE" not in os.environ


@pytest.mark.asyncio
async def test_model_config_delete_callback_reports_failure_without_exiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)
    registry.register(ModelProfile(id="model-b", name="Model B", model_id="gpt-test-2"))

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()
        captured = _capture_notifications(screen)

        async def _fail_delete(_profile_id: str) -> None:
            raise TimeoutError("settings lock timed out")

        monkeypatch.setattr(screen, "_do_delete", _fail_delete)
        screen.query_one("#mc-delete", Button).press()
        await pilot.pause()
        dialog = app.screen
        # The buttons are composed by the nested DialogButtonRow, which mounts
        # a refresh after the dialog itself; poll rather than assume one pause.
        assert await wait_until(lambda: bool(dialog.query("#confirm-yes")), pilot=pilot), "confirm button never mounted"
        dialog.query_one("#confirm-yes", Button).press()
        await pilot.pause()

        assert app.screen is screen
        assert captured == [("error", "settings lock timed out")]


@pytest.mark.asyncio
async def test_model_config_read_only_hides_mutations_and_does_not_write(tmp_path: Path) -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        provider="openai",
        model_id="gpt-test",
        http_headers=json.dumps({"X-Test": "true"}),
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id, read_only=True)
        await app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#mc-name", Input).disabled is True
        assert screen.query_one("#mc-provider", Select).disabled is True
        assert screen.query_one("#mc-stream", Checkbox).disabled is True
        assert screen.query_one("#mc-right", Vertical).disabled is False
        assert screen.query_one("#mc-cancel", Button).display is True
        assert screen.query_one("#mc-cancel", Button).disabled is False
        notice = screen.query_one("#mc-read-only-notice", Static)
        assert notice.display is True
        assert notice.render().plain == "• Agent is running. This page is read-only."
        assert screen.query_one("#mc-buttons-spacer", Static).display is True
        footer = screen.query_one("#mc-footer", Vertical)
        close = screen.query_one("#mc-cancel", Button)
        assert notice.region.y == footer.region.y + footer.region.height - 1
        assert notice.region.y > close.region.y
        assert notice.region.x == footer.region.x + 1
        assert notice.region.width == footer.region.width - 2
        assert list(screen.query("#mc-activate")) == []
        for button_id in ("mc-new", "mc-clone", "mc-delete", "mc-save"):
            button = screen.query_one(f"#{button_id}", Button)
            assert button.display is False
            assert button.disabled is True
        assert all(button.display is False for button in screen.query(".mc-kv-add-btn"))
        assert all(button.display is False for button in screen.query(".mc-kv-remove-btn"))

        screen.query_one("#mc-save", Button).press()
        await pilot.pause()

        assert screen._cancel_result() == ""
        assert not (tmp_path / "models").exists()


@pytest.mark.asyncio
async def test_model_config_read_only_empty_registry_does_not_seed_profile(tmp_path: Path) -> None:
    registry = ModelProfileRegistry()

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, read_only=True)
        await app.push_screen(screen)
        await pilot.pause()

        assert registry.list_profiles() == []
        assert not (tmp_path / "models").exists()
        assert screen.query_one("#mc-save", Button).display is False


@pytest.mark.asyncio
async def test_model_config_saves_draft_key_value_rows_without_add() -> None:
    """Filled header and chat-option rows should be serialized without pressing Add."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        headers_container = screen.query_one("#mc-headers-list")
        header_row = headers_container.query_one(".mc-kv-item-row")
        header_row.query_one(".mc-kv-key-input", Input).value = "X-Team"
        header_row.query_one(".mc-kv-value-input", Input).value = "platform"

        options_container = screen.query_one("#mc-options-list")
        option_row = options_container.query_one(".mc-kv-item-row")
        option_row.query_one(".mc-kv-key-input", Input).value = "temperature"
        option_row.query_one(".mc-kv-value-input", Input).value = "0.7"

        saved = screen._build_profile_from_form()

    assert json.loads(saved.http_headers) == {"X-Team": "platform"}
    assert json.loads(saved.chat_options) == {"temperature": 0.7}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "deepseek-openai"])
async def test_model_config_responses_capable_provider_api_style_round_trip(provider: str) -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        provider=provider,
        api_style="responses",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        api_style = screen.query_one("#mc-api-style", Select)
        assert screen.query_one("#mc-api-style-label", Label).display is True
        assert api_style.display is True
        assert api_style.value == "responses"

        responses_saved = screen._build_profile_from_form()
        api_style.value = "chat_completions"
        saved = screen._build_profile_from_form()

        api_style.value = "responses"
        screen.query_one("#mc-provider", Select).value = "anthropic"
        screen._update_provider_labels("anthropic")
        hidden_saved = screen._build_profile_from_form()

    assert responses_saved.provider == provider
    assert responses_saved.api_style == "responses"
    assert saved.api_style == "chat_completions"
    assert hidden_saved.provider == "anthropic"
    assert hidden_saved.api_style == "chat_completions"


@pytest.mark.asyncio
async def test_max_output_tokens_label_shows_wire_param_per_provider() -> None:
    """The Max Output Tokens label surfaces the actual wire parameter so users
    can tell max_tokens providers apart from max_completion_tokens ones."""
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        label = screen.query_one("#mc-max-output-label", Label)

        def label_shows(param: str) -> Callable[[], bool]:
            return lambda: f"({param})" in label.visual.plain

        await wait_for(label_shows("max_completion_tokens"), pilot=pilot, description="initial label")

        screen.query_one("#mc-api-style", Select).value = "responses"
        await wait_for(label_shows("max_output_tokens"), pilot=pilot, description="responses label")

        screen.query_one("#mc-provider", Select).value = "deepseek-openai"
        screen.query_one("#mc-api-style", Select).value = "responses"
        await wait_for(label_shows("max_output_tokens"), pilot=pilot, description="deepseek responses label")

        screen.query_one("#mc-api-style", Select).value = "chat_completions"
        await wait_for(label_shows("max_tokens"), pilot=pilot, description="deepseek chat label")

        for provider in ("glm-openai", "anthropic"):
            screen.query_one("#mc-provider", Select).value = provider
            await wait_for(label_shows("max_tokens"), pilot=pilot, description=f"{provider} label")

        screen.query_one("#mc-provider", Select).value = "openai"
        await wait_for(label_shows("max_completion_tokens"), pilot=pilot, description="openai label restored")


@pytest.mark.asyncio
async def test_model_config_preserves_env_templates_in_value_fields() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#mc-api-key", Input).value = "{{CHRYS_OPENAI_KEY}}"

        headers_container = screen.query_one("#mc-headers-list")
        header_row = headers_container.query_one(".mc-kv-item-row")
        header_row.query_one(".mc-kv-key-input", Input).value = "Authorization"
        header_row.query_one(".mc-kv-value-input", Input).value = "Bearer {{CHRYS_HEADER_TOKEN}}"

        options_container = screen.query_one("#mc-options-list")
        option_row = options_container.query_one(".mc-kv-item-row")
        option_row.query_one(".mc-kv-key-input", Input).value = "metadata"
        option_row.query_one(".mc-kv-value-input", Input).value = '{"token": "{{CHRYS_CHAT_TOKEN}}"}'

        saved = screen._build_profile_from_form()

    assert saved.api_key == "{{CHRYS_OPENAI_KEY}}"
    assert json.loads(saved.http_headers) == {"Authorization": "Bearer {{CHRYS_HEADER_TOKEN}}"}
    assert json.loads(saved.chat_options) == {"metadata": {"token": "{{CHRYS_CHAT_TOKEN}}"}}


@pytest.mark.asyncio
async def test_model_config_saves_http_header_values_as_strings_when_json_like() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        headers_container = screen.query_one("#mc-headers-list")
        header_row = headers_container.query_one(".mc-kv-item-row")
        header_row.query_one(".mc-kv-key-input", Input).value = "X-Config"
        header_row.query_one(".mc-kv-value-input", Input).value = '{"nested": true}'

        saved = screen._build_profile_from_form()

    assert json.loads(saved.http_headers) == {"X-Config": '{"nested": true}'}


@pytest.mark.asyncio
async def test_model_config_add_button_appends_another_editable_row() -> None:
    """Add creates another editable row instead of committing the current row."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        headers_container = screen.query_one("#mc-headers-list")
        first_row = headers_container.query_one(".mc-kv-item-row")
        first_row.query_one(".mc-kv-key-input", Input).value = "X-Team"
        first_row.query_one(".mc-kv-value-input", Input).value = "platform"

        screen.query_one("#mc-hadd").press()
        await _wait_for_kv_rows(headers_container, pilot, 2)

        rows = list(headers_container.query(".mc-kv-item-row"))
        assert len(rows) == 2
        rows[1].query_one(".mc-kv-key-input", Input).value = "X-Env"
        rows[1].query_one(".mc-kv-value-input", Input).value = "dev"

        saved = screen._build_profile_from_form()

    assert json.loads(saved.http_headers) == {"X-Team": "platform", "X-Env": "dev"}


@pytest.mark.asyncio
async def test_model_config_validate_rejects_partial_key_value_rows() -> None:
    """Partially-filled key-value rows should block Save with clear errors."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        headers_container = screen.query_one("#mc-headers-list")
        header_row = headers_container.query_one(".mc-kv-item-row")
        header_row.query_one(".mc-kv-key-input", Input).value = "X-Team"

        options_container = screen.query_one("#mc-options-list")
        option_row = options_container.query_one(".mc-kv-item-row")
        option_row.query_one(".mc-kv-value-input", Input).value = "0.7"

        errors = screen._validate()

    assert "HTTP Extra Headers row 1: value is required for key 'X-Team'." in errors
    assert "Chat Options row 1: key name is required when a value is set." in errors


@pytest.mark.asyncio
async def test_model_config_cross_validates_context_and_output_token_limits() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()
        context_input = screen.query_one("#mc-max-tokens", Input)
        output_input = screen.query_one("#mc-max-output-tokens", Input)

        context_input.value = "99"
        output_input.value = "50"
        too_small = screen._validate()
        context_input.value = "100"
        output_input.value = "100"
        overlap = screen._validate()
        output_input.value = "99"
        valid = screen._validate()
        context_input.value = "bad"
        malformed = screen._validate()
        context_input.value = ""
        blank = screen._validate()

    assert "Max context tokens must be at least 100." in too_small
    assert "Max output tokens must be less than max context tokens." in overlap
    assert not any("must be at least" in error or "must be less than" in error for error in valid)
    assert not any("must be at least" in error or "must be less than" in error for error in malformed)
    assert not any("must be at least" in error or "must be less than" in error for error in blank)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("messages", "[]", "'messages' is protected"),
        ("prompt", '{"id": "pmpt_1"}', "'prompt' is protected"),
        ("conversation_id", '"resp_1"', "'conversation_id' is protected"),
        ("extra_body", '{"input": []}', "extra_body contains protected key(s): input"),
        ("extra_body", '{"custom": true}', ""),
    ],
)
async def test_model_config_rejects_protected_chat_option_keys(key: str, value: str, expected: str) -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()
        row = screen.query_one("#mc-options-list").query_one(".mc-kv-item-row")
        row.query_one(".mc-kv-key-input", Input).value = key
        row.query_one(".mc-kv-value-input", Input).value = value
        errors = screen._validate()

    if expected:
        assert any(expected in error for error in errors)
    else:
        assert not any("protected" in error for error in errors)


@pytest.mark.asyncio
async def test_model_config_validate_rejects_duplicate_key_value_rows() -> None:
    """Duplicate keys should block Save instead of silently overwriting."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        headers_container = screen.query_one("#mc-headers-list")
        first_row = headers_container.query_one(".mc-kv-item-row")
        first_row.query_one(".mc-kv-key-input", Input).value = "X-Team"
        first_row.query_one(".mc-kv-value-input", Input).value = "platform"

        screen.query_one("#mc-hadd").press()
        await _wait_for_kv_rows(headers_container, pilot, 2)

        rows = list(headers_container.query(".mc-kv-item-row"))
        rows[1].query_one(".mc-kv-key-input", Input).value = "X-Team"
        rows[1].query_one(".mc-kv-value-input", Input).value = "infra"

        errors = screen._validate()

    assert "HTTP Extra Headers row 2: duplicate key 'X-Team'." in errors


@pytest.mark.asyncio
async def test_model_config_validate_rejects_chrys_prefixed_http_headers() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        headers_container = screen.query_one("#mc-headers-list")
        row = headers_container.query_one(".mc-kv-item-row")
        row.query_one(".mc-kv-key-input", Input).value = "CHRYS_TRACE"
        row.query_one(".mc-kv-value-input", Input).value = "1"

        errors = screen._validate()

    assert f"HTTP Extra Headers row 1: header key 'CHRYS_TRACE' is reserved for {APP_DISPLAY_NAME}." in errors


@pytest.mark.asyncio
async def test_model_config_validate_rejects_x_session_http_header() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        headers_container = screen.query_one("#mc-headers-list")
        row = headers_container.query_one(".mc-kv-item-row")
        row.query_one(".mc-kv-key-input", Input).value = X_SESSION_ID_HEADER
        row.query_one(".mc-kv-value-input", Input).value = "1"

        errors = screen._validate()

    assert f"HTTP Extra Headers row 1: header key '{X_SESSION_ID_HEADER}' is reserved for {APP_DISPLAY_NAME}." in errors


@pytest.mark.asyncio
async def test_model_config_validate_rejects_common_mapping_chat_options_that_are_not_objects() -> None:
    """Common mapping-valued chat options should fail in the modal, not during the next send."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        options_container = screen.query_one("#mc-options-list")
        option_row = options_container.query_one(".mc-kv-item-row")
        option_row.query_one(".mc-kv-key-input", Input).value = "metadata"
        value_input = option_row.query_one(".mc-kv-value-input", Input)

        value_input.value = '"not-a-map"'
        non_object_errors = screen._validate()

        value_input.value = "{not-json"
        invalid_json_errors = screen._validate()

        value_input.value = '{"X-Team": "platform"}'
        valid_errors = screen._validate()

        option_row.query_one(".mc-kv-key-input", Input).value = "extra_body"
        value_input.value = '"not-a-map"'
        extra_body_errors = screen._validate()

        option_row.query_one(".mc-kv-key-input", Input).value = "extra_headers"
        value_input.value = '{"X-Config": {"nested": true}}'
        nested_header_errors = screen._validate()

        value_input.value = '{"X-Config": "{\\"nested\\": true}"}'
        valid_header_errors = screen._validate()

        value_input.value = '{"chrys-trace": "1"}'
        reserved_header_errors = screen._validate()

        value_input.value = f'{{"{MODEL_ID_HEADER}": "wrong"}}'
        reserved_model_header_errors = screen._validate()

        value_input.value = f'{{"{X_SESSION_ID_HEADER}": "wrong"}}'
        reserved_x_session_header_errors = screen._validate()

    assert "Chat Options row 1: 'metadata' must be a JSON object/map, got str." in non_object_errors
    assert any("'metadata' must be a valid JSON object/map" in error for error in invalid_json_errors)
    assert not any("metadata" in error for error in valid_errors)
    assert "Chat Options row 1: 'extra_body' must be a JSON object/map, got str." in extra_body_errors
    assert "Chat Options row 1: 'extra_headers' value for header 'X-Config' must be a string." in nested_header_errors
    assert not any("extra_headers" in error for error in valid_header_errors)
    assert (
        f"Chat Options row 1: 'extra_headers' header 'chrys-trace' is reserved for {APP_DISPLAY_NAME}."
        in reserved_header_errors
    )
    assert (
        f"Chat Options row 1: 'extra_headers' header '{MODEL_ID_HEADER}' is reserved for {APP_DISPLAY_NAME}."
        in reserved_model_header_errors
    )
    assert (
        f"Chat Options row 1: 'extra_headers' header '{X_SESSION_ID_HEADER}' is reserved for {APP_DISPLAY_NAME}."
        in reserved_x_session_header_errors
    )


@pytest.mark.asyncio
async def test_model_config_validate_rejects_known_typed_chat_options() -> None:
    """Common chat options should be type-checked before save."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        options_container = screen.query_one("#mc-options-list")
        option_row = options_container.query_one(".mc-kv-item-row")
        key_input = option_row.query_one(".mc-kv-key-input", Input)
        value_input = option_row.query_one(".mc-kv-value-input", Input)

        key_input.value = "temperature"
        value_input.value = '"hot"'
        temperature_errors = screen._validate()

        value_input.value = "0.7"
        valid_temperature_errors = screen._validate()

        key_input.value = "max_tokens"
        value_input.value = "0"
        max_tokens_errors = screen._validate()

        key_input.value = "max_output_tokens"
        value_input.value = "4096"
        max_output_tokens_errors = screen._validate()

        key_input.value = "max_completion_tokens"
        value_input.value = "4096"
        max_completion_tokens_errors = screen._validate()

        key_input.value = "top_p"
        value_input.value = "1.5"
        top_p_errors = screen._validate()

        key_input.value = "seed"
        value_input.value = "1.5"
        seed_errors = screen._validate()

        key_input.value = "store"
        value_input.value = '"true"'
        bool_errors = screen._validate()

        key_input.value = "logit_bias"
        value_input.value = '{"42": 101}'
        logit_bias_errors = screen._validate()

        key_input.value = "stop"
        value_input.value = '["DONE"]'
        valid_stop_errors = screen._validate()

        key_input.value = "top_k"
        value_input.value = "0"
        ignored_provider_option_errors = screen._validate()

    assert "Chat Options row 1: 'temperature' must be a JSON number between 0.0 and 2.0." in temperature_errors
    assert not any("temperature" in error for error in valid_temperature_errors)
    assert (
        "Chat Options row 1: 'max_tokens' is not saved on model profiles — "
        "remove this row and set the Max Output Tokens field above instead." in max_tokens_errors
    )
    assert (
        "Chat Options row 1: 'max_output_tokens' is not saved on model profiles — "
        "remove this row and set the Max Output Tokens field above instead." in max_output_tokens_errors
    )
    assert (
        "Chat Options row 1: 'max_completion_tokens' is not saved on model profiles — "
        "remove this row and set the Max Output Tokens field above instead." in max_completion_tokens_errors
    )
    assert "Chat Options row 1: 'top_p' must be a JSON number between 0.0 and 1.0." in top_p_errors
    assert "Chat Options row 1: 'seed' must be a JSON integer." in seed_errors
    assert "Chat Options row 1: 'store' must be a JSON boolean (true or false)." in bool_errors
    assert (
        "Chat Options row 1: 'logit_bias' value for token '42' must be a JSON number between -100 and 100."
        in logit_bias_errors
    )
    assert not any("stop" in error for error in valid_stop_errors)
    assert not any("top_k" in error for error in ignored_provider_option_errors)


@pytest.mark.asyncio
async def test_model_config_transport_checkboxes_default_secure_and_proxy_enabled() -> None:
    """Default model transport settings verify TLS and honor configured proxies."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        skip_tls = screen.query_one("#mc-skip-tls", Checkbox)
        bypass_proxy = screen.query_one("#mc-bypass-proxy", Checkbox)
        stream = screen.query_one("#mc-stream", Checkbox)
        vision = screen.query_one("#mc-vision", Checkbox)
        tls_hint = screen.query_one("#mc-skip-tls-hint", Label)
        provider_select = screen.query_one("#mc-provider", Select)
        model_input = screen.query_one("#mc-model", Input)
        api_key_input = screen.query_one("#mc-api-key", Input)
        connect_timeout_input = screen.query_one("#mc-connect-timeout", Input)
        headers_list = screen.query_one("#mc-headers-list")
        chat_options_list = screen.query_one("#mc-options-list")
        model_options = screen.query_one("#mc-model-options")
        connection_options = screen.query_one("#mc-connection-options")
        http_options = screen.query_one("#mc-http-options")
        extra_options = screen.query_one("#mc-extra-options")
        scroll_children = list(screen.query_one("#mc-scroll").children)
        model_children = list(model_options.children)
        sections_in_order = [
            scroll_children.index(model_options),
            scroll_children.index(connection_options),
            scroll_children.index(http_options),
            scroll_children.index(extra_options),
        ]
        streaming_in_model_options = stream.parent is model_options
        vision_in_model_options = vision.parent is model_options
        vision_is_last_in_model_options = model_children[-1] is vision
        provider_in_model_options = provider_select.parent is model_options
        model_id_in_model_options = model_input.parent is model_options
        api_key_in_model_options = api_key_input.parent is model_options
        http_timeout_in_http_options = connect_timeout_input.parent is http_options
        headers_in_extra_options = headers_list.parent is extra_options
        chat_options_in_extra_options = chat_options_list.parent is extra_options
        skip_tls_in_options = skip_tls.parent is connection_options
        saved = screen._build_profile_from_form()

    assert skip_tls.value is False
    assert bypass_proxy.value is False
    assert stream.value is False
    assert vision.value is False
    assert model_options.border_title == "Model Options"
    assert connection_options.border_title == "Connection Options"
    assert http_options.border_title == "HTTP Options"
    assert extra_options.border_title == "Extra Options"
    assert provider_in_model_options is True
    assert model_id_in_model_options is True
    assert api_key_in_model_options is True
    assert http_timeout_in_http_options is True
    assert headers_in_extra_options is True
    assert chat_options_in_extra_options is True
    assert skip_tls_in_options is True
    assert sections_in_order == sorted(sections_in_order)
    assert streaming_in_model_options is True
    assert vision_in_model_options is True
    assert vision_is_last_in_model_options is True
    assert tls_hint.display is False
    assert saved.verify_ssl is True
    assert saved.bypass_proxy is False
    assert saved.stream is False
    assert saved.vision is False


@pytest.mark.asyncio
async def test_model_config_max_output_tokens_round_trip() -> None:
    """The output cap is a required positive integer; blank/zero are rejected."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
        max_output_tokens=8192,
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        cap_input = screen.query_one("#mc-max-output-tokens", Input)
        assert cap_input.value == "8192"

        cap_input.value = "64000"
        assert screen._validate() == []
        assert screen._build_profile_from_form().max_output_tokens == 64000

        cap_input.value = ""
        assert any("Max output tokens is required" in e for e in screen._validate())

        for invalid in ("0", "-1", "abc"):
            cap_input.value = invalid
            assert any("Max output tokens" in e for e in screen._validate())


@pytest.mark.asyncio
async def test_model_config_transport_checkboxes_round_trip() -> None:
    """Skip TLS is inverted to verify_ssl; bypass proxy maps directly."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
        verify_ssl=False,
        bypass_proxy=True,
        stream=True,
        vision=True,
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        skip_tls = screen.query_one("#mc-skip-tls", Checkbox)
        bypass_proxy = screen.query_one("#mc-bypass-proxy", Checkbox)
        stream = screen.query_one("#mc-stream", Checkbox)
        vision = screen.query_one("#mc-vision", Checkbox)
        tls_hint = screen.query_one("#mc-skip-tls-hint", Label)
        api_key_input = screen.query_one("#mc-api-key", Input)
        model_options = screen.query_one("#mc-model-options")
        connection_options = screen.query_one("#mc-connection-options")
        http_options = screen.query_one("#mc-http-options")
        extra_options = screen.query_one("#mc-extra-options")
        scroll_children = list(screen.query_one("#mc-scroll").children)
        model_children = list(model_options.children)
        sections_in_order = [
            scroll_children.index(model_options),
            scroll_children.index(connection_options),
            scroll_children.index(http_options),
            scroll_children.index(extra_options),
        ]
        streaming_in_model_options = stream.parent is model_options
        vision_in_model_options = vision.parent is model_options
        vision_is_last_in_model_options = model_children[-1] is vision
        api_key_in_model_options = api_key_input.parent is model_options
        skip_tls_in_options = skip_tls.parent is connection_options

        assert skip_tls.value is True
        assert bypass_proxy.value is True
        assert stream.value is True
        assert vision.value is True
        assert model_options.border_title == "Model Options"
        assert connection_options.border_title == "Connection Options"
        assert http_options.border_title == "HTTP Options"
        assert extra_options.border_title == "Extra Options"
        assert api_key_in_model_options is True
        assert skip_tls_in_options is True
        assert sections_in_order == sorted(sections_in_order)
        assert streaming_in_model_options is True
        assert vision_in_model_options is True
        assert vision_is_last_in_model_options is True
        assert tls_hint.display is True

        skip_tls.value = False
        bypass_proxy.value = False
        stream.value = False
        vision.value = False
        await pilot.pause()
        saved = screen._build_profile_from_form()

    assert tls_hint.display is False
    assert saved.verify_ssl is True
    assert saved.bypass_proxy is False
    assert saved.stream is False
    assert saved.vision is False


@pytest.mark.asyncio
async def test_model_config_delete_last_key_value_row_recreates_blank_row() -> None:
    """Removing the last row should keep one blank editable row visible."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        headers_container = screen.query_one("#mc-headers-list")
        headers_container.query_one(".mc-kv-remove-btn", Button).press()
        await pilot.pause()

        rows = list(headers_container.query(".mc-kv-item-row"))

        assert len(rows) == 1
        assert rows[0].query_one(".mc-kv-key-input", Input).value == ""
        assert rows[0].query_one(".mc-kv-value-input", Input).value == ""


def _capture_notifications(screen: ModelConfigScreen) -> list[tuple[str, str]]:
    """Shadow ``screen.notify`` and record ``(severity, message)`` pairs."""
    captured: list[tuple[str, str]] = []

    def _notify(
        message: str,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float | None = None,
        markup: bool = True,
    ) -> None:
        captured.append((severity, message))

    screen.notify = _notify  # type: ignore[method-assign]
    return captured


@pytest.mark.asyncio
async def test_model_config_save_rejects_output_not_less_than_context(tmp_path: Path) -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#mc-max-tokens", Input).value = "32000"
        screen.query_one("#mc-max-output-tokens", Input).value = "32000"
        captured = _capture_notifications(screen)
        screen.query_one("#mc-save", Button).press()
        await pilot.pause()

    assert ("error", "Max output tokens must be less than max context tokens.") in captured
    assert not (tmp_path / "models" / "model-a.yaml").exists()
    assert registry.get("model-a") is profile


@pytest.mark.asyncio
async def test_model_config_save_warns_for_responses_store_true(tmp_path: Path) -> None:
    """Saving an OpenAI Responses profile with store=true surfaces the compaction warning."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        provider="openai",
        api_style="responses",
        model_id="gpt-test",
        chat_options=json.dumps({"store": True}),
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        captured = _capture_notifications(screen)
        screen.query_one("#mc-save", Button).press()
        await pilot.pause()

    assert ("information", "Model profile saved") in captured
    warnings = [message for severity, message in captured if severity == "warning"]
    assert len(warnings) == 1
    assert "compaction" in warnings[0]
    assert (tmp_path / "models" / "model-a.yaml").is_file()


@pytest.mark.asyncio
async def test_model_config_save_does_not_warn_without_store_true(tmp_path: Path) -> None:
    """A Responses profile without store=true saves with no warning toast."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        provider="openai",
        api_style="responses",
        model_id="gpt-test",
        chat_options=json.dumps({"store": False}),
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        captured = _capture_notifications(screen)
        screen.query_one("#mc-save", Button).press()
        await pilot.pause()

    assert ("information", "Model profile saved") in captured
    assert not any(severity == "warning" for severity, _message in captured)


@pytest.mark.asyncio
async def test_model_config_validate_rejects_wire_unsafe_charsets() -> None:
    """Non-ASCII in key/model/header fields is caught at save time, not first chat."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#mc-api-key", Input).value = "sk-abc▼def"
        screen.query_one("#mc-model", Input).value = "gpt▼4o"

        headers_container = screen.query_one("#mc-headers-list")
        header_row = headers_container.query_one(".mc-kv-item-row")
        header_row.query_one(".mc-kv-key-input", Input).value = "X 密"
        header_row.query_one(".mc-kv-value-input", Input).value = "secret▼"

        options_container = screen.query_one("#mc-options-list")
        option_row = options_container.query_one(".mc-kv-item-row")
        option_row.query_one(".mc-kv-key-input", Input).value = "extra_headers"
        option_row.query_one(".mc-kv-value-input", Input).value = '{"X-Custom": "秘密token"}'

        errors = screen._validate()

    joined = "\n".join(errors)
    # API key: position only, never the content.
    assert "API key contains a non-ASCII or control character at position 7" in joined
    assert "sk-abc" not in joined
    # Model ID: offending character is echoed.
    assert "Model ID contains" in joined
    assert "U+25BC" in joined
    # Extra header rows: name errors echo the name, value errors never
    # echo the value.
    assert "HTTP Extra Headers row 1: Header name" in joined
    assert "value contains a non-ASCII or control character" in joined
    assert "secret" not in joined
    # Chat-options extra_headers ride per-request headers and get the
    # same charset gate.
    assert "Chat Options row 1: 'extra_headers':" in joined
    assert "秘密" not in joined


@pytest.mark.asyncio
async def test_model_config_validation_notifications_disable_markup() -> None:
    """Invalid header names remain literal text in save error toasts."""
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry)
        await app.push_screen(screen)
        await pilot.pause()

        header_row = screen.query_one("#mc-headers-list .mc-kv-item-row")
        header_row.query_one(".mc-kv-key-input", Input).value = "[/]"
        header_row.query_one(".mc-kv-value-input", Input).value = "value"

        captured: list[tuple[str, bool]] = []

        def _notify(
            message: str,
            *,
            title: str = "",
            severity: str = "information",
            timeout: float | None = None,
            markup: bool = True,
        ) -> None:
            captured.append((message, markup))

        screen.notify = _notify  # type: ignore[method-assign]
        screen.query_one("#mc-save", Button).press()
        await pilot.pause()

    assert captured
    assert "Header name '[/]'" in captured[0][0]
    assert captured[0][1] is False


@pytest.mark.asyncio
async def test_model_config_validate_accepts_env_templates_and_printable_ascii() -> None:
    """Templates and permissive printable-ASCII values pass save-time validation."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="huggingface/WizardLM/WizardCoder-Python-34B-V1.0",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#mc-api-key", Input).value = "{{CHRYS_OPENAI_KEY}}"

        headers_container = screen.query_one("#mc-headers-list")
        header_row = headers_container.query_one(".mc-kv-item-row")
        header_row.query_one(".mc-kv-key-input", Input).value = "X-Api-Key"
        header_row.query_one(".mc-kv-value-input", Input).value = "Bearer {{CHRYS_HEADER_TOKEN}}"

        errors = screen._validate()

    assert errors == []


@pytest.mark.asyncio
async def test_model_config_validate_rejects_non_ascii_chat_option_model() -> None:
    """A chat-options model override rides the Chrys-Model-Id header; same charset gate."""

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
    )
    registry.register(profile)

    app = _ModelConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        options_container = screen.query_one("#mc-options-list")
        option_row = options_container.query_one(".mc-kv-item-row")
        option_row.query_one(".mc-kv-key-input", Input).value = "model"
        option_row.query_one(".mc-kv-value-input", Input).value = "模型"

        errors = screen._validate()

    joined = "\n".join(errors)
    assert "Chat Options row 1: 'model':" in joined
    assert "U+6A21" in joined
