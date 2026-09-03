# Copyright (c) 2026 Chrys. All rights reserved.

"""Settings panel layout: which persisted settings render where, and how.

The panel is a renderer over :class:`~chrys.foundation.config.spec.SettingSpec`
metadata; this module is the single place that decides which tab and section a
key lives in, which control variant it gets, and what hint sits under it. Keys
that are persisted but deliberately not rendered yet are listed in
:data:`DEFERRED_KEYS` so an architecture test can prove nothing was forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from chrys.foundation.i18n import MessageDef, msg

# ── Tab titles ─────────────────────────────────────────────────────────
_TAB_GENERAL = msg("tui.settings.tab.general", fallback="General")
_TAB_MODELS_AGENTS = msg("tui.settings.tab.models_agents", fallback="Models & Agents")
_TAB_SECURITY = msg("tui.settings.tab.security", fallback="Security")
_TAB_SESSIONS = msg("tui.settings.tab.sessions", fallback="Sessions")
_TAB_TOOLS = msg("tui.settings.tab.tools", fallback="Tools")
_TAB_NOTIFICATIONS = msg("tui.settings.tab.notifications", fallback="Notifications")

# ── Section titles ─────────────────────────────────────────────────────
_SECTION_APPEARANCE = msg("tui.settings.section.appearance", fallback="Appearance")
_SECTION_INPUT = msg("tui.settings.section.input", fallback="Input")
_SECTION_SESSION_TITLES = msg("tui.settings.section.session_titles", fallback="Titles")
_SECTION_AGENT = msg("tui.settings.section.agent", fallback="Agent")
_SECTION_MODEL_ROLES = msg("tui.settings.section.model_roles", fallback="Model roles")
_SECTION_LLM = msg("tui.settings.section.llm", fallback="LLM")
_SECTION_APPROVAL = msg("tui.settings.section.approval", fallback="Approval")
_SECTION_PROJECT_TRUST = msg("tui.settings.section.project_trust", fallback="Project trust")
_SECTION_DIAGNOSTICS = msg("tui.settings.section.diagnostics", fallback="Diagnostics")
_SECTION_TELEMETRY = msg("tui.settings.section.telemetry", fallback="Telemetry")
_SECTION_LOCATION = msg("tui.settings.section.location", fallback="Location")
_SECTION_ROLLBACK = msg("tui.settings.section.rollback", fallback="Rollback")
_SECTION_TOOLS = msg("tui.settings.section.tools", fallback="Tools")
_SECTION_WORKSPACE_NOTICE = msg("tui.settings.section.workspace_notice", fallback="Workspace notice")
_SECTION_NOTIFICATIONS = msg("tui.settings.section.notifications", fallback="Notifications")

# ── Hints (short, one line; defaults spelled out because there is no reset) ──
_HINT_UI_THEME = msg("tui.settings.hint.ui.theme", fallback="Preview themes with F9.")
_HINT_UI_LOCALE = msg(
    "tui.settings.hint.ui.locale",
    fallback="Language of menus, hints and messages; changes apply immediately.",
)
_HINT_HISTORY_PROMPT_ENABLED = msg(
    "tui.settings.hint.history.prompt.enabled",
    fallback="Remember what you type so you can bring it back later with Ctrl+R.",
)
_HINT_SESSION_TITLE_AUTO = msg(
    "tui.settings.hint.session.title.auto",
    fallback="Update the session title after each successful turn.",
)
_HINT_AGENT_DEFAULT_PROFILE = msg(
    "tui.settings.hint.agent.default_profile",
    fallback="Agent started by default at the next launch.",
)
_HINT_MODEL_ROLE_SESSION_TITLE = msg(
    "tui.settings.hint.model.role.session_title",
    fallback="Model that writes session titles.",
)
_HINT_MODEL_ROLE_APPROVAL_JUDGE = msg(
    "tui.settings.hint.model.role.approval_judge",
    fallback="Model that judges tool approvals in auto mode.",
)
_HINT_MODEL_ROLE_BUDDY_MODEL_ID = msg(
    "tui.settings.hint.model.role.buddy_model_id",
    fallback="Model behind the buddy companion.",
)
_HINT_LLM_RETRY_MAX_TRANSIENT = msg(
    "tui.settings.hint.llm.retry.max_transient",
    fallback="Blank = the frontend default ({default}); 0 = never retry; at most {maximum}.",
)
_HINT_APPROVAL_DEFAULT_MODE = msg(
    "tui.settings.hint.approval.default_mode",
    fallback=(
        "Approval mode at the next launch; /approval changes both the current session and this default. "
        "bypass is only available for one launch via --approval bypass."
    ),
)
_HINT_PROJECT_CONFIG_ENABLED = msg(
    "tui.settings.hint.project.config_enabled",
    fallback="Let <workspace>/.chrys/settings.yaml adjust engineering settings for that workspace.",
)
_HINT_PROJECT_HOOKS_ENABLED = msg(
    "tui.settings.hint.project.hooks_enabled",
    fallback="Run the hooks defined in <workspace>/.chrys/hooks alongside your global hooks.",
)
HINT_PROJECT_CONFIG_DORMANT = msg(
    "tui.settings.hint.project.config_dormant",
    fallback="Project settings found ({count} key) — enable to apply them.",
    plural_fallback="Project settings found ({count} keys) — enable to apply them.",
)
_HINT_LOG_RAW_HTTP_CAPTURE = msg(
    "tui.settings.hint.log.raw_http_capture",
    fallback="Writes API keys and full prompts in clear text to <session>/llm_raw_http.jsonl.",
)
_HINT_OTEL_ENABLED = msg(
    "tui.settings.hint.otel.enabled",
    fallback="Record traces and logs of what the agent does.",
)
_HINT_OTEL_ENDPOINT = msg(
    "tui.settings.hint.otel.endpoint",
    fallback="Send telemetry to a collector, e.g. http://localhost:4317. Blank = keep it in the session folder.",
)
_HINT_OTEL_SENSITIVE_DATA = msg(
    "tui.settings.hint.otel.sensitive_data",
    fallback="Include prompts and tool payloads in telemetry.",
)
_HINT_STORAGE_SESSION_ROOT_DIR = msg(
    "tui.settings.hint.storage.session_root_dir",
    fallback="Blank = {default}. Sessions are stored under <root>/sessions.",
)
_HINT_ROLLBACK_SNAPSHOTS_KEEP = msg(
    "tui.settings.hint.rollback.snapshots_keep",
    fallback="Per-turn snapshots kept for /rollback; default {default}.",
)
_HINT_TOOLS_ASK_USER_INLINE = msg(
    "tui.settings.hint.tools.ask_user.inline",
    fallback="Answer the agent's questions inside the conversation instead of a popup.",
)
_HINT_TOOLS_ASK_USER_TIMEOUT_SECONDS = msg(
    "tui.settings.hint.tools.ask_user.timeout_seconds",
    fallback="Blank = default {default} seconds; 0 = no timeout.",
)
_HINT_WORKSPACE_CHANGE_NOTICE_ENABLED = msg(
    "tui.settings.hint.workspace.change_notice.enabled",
    fallback="Tell the agent about files changed outside the conversation at turn boundaries.",
)


class Suggestions(Enum):
    """App-layer suggestion lists for open ``TEXT`` keys (not a closed choice set)."""

    AGENT_PROFILES = auto()
    MODEL_PROFILES = auto()
    """``(profile id, display name)`` — for keys that store a profile id."""
    MODEL_IDS = auto()
    """``(model id, model id)`` across the registered profiles — for keys that store a bare model id."""


class RowKind(Enum):
    """How a row is rendered beyond what ``Kind`` alone decides."""

    PLAIN = auto()
    SESSION_ROOT = auto()
    """The one path row: input + folder browser + migration entry point."""


PROJECT_CONFIG_KEY = "project.config_enabled"
"""The row whose hint swaps to :data:`HINT_PROJECT_CONFIG_DORMANT` while it is off and a project file waits."""


class HintArgs(Enum):
    """Which arguments a hint message binds (resolved by the row at render time)."""

    NONE = auto()
    FIELD_DEFAULT = auto()
    """``default`` = the field's built-in default."""
    RETRY_DEFAULTS = auto()
    """``default`` = the TUI's frontend retry default, ``maximum`` = the cap."""
    SESSION_ROOT_DEFAULT = auto()
    """``default`` = the default session root directory."""


class Placeholder(Enum):
    """What an empty text field shows in grey (resolved by the row at projection time)."""

    NONE = auto()
    RETRY_DEFAULT = auto()
    """The TUI's frontend retry default (what a blank retry cap falls back to)."""


@dataclass(frozen=True, slots=True)
class SettingRowSpec:
    """One rendered setting."""

    key: str
    hint: MessageDef | None = None
    hint_args: HintArgs = HintArgs.NONE
    placeholder: Placeholder = Placeholder.NONE
    suggestions: Suggestions | None = None
    special: RowKind = RowKind.PLAIN


@dataclass(frozen=True, slots=True)
class SettingsSection:
    title: MessageDef
    rows: tuple[SettingRowSpec, ...]


@dataclass(frozen=True, slots=True)
class SettingsTab:
    id: str
    """Also the argument accepted by ``/settings <tab>``."""
    title: MessageDef
    sections: tuple[SettingsSection, ...]
    custom_pane: bool = False
    """Rendered by a dedicated pane rather than the generic row renderer."""


GENERAL_TAB_ID = "general"
NOTIFICATIONS_TAB_ID = "notifications"
SESSIONS_TAB_ID = "sessions"

TABS: tuple[SettingsTab, ...] = (
    SettingsTab(
        id=GENERAL_TAB_ID,
        title=_TAB_GENERAL,
        sections=(
            SettingsSection(
                _SECTION_APPEARANCE,
                (
                    SettingRowSpec("ui.theme", hint=_HINT_UI_THEME),
                    SettingRowSpec("ui.locale", hint=_HINT_UI_LOCALE),
                ),
            ),
            SettingsSection(
                _SECTION_INPUT,
                (SettingRowSpec("history.prompt.enabled", hint=_HINT_HISTORY_PROMPT_ENABLED),),
            ),
        ),
    ),
    SettingsTab(
        id="models",
        title=_TAB_MODELS_AGENTS,
        sections=(
            SettingsSection(
                _SECTION_AGENT,
                (
                    SettingRowSpec(
                        "agent.default_profile",
                        hint=_HINT_AGENT_DEFAULT_PROFILE,
                        suggestions=Suggestions.AGENT_PROFILES,
                    ),
                ),
            ),
            SettingsSection(
                _SECTION_MODEL_ROLES,
                (
                    SettingRowSpec(
                        "model.role.session_title",
                        hint=_HINT_MODEL_ROLE_SESSION_TITLE,
                        suggestions=Suggestions.MODEL_PROFILES,
                    ),
                    SettingRowSpec(
                        "model.role.approval_judge",
                        hint=_HINT_MODEL_ROLE_APPROVAL_JUDGE,
                        suggestions=Suggestions.MODEL_PROFILES,
                    ),
                    SettingRowSpec(
                        "model.role.buddy_model_id",
                        hint=_HINT_MODEL_ROLE_BUDDY_MODEL_ID,
                        suggestions=Suggestions.MODEL_IDS,
                    ),
                ),
            ),
            SettingsSection(
                _SECTION_LLM,
                (
                    SettingRowSpec(
                        "llm.retry.max_transient",
                        hint=_HINT_LLM_RETRY_MAX_TRANSIENT,
                        hint_args=HintArgs.RETRY_DEFAULTS,
                        placeholder=Placeholder.RETRY_DEFAULT,
                    ),
                ),
            ),
        ),
    ),
    SettingsTab(
        id="security",
        title=_TAB_SECURITY,
        sections=(
            SettingsSection(
                _SECTION_APPROVAL,
                (SettingRowSpec("approval.default_mode", hint=_HINT_APPROVAL_DEFAULT_MODE),),
            ),
            SettingsSection(
                _SECTION_PROJECT_TRUST,
                (
                    SettingRowSpec(PROJECT_CONFIG_KEY, hint=_HINT_PROJECT_CONFIG_ENABLED),
                    SettingRowSpec("project.hooks_enabled", hint=_HINT_PROJECT_HOOKS_ENABLED),
                ),
            ),
            SettingsSection(
                _SECTION_DIAGNOSTICS,
                (SettingRowSpec("log.raw_http_capture", hint=_HINT_LOG_RAW_HTTP_CAPTURE),),
            ),
            SettingsSection(
                _SECTION_TELEMETRY,
                (
                    SettingRowSpec("otel.enabled", hint=_HINT_OTEL_ENABLED),
                    SettingRowSpec("otel.endpoint", hint=_HINT_OTEL_ENDPOINT),
                    SettingRowSpec("otel.sensitive_data", hint=_HINT_OTEL_SENSITIVE_DATA),
                ),
            ),
        ),
    ),
    SettingsTab(
        id=SESSIONS_TAB_ID,
        title=_TAB_SESSIONS,
        sections=(
            SettingsSection(
                _SECTION_SESSION_TITLES,
                (SettingRowSpec("session.title.auto", hint=_HINT_SESSION_TITLE_AUTO),),
            ),
            SettingsSection(
                _SECTION_LOCATION,
                (
                    SettingRowSpec(
                        "storage.session_root_dir",
                        hint=_HINT_STORAGE_SESSION_ROOT_DIR,
                        hint_args=HintArgs.SESSION_ROOT_DEFAULT,
                        special=RowKind.SESSION_ROOT,
                    ),
                ),
            ),
            SettingsSection(
                _SECTION_ROLLBACK,
                (
                    SettingRowSpec(
                        "rollback.snapshots_keep",
                        hint=_HINT_ROLLBACK_SNAPSHOTS_KEEP,
                        hint_args=HintArgs.FIELD_DEFAULT,
                    ),
                ),
            ),
        ),
    ),
    SettingsTab(
        id="tools",
        title=_TAB_TOOLS,
        sections=(
            SettingsSection(
                _SECTION_TOOLS,
                (
                    SettingRowSpec("tools.ask_user.inline", hint=_HINT_TOOLS_ASK_USER_INLINE),
                    SettingRowSpec(
                        "tools.ask_user.timeout_seconds",
                        hint=_HINT_TOOLS_ASK_USER_TIMEOUT_SECONDS,
                        hint_args=HintArgs.FIELD_DEFAULT,
                    ),
                ),
            ),
            SettingsSection(
                _SECTION_WORKSPACE_NOTICE,
                (SettingRowSpec("workspace.change_notice.enabled", hint=_HINT_WORKSPACE_CHANGE_NOTICE_ENABLED),),
            ),
        ),
    ),
    SettingsTab(
        id=NOTIFICATIONS_TAB_ID,
        title=_TAB_NOTIFICATIONS,
        # Rendered by NotificationsPane; the rows are listed so the coverage
        # invariant (rendered + deferred == every persisted key) stays exact.
        sections=(
            SettingsSection(
                _SECTION_NOTIFICATIONS,
                (
                    SettingRowSpec("notifications.enabled"),
                    SettingRowSpec("notifications.delivery.desktop"),
                    SettingRowSpec("notifications.delivery.sound"),
                    SettingRowSpec("notifications.suppress_when_focused"),
                    SettingRowSpec("notifications.events.approval_required"),
                    SettingRowSpec("notifications.events.ask_user"),
                    SettingRowSpec("notifications.events.turn_complete"),
                    SettingRowSpec("notifications.events.turn_error"),
                ),
            ),
        ),
        custom_pane=True,
    ),
)

DEFERRED_KEYS: frozenset[str] = frozenset(
    {
        # Advanced tab candidates (not built yet).
        "ui.chat.file_snapshot_inline_chars",
        "mutations.trace.mode",
        "mutations.trace.fsatrace_path",
        # Owned by a dedicated editor (Ctrl+O keymap picker, F4 model picker).
        "ui.editor.keymap",
        "model.profile.active",
        # Held back from the panel for now.
        "tools.result.ceiling_tokens",
        "app.dev_mode",
        "context.warn_threshold_pct",
        # Keep the backend/config surface while its product UX is undecided.
        "trajectory.verify_commands",
        "memory.mcp.enabled",
        "pact.verify_command",
        "routing.mode",
        "routing.tiebreaker_model_profile",
        "semantic_search.model_profile",
        "memory.writeback.idle_seconds",
        "memory.writeback.on_session_end",
        # Low-frequency keys with a natural home, held back until asked for.
        "workspace.mru_max_entries",
        "mutations.snapshot.max_file_mb",
        "mutations.snapshot.skip_binary",
        "mutations.coordination.enabled",
        "mutations.parallel_implicit_tools",
        "workspace.change_notice.max_entries",
    }
)

TAB_IDS: tuple[str, ...] = tuple(tab.id for tab in TABS)


def rendered_row_specs() -> tuple[SettingRowSpec, ...]:
    """Every row across every tab, in layout order."""
    return tuple(row for tab in TABS for section in tab.sections for row in section.rows)


def rendered_keys() -> frozenset[str]:
    return frozenset(row.key for row in rendered_row_specs())


def tab_by_id(tab_id: str) -> SettingsTab | None:
    for tab in TABS:
        if tab.id == tab_id:
            return tab
    return None
