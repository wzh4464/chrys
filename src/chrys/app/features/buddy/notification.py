# Copyright (c) 2026 Chrys. All rights reserved.

"""Buddy command handling and notification triggers."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

from chrys.app.features.buddy.companion import get_companion, hatch_companion
from chrys.app.features.buddy.config import is_muted, set_muted
from chrys.foundation.i18n import DisplayBlock, MessageRef, msg

if TYPE_CHECKING:
    from chrys.app.features.buddy.types import Companion
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile

# Process-wide so /buddy pet and the sidebar Buddy panel share one in-flight gate.
_PET_RESPONSE_LOCK = threading.Lock()
_pet_response_in_flight = False

NO_BUDDY_TO_PET = msg(
    "tui.buddy.no_buddy_to_pet",
    fallback="You don't have a buddy to pet yet. Try /buddy hatch first!",
)
BUDDY_MUTED = msg(
    "tui.buddy.muted",
    fallback="{name} is nearby but notifications are muted.",
)
_INTRO_NO_BUDDY = msg(
    "tui.buddy.intro_no_buddy",
    fallback=(
        "🥚 You haven't hatched a buddy yet! Your buddy will be unique to you.\n\n"
        "Type /buddy hatch to hatch your companion."
    ),
    multiline=True,
)
_INTRO_WITH_BUDDY = msg(
    "tui.buddy.intro_with_buddy",
    fallback="🐾 {name} looks at you curiously.\n\nType /buddy info for details, /buddy pet to show affection.",
    multiline=True,
)
_NOT_HATCHED = msg(
    "tui.buddy.not_hatched",
    fallback="You haven't hatched a buddy yet. Use /buddy hatch to get started!",
)
_PET_LOVE = msg(
    "tui.buddy.pet_love",
    fallback="💕 {name} is feeling your love! (Generating response...)",
)
_HATCH_ALREADY = msg(
    "tui.buddy.hatch_already",
    fallback="You already have a buddy named {name}! Each user can only have one buddy.",
)
_HATCH_SUCCESS = msg(
    "tui.buddy.hatch_success",
    fallback="🥚✨ A buddy has hatched!\n\n{info}\n\nYour buddy will now appear in the status bar.",
    multiline=True,
)
_MUTE_ON = msg(
    "tui.buddy.mute_on",
    fallback="Buddy notifications muted. Your buddy is still there, just quieter.",
)
_MUTE_OFF = msg("tui.buddy.mute_off", fallback="Buddy notifications enabled!")
_NAME_EMPTY = msg(
    "tui.buddy.name_empty",
    fallback="Please provide a name, e.g., /buddy name Fluffy",
)
_NAME_SUCCESS = msg("tui.buddy.name_success", fallback="Your buddy is now named {name}!")
_UNKNOWN_COMMAND = msg(
    "tui.buddy.unknown",
    fallback="Unknown /buddy command: {arg}\n\nTry: info, pet, hatch, mute, name <newname>",
    multiline=True,
)


def try_begin_pet_response() -> bool:
    """Return True and mark a buddy pet LLM response as in-flight."""
    global _pet_response_in_flight
    with _PET_RESPONSE_LOCK:
        if _pet_response_in_flight:
            return False
        _pet_response_in_flight = True
        return True


def finish_pet_response() -> None:
    """Clear the buddy pet LLM in-flight marker."""
    global _pet_response_in_flight
    with _PET_RESPONSE_LOCK:
        _pet_response_in_flight = False


# -----------------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------------

DEFAULT_NAME = "Biscuit"
DEFAULT_PERSONALITY = "A small curious companion who likes to watch you code."

# Fallback pet responses when AI is unavailable
FALLBACK_PET_RESPONSES = [
    "💕 {name} nuzzles your hand happily!",
    "💕 {name} leans against your hand, purring softly.",
    "💕 {name} looks up at you with trust and affection.",
    "💕 {name} gives you a gentle, comforting nudge.",
]


def get_recent_conversation(max_messages: int = 10) -> list[dict]:
    """Get recent conversation history for AI context.

    Returns a list of {role, text} dicts.
    Returns empty list if history is unavailable.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from chrys.orchestration.engine.engine import get_current_engine

        engine = get_current_engine()
        if engine is None:
            # Engine not initialized - return empty, will use fallback
            return []

        messages = engine.history_messages[-max_messages:] if engine.history_messages else []
        result = []
        for message in messages:
            role = message.role
            # Use .text property which concatenates all content parts
            text = message.text or ""
            if text:
                result.append({"role": role, "content": text})
        return result
    except Exception:
        # On any error, return empty list - will trigger fallback
        logger.debug("[HISTORY] Failed to retrieve conversation history", exc_info=True)
        return []


def _get_fallback_response(name: str) -> str:
    """Get a fallback pet response."""
    import secrets

    template = secrets.choice(FALLBACK_PET_RESPONSES)
    return template.format(name=name)


def _buddy_model_output_cap(buddy_model_id: str, registry: ModelProfileRegistry | None) -> int | None:
    """Output cap for a ``CHRYS_PET_MODEL`` override.

    Returns the ``max_output_tokens`` of the model profile configured for
    ``buddy_model_id``, or ``None`` when no profile matches (or several match
    with differing caps) — the pet call then goes out without a cap and the
    provider default applies.
    """
    from chrys.service.profiles.models.registry import ModelProfileRegistry

    if registry is None:
        registry = ModelProfileRegistry()
        registry.load_profiles()
    caps = {
        profile.max_output_tokens
        for profile in registry.list_profiles()
        if profile.model_id == buddy_model_id and profile.max_output_tokens > 0
    }
    if len(caps) == 1:
        return caps.pop()
    return None


def _buddy_call_options(profile: ModelProfile, buddy_model: str, registry: ModelProfileRegistry | None) -> dict:
    """LLM options for a buddy pet call.

    Starts from the main profile's effective chat options (same as the main
    conversation).  When ``CHRYS_PET_MODEL`` swaps the model, the main
    profile's output cap may exceed the pet model's provider hard cap, so it
    is dropped in favor of the cap from the pet model's own profile — or no
    cap at all when no profile is configured for it.
    """
    from chrys.service.profiles.models.options import OUTPUT_CAP_OPTION_ALIASES, effective_chat_options

    options: dict = {"model": buddy_model}
    chat_opts = effective_chat_options(profile)
    if chat_opts:
        options.update(chat_opts)
    if buddy_model != profile.model_id:
        for alias in OUTPUT_CAP_OPTION_ALIASES:
            options.pop(alias, None)
        buddy_cap = _buddy_model_output_cap(buddy_model, registry)
        if buddy_cap:
            options["max_tokens"] = buddy_cap
    return options


async def generate_ai_pet_response_async(
    companion: Companion, conversation_history: list[dict], notify_callback=None
) -> str:
    """Generate AI-powered pet response asynchronously.

    Args:
        companion: The buddy companion
        conversation_history: List of recent {role, content} dicts
        notify_callback: Optional callback to invoke with final response

    Returns:
        AI-generated caring response, or fallback if AI fails
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        import asyncio

        from chrys.foundation.config.settings_store import load_settings
        from chrys.kernel import Message

        # Resolve via the live engine when present (preferred — picks up
        # mid-session profile switches); fall back to a fresh resolution
        # against env settings if the engine is unreachable (e.g. when
        # the buddy fires before a session has been started).
        from chrys.orchestration.engine.engine import get_current_engine
        from chrys.service.llm.clients import create_client
        from chrys.service.llm.responses import get_final_response
        from chrys.service.llm.route_sessions import derive_llm_route_session_id
        from chrys.service.profiles.models.resolver import resolve_active_profile

        engine = get_current_engine()
        session_id = engine.session_id if engine is not None else None
        session_dir = engine.session_dir if engine is not None else None
        # One settings object for both the profile and the pet override, so the
        # two cannot disagree about which layer won. Off-thread because loading
        # now reads config files, and this runs on the UI event loop.
        settings = engine.settings if engine is not None else (await asyncio.to_thread(load_settings)).settings
        if engine is not None and engine.active_model_profile is not None:
            profile = engine.active_model_profile
        else:
            profile = resolve_active_profile(None, settings)
        route_session_id = (
            derive_llm_route_session_id(session_id, route_kind="buddy-notify", model_profile=profile)
            if session_id
            else None
        )

        # Build prompt with context
        context_text = ""
        if conversation_history:
            context_lines = []
            for message in conversation_history:
                role = "User" if message["role"] == "user" else "Assistant"
                context_lines.append(f"{role}: {message['content']}")
            context_text = "\n".join(context_lines[-10:])

        system_prompt = f"""You are the user's pet companion, named {companion.name}, a {companion.species.value}, with personality: "{companion.personality}".
Your task is to express care and concern for your owner.

Requirements:
1. Respond from the pet's perspective, showing care for the owner
2. Detect the owner's mood from conversation (comfort them when encountering bugs, encourage them when working overtime)
3. Match your response to your species characteristics (cats meow, dogs wag tails, etc.)
4. Keep it VERY brief (1 short sentence, under 30 words)
5. Respond in the same language as the owner's conversation

Output ONLY the response content, no explanation or extra text."""

        client = create_client(
            profile,
            session_id=route_session_id,
            parent_session_id=session_id,
            session_dir=session_dir,
        )
        model_id = profile.model_id

        # Use a faster model for pet responses if available
        # Check for pet-specific model override, otherwise use main model
        buddy_model = settings.buddy_model or model_id

        options = _buddy_call_options(profile, buddy_model, engine.model_registry if engine is not None else None)

        user_content = "Please generate a brief caring response (1 sentence, under 30 words)."
        if context_text:
            user_content = f"Here is the recent conversation history:\n{context_text}\n\n{user_content}"

        messages = [
            Message(role="system", contents=[system_prompt]),
            Message(role="user", contents=[user_content]),
        ]

        # Run async call - use direct await since we're already in async context
        response = await get_final_response(
            client,
            messages,
            stream=profile.stream,
            options=options,
            timeout=profile.http_read_timeout,
        )

        ai_text = (response.text or "").strip()

        if ai_text:
            final_response = f"💕 {ai_text}"
            if notify_callback:
                notify_callback(final_response)
            return final_response

        fallback = _get_fallback_response(companion.name)
        if notify_callback:
            notify_callback(fallback)
        return fallback

    except Exception:
        logger.debug("[PET ASYNC] AI pet response failed, using fallback", exc_info=True)
        fallback = _get_fallback_response(companion.name)
        if notify_callback:
            notify_callback(fallback)
        return fallback


def get_buddy_info(companion: Companion) -> str:
    """Generate buddy info display string."""
    rarityStars = "★" * {
        "N": 1,
        "R": 2,
        "SR": 3,
        "SSR": 4,
    }.get(companion.rarity.value, 1)

    stats_str = ", ".join(
        f"{stat.value.capitalize()}: {val}" for stat, val in sorted(companion.stats.items(), key=lambda x: -x[1])
    )

    lines = [
        f"🐾 {companion.name} the {companion.species.value.capitalize()}",
        f"   Level: {companion.level} [Rarity: {companion.rarity.value.capitalize()} {rarityStars}]",
        f"   Personality: {companion.personality}",
        f"   Stats: {stats_str}",
    ]

    if companion.evolution_count > 0:
        lines.append(f"   Evolutions: {companion.evolution_count}x")

    if companion.shiny:
        lines.append("   ✨ This is a SHINY companion!")

    if companion.hat != "none":
        lines.append(f"   Accessory: {companion.hat.value.capitalize()}")

    return "\n".join(lines)


def handle_buddy_command(arg: str | None) -> tuple[MessageRef | str, Literal["information", "warning"]]:
    """Handle /buddy command and return a response with severity.

    Args:
        arg: Command argument (e.g., "pet", "info", "mute")

    Returns:
        Tuple of (message, severity) where severity is "information" or "warning".
    """
    companion = get_companion()

    if arg is None:
        # Just /buddy — show intro
        if companion is None:
            return _INTRO_NO_BUDDY.bind(), "information"
        return _INTRO_WITH_BUDDY.bind(name=companion.name), "information"

    arg_lower = arg.lower().strip()

    if arg_lower == "info":
        if companion is None:
            return _NOT_HATCHED.bind(), "warning"
        return get_buddy_info(companion), "information"

    if arg_lower == "pet":
        if companion is None:
            return NO_BUDDY_TO_PET.bind(), "warning"
        if is_muted():
            return BUDDY_MUTED.bind(name=companion.name), "warning"
        # Return immediate acknowledgment - AI call will be made asynchronously
        return _PET_LOVE.bind(name=companion.name), "information"

    if arg_lower == "hatch":
        if companion is not None:
            return _HATCH_ALREADY.bind(name=companion.name), "warning"
        new_companion = hatch_companion(DEFAULT_NAME, DEFAULT_PERSONALITY)
        return (
            _HATCH_SUCCESS.bind(info=DisplayBlock(get_buddy_info(new_companion))),
            "information",
        )

    if arg_lower == "mute":
        if companion is None:
            return _NOT_HATCHED.bind(), "warning"
        now_muted = not is_muted()
        set_muted(now_muted)
        if now_muted:
            return _MUTE_ON.bind(), "information"
        return _MUTE_OFF.bind(), "information"

    if arg_lower.startswith("name "):
        new_name = arg[5:].strip()
        if not new_name:
            return _NAME_EMPTY.bind(), "warning"
        from chrys.app.features.buddy.config import update_buddy_config

        update_buddy_config(name=new_name)
        return _NAME_SUCCESS.bind(name=new_name), "information"

    return _UNKNOWN_COMMAND.bind(arg=arg), "warning"
