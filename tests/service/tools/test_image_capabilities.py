# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for per-agent image capability helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from chrys.foundation.text.images import is_image_media_type as common_is_image_media_type
from chrys.kernel import ChatContext, Content, Message
from chrys.kernel import is_image_media_type as kernel_is_image_media_type
from chrys.service.tools.builtins.filesystem import view_image
from chrys.service.vision import filter_image_tools, image_stub_middleware_for_model


def test_common_and_kernel_image_media_type_predicates_agree() -> None:
    media_types = [
        "image/png",
        "image/jpeg; charset=binary",
        " IMAGE/webp ",
        "text/plain",
        "application/json",
        "",
        None,
        123,
    ]

    assert [kernel_is_image_media_type(item) for item in media_types] == [
        common_is_image_media_type(item) for item in media_types
    ]


def test_filter_image_tools_removes_view_image_for_text_only_models() -> None:
    assert filter_image_tools([view_image], vision_enabled=False) == []
    assert filter_image_tools([view_image], vision_enabled=True) == [view_image]


async def test_non_vision_stub_replaces_tool_result_images_without_mutating_original() -> None:
    image = Content.from_data(
        b"image-bytes",
        "image/png",
        additional_properties={"width": 12, "height": 8, "media_type": "image/png"},
    )
    original_result = Content.from_function_result(
        "call_1",
        result=[Content.from_text("caption"), image],
    )
    original_message = Message("tool", [original_result], message_id="msg_1")
    context = ChatContext(client=MagicMock(), messages=[original_message], options={})
    middleware = image_stub_middleware_for_model(vision_enabled=False)
    assert middleware is not None

    async def _final() -> None:
        return None

    await middleware.process(context, _final)

    assert context.messages[0] is not original_message
    assert original_result.items == [Content.from_text("caption"), image]
    replacement_result = context.messages[0].contents[0]
    assert replacement_result is not original_result
    assert replacement_result.items is not None
    assert [item.type for item in replacement_result.items] == ["text", "text"]
    assert "12x8 image/png omitted" in (replacement_result.items[1].text or "")
    assert original_result.items[1] is image


async def test_non_vision_stub_replaces_direct_images_without_mutating_original() -> None:
    image = Content.from_data(
        b"image-bytes",
        "image/png",
        additional_properties={"width": 1, "height": 2, "media_type": "image/png"},
    )
    original_message = Message("user", ["describe", image], message_id="msg_1")
    context = ChatContext(client=MagicMock(), messages=[original_message], options={})
    middleware = image_stub_middleware_for_model(vision_enabled=False)
    assert middleware is not None

    async def _final() -> None:
        return None

    await middleware.process(context, _final)

    assert context.messages[0] is not original_message
    assert original_message.contents[1] is image
    assert context.messages[0].contents[1].type == "text"
    assert "1x2 image/png omitted" in (context.messages[0].contents[1].text or "")


async def test_non_vision_stub_ignores_unknown_uri_without_media_type() -> None:
    uri = Content.from_uri("https://example.com/blob")
    original_message = Message("user", [uri], message_id="msg_1")
    context = ChatContext(client=MagicMock(), messages=[original_message], options={})
    middleware = image_stub_middleware_for_model(vision_enabled=False)
    assert middleware is not None

    async def _final() -> None:
        return None

    await middleware.process(context, _final)

    assert context.messages[0] is original_message
    assert context.messages[0].contents[0] is uri
