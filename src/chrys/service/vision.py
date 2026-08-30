# Copyright (c) 2026 Chrys. All rights reserved.

"""Per-agent image capability wiring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from chrys.kernel import Content, FunctionTool, Message
from chrys.kernel import is_image_content as _is_image_content
from chrys.kernel.middleware import ChatContext, ChatMiddleware

VIEW_IMAGE_TOOL_NAME = "view_image"


def filter_image_tools(tools: Sequence[Any], *, vision_enabled: bool) -> list[Any]:
    """Return tools with image-reading tools removed for text-only agents."""
    if vision_enabled:
        return list(tools)
    return [item for item in tools if not (isinstance(item, FunctionTool) and item.name == VIEW_IMAGE_TOOL_NAME)]


def image_stub_middleware_for_model(*, vision_enabled: bool) -> ChatMiddleware | None:
    """Return middleware that hides image content from text-only models."""
    if vision_enabled:
        return None
    return NonVisionImageStubMiddleware()


class NonVisionImageStubMiddleware(ChatMiddleware):
    """Replace model-visible image content with text stubs for text-only models."""

    async def process(self, context: ChatContext, call_next: Any) -> None:
        replacement_messages: list[Message] = []
        changed = False
        for message in context.messages:
            replacement_contents: list[Content] = []
            message_changed = False
            for content in message.contents:
                replacement = _stub_content_images(content)
                replacement_contents.append(replacement)
                if replacement is not content:
                    message_changed = True
            if message_changed:
                replacement_messages.append(
                    Message(
                        message.role,
                        replacement_contents,
                        author_name=message.author_name,
                        message_id=message.message_id,
                        additional_properties=message.additional_properties,
                        raw_representation=message.raw_representation,
                    )
                )
                changed = True
            else:
                replacement_messages.append(message)
        if changed:
            context.messages = replacement_messages
        await call_next()


def _stub_content_images(content: Content) -> Content:
    if content.type == "function_result" and content.items:
        replacement_items: list[Content] = []
        changed = False
        for item in content.items:
            replacement = _stub_direct_image(item)
            replacement_items.append(replacement)
            if replacement is not item:
                changed = True
        if not changed:
            return content
        return Content.from_function_result(
            content.call_id or "",
            result=replacement_items,
            exception=content.exception,
            annotations=content.annotations,
            additional_properties=content.additional_properties,
            raw_representation=content.raw_representation,
        )
    return _stub_direct_image(content)


def _stub_direct_image(content: Content) -> Content:
    if not _is_image_content(content):
        return content
    return Content.from_text(_image_stub_text(content), annotations=content.annotations)


def _image_stub_text(content: Content) -> str:
    props = content.additional_properties
    width = props.get("width")
    height = props.get("height")
    media_type = content.media_type or props.get("media_type") or "image"
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return f"[image {width}x{height} {media_type} omitted: model is not vision-capable]"
    return f"[image {media_type} omitted: model is not vision-capable]"
