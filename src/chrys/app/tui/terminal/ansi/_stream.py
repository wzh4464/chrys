# Copyright (c) 2026 Chrys. All rights reserved.
# Adapted from toad (https://github.com/batrachianai/toad).

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Iterable
from functools import lru_cache
from typing import Literal

from textual.color import Color
from textual.style import NULL_STYLE, Style

from chrys.app.tui.terminal.ansi._ansi_colors import ANSI_COLORS
from chrys.app.tui.terminal.ansi._commands import (
    DEC,
    DEC_SLOTS,
    MOUSE_FORMAT,
    MOUSE_TRACKING_MODES,
    ANSIBell,
    ANSICharacterSet,
    ANSIClear,
    ANSICommand,
    ANSIContent,
    ANSICursor,
    ANSICursorPositionRequest,
    ANSICursorRestore,
    ANSICursorSave,
    ANSIDeviceAttributesRequest,
    ANSIFeatures,
    ANSIHorizontalTab,
    ANSIModeStatusRequest,
    ANSIMouseTracking,
    ANSINewLine,
    ANSIScroll,
    ANSIScrollMargin,
    ANSIShellCommand,
    ANSIStyle,
    ANSIWorkingDirectory,
    DECInvoke,
)
from chrys.app.tui.terminal.ansi._control_codes import CONTROL_CODES
from chrys.app.tui.terminal.ansi._parser import ANSIParser
from chrys.app.tui.terminal.ansi._sgr_styles import SGR_STYLES
from chrys.app.tui.terminal.ansi._stream_parser import Token


class ANSIStream:
    def __init__(self) -> None:
        self.parser = ANSIParser()
        self.style = NULL_STYLE
        self.show_cursor = True
        self.last_command: str = ""

    @classmethod
    @lru_cache(maxsize=1024)
    def _parse_sgr(cls, sgr: str) -> Style | None:
        """Parse a SGR (Select Graphics Rendition) code in to a Style instance,
        or `None` to indicate a reset.

        Args:
            sgr: SGR sequence.

        Returns:
            A Visual Style, or `None`.
        """
        codes = [code if code < 255 else 255 for code in map(int, [sgr_code or "0" for sgr_code in sgr.split(";")])]
        style = NULL_STYLE
        while codes:
            match codes:
                case [38, 2, red, green, blue, *codes]:
                    # Foreground RGB
                    style += Style(foreground=Color(red, green, blue))
                case [48, 2, red, green, blue, *codes]:
                    # Background RGB
                    style += Style(background=Color(red, green, blue))
                case [38, 5, ansi_color, *codes]:
                    # Foreground ANSI
                    style += Style(foreground=ANSI_COLORS[ansi_color])
                case [48, 5, ansi_color, *codes]:
                    # Background ANSI
                    style += Style(background=ANSI_COLORS[ansi_color])
                case [0, *codes]:
                    # reset
                    return None
                case [code, *codes]:
                    if sgr_style := SGR_STYLES.get(code):
                        style += sgr_style

        return style

    def feed(self, text: str) -> Iterable[ANSICommand]:
        """Feed text potentially containing ANSI sequences, and parse in to
        an iterable of ansi commands.

        Args:
            text: Text to feed.

        Yields:
            `ANSICommand` instances.
        """

        for token in self.parser.feed(text):
            if not isinstance(token, Token):
                yield from self.on_token(token)

    C0_CURSOR_COMMANDS = {
        "\r": ANSICursor(absolute_x=0),
        "\x08": ANSICursor(delta_x=-1),
    }
    CLEAR_LINE_CURSOR_TO_END = ANSICursor(clear_range=(None, -1), erase=True, update_background=True)
    CLEAR_LINE_CURSOR_TO_BEGINNING = ANSICursor(clear_range=(0, None), erase=True, update_background=True)
    CLEAR_LINE = ANSICursor(clear_range=(0, -1), erase=True, update_background=True)
    CLEAR_SCREEN_CURSOR_TO_END = ANSIClear("cursor_to_end")
    CLEAR_SCREEN_CURSOR_TO_BEGINNING = ANSIClear("cursor_to_beginning")
    CLEAR_SCREEN = ANSIClear("screen")
    CLEAR_SCREEN_SCROLLBACK = ANSIClear("scrollback")
    SHOW_CURSOR = ANSIFeatures(show_cursor=True)
    HIDE_CURSOR = ANSIFeatures(show_cursor=False)
    ENABLE_ALTERNATE_SCREEN = ANSIFeatures(alternate_screen=True)
    DISABLE_ALTERNATE_SCREEN = ANSIFeatures(alternate_screen=False)
    ENABLE_BRACKETED_PASTE = ANSIFeatures(bracketed_paste=True)
    DISABLE_BRACKETED_PASTE = ANSIFeatures(bracketed_paste=False)
    ENABLE_CURSOR_BLINK = ANSIFeatures(cursor_blink=True)
    DISABLE_CURSOR_BLINK = ANSIFeatures(cursor_blink=False)
    ENABLE_CURSOR_KEYS_APPLICATION_MODE = ANSIFeatures(cursor_keys=True)
    DISABLE_CURSOR_KEYS_APPLICATION_MODE = ANSIFeatures(cursor_keys=False)
    ENABLE_REPLACE_MODE = ANSIFeatures(replace_mode=True)
    DISABLE_REPLACE_MODE = ANSIFeatures(replace_mode=False)
    ENABLE_AUTO_WRAP = ANSIFeatures(auto_wrap=True)
    DISABLE_AUTO_WRAP = ANSIFeatures(auto_wrap=False)

    INVOKE_G0_INTO_GL = DECInvoke(gl=0)
    INVOKE_G1_INTO_GL = DECInvoke(gl=1)
    INVOKE_G2_INTO_GL = DECInvoke(gl=2)
    INVOKE_G3_INTO_GL = DECInvoke(gl=3)
    INVOKE_G1_INTO_GR = DECInvoke(gr=1)
    INVOKE_G2_INTO_GR = DECInvoke(gr=2)
    INVOKE_G3_INTO_GR = DECInvoke(gr=3)
    SHIFT_G2 = DECInvoke(shift=2)
    SHIFT_G3 = DECInvoke(shift=3)

    DEC_INVOKE_MAP = {
        "n": INVOKE_G2_INTO_GL,
        "o": INVOKE_G3_INTO_GL,
        "~": INVOKE_G1_INTO_GR,
        "}": INVOKE_G2_INTO_GR,
        "|": INVOKE_G3_INTO_GR,
        "N": SHIFT_G2,
        "O": SHIFT_G3,
    }

    @classmethod
    @lru_cache(maxsize=1024)
    def _parse_csi(cls, csi: str) -> ANSICommand | None:
        """Parse CSI sequence in to an ansi segment.

        Args:
            csi: CSI sequence.

        Returns:
            Ansi segment, or `None` if one couldn't be decoded.
        """

        if match := re.fullmatch(r"\[(\d+)?(?:;)?(\d*)?(\w)", csi):
            match_groups = match.groups(default="")
            match match_groups:
                case [lines, _, "A"]:
                    # CUU - Cursor Up: ESC[nA
                    return ANSICursor(delta_y=-int(lines or 1))
                case [lines, _, "B"]:
                    # CUD - Cursor Down: ESC[nB
                    return ANSICursor(delta_y=+int(lines or 1))
                case [cells, _, "C"]:
                    # CUF - Cursor Forward: ESC[nC
                    return ANSICursor(delta_x=+int(cells or 1))
                case [cells, _, "D"]:
                    # CUB - Cursor Back: ESC[nD
                    return ANSICursor(delta_x=-int(cells or 1))
                case [lines, _, "E"]:
                    # CNL - Cursor Next Line: ESC[nE
                    return ANSICursor(absolute_x=0, delta_y=+int(lines or 1))
                case [lines, _, "F"]:
                    # CPL - Cursor Previous Line: ESC[nF
                    return ANSICursor(absolute_x=0, delta_y=-int(lines or 1))
                case [cells, _, "G"]:
                    # CHA - Cursor Horizontal Absolute: ESC[nG
                    return ANSICursor(absolute_x=+int(cells or 1) - 1)
                case [row, column, "H" | "f"]:
                    # CUP - Cursor Position: ESC[n;mH
                    # HVP - Horizontal Vertical Position: ESC[n;mf
                    return ANSICursor(
                        absolute_x=int(column or 1) - 1,
                        absolute_y=int(row or 1) - 1,
                    )
                case [characters, _, "P"]:
                    return ANSICursor(
                        clear_range=(0, int(characters or 1) - 1),
                        relative=True,
                        erase=True,
                    )
                case [lines, _, "S"]:
                    return ANSIScroll(-1, int(lines or 1))
                case [lines, _, "T"]:
                    return ANSIScroll(+1, int(lines or 1))
                case [row, _, "d"]:
                    # VPA - Vertical Position Absolute: ESC[nd
                    return ANSICursor(absolute_y=int(row or 1) - 1)
                case [characters, _, "X"]:
                    return ANSICursor(
                        clear_range=(0, int(characters or 1) - 1),
                        relative=True,
                        erase=False,
                    )
                case ["0" | "", _, "J"]:
                    return cls.CLEAR_SCREEN_CURSOR_TO_END
                case ["1", _, "J"]:
                    return cls.CLEAR_SCREEN_CURSOR_TO_BEGINNING
                case ["2", _, "J"]:
                    return cls.CLEAR_SCREEN
                case ["3", _, "J"]:
                    return cls.CLEAR_SCREEN_SCROLLBACK
                case ["0" | "", _, "K"]:
                    return cls.CLEAR_LINE_CURSOR_TO_END
                case ["1", _, "K"]:
                    return cls.CLEAR_LINE_CURSOR_TO_BEGINNING
                case ["2", _, "K"]:
                    return cls.CLEAR_LINE
                case [top, bottom, "r"]:
                    return ANSIScrollMargin(
                        int(top or "1") - 1 if top else None,
                        int(bottom or "1") - 1 if top else None,
                    )
                case ["4", _, "h" | "l" as replace_mode]:
                    return cls.DISABLE_REPLACE_MODE if replace_mode == "h" else cls.ENABLE_REPLACE_MODE
                case ["", _, "s"]:
                    return ANSICursorSave()
                case ["", _, "u"]:
                    return ANSICursorRestore()

                case ["6", _, "n"]:
                    return ANSICursorPositionRequest()

                # Primary Device Attributes: ESC[c or ESC[0c
                case ["" | "0", _, "c"]:
                    return ANSIDeviceAttributesRequest(secondary=False)

                case _:
                    return None

        elif match := re.fullmatch(r"\[([0-9:;<=>?]*)([!-/]*)([@-~])", csi):
            match match.groups(default=""):
                case ["?25", "", "h"]:
                    return cls.SHOW_CURSOR
                case ["?25", "", "l"]:
                    return cls.HIDE_CURSOR
                case ["?1049", "", "h"]:
                    return cls.ENABLE_ALTERNATE_SCREEN
                case ["?1049", "", "l"]:
                    return cls.DISABLE_ALTERNATE_SCREEN
                case ["?2004", "", "h"]:
                    return cls.ENABLE_BRACKETED_PASTE
                case ["?2004", "", "l"]:
                    return cls.DISABLE_BRACKETED_PASTE
                case ["?12", "", "h"]:
                    return cls.ENABLE_CURSOR_BLINK
                case ["?12", "", "l"]:
                    return cls.DISABLE_CURSOR_BLINK
                case ["?1", "", "h"]:
                    return cls.ENABLE_CURSOR_KEYS_APPLICATION_MODE
                case ["?1", "", "l"]:
                    return cls.DISABLE_CURSOR_KEYS_APPLICATION_MODE
                case ["?7", "", "h"]:
                    return cls.ENABLE_AUTO_WRAP
                case ["?7", "", "l"]:
                    return cls.DISABLE_AUTO_WRAP

                # Older alternate screen modes (used by some vim builds)
                case ["?47" | "?1047", "", "h"]:
                    return cls.ENABLE_ALTERNATE_SCREEN
                case ["?47" | "?1047", "", "l"]:
                    return cls.DISABLE_ALTERNATE_SCREEN

                # Secondary Device Attributes: ESC[>c or ESC[>0c
                case [">" | ">0", "", "c"]:
                    return ANSIDeviceAttributesRequest(secondary=True)

                # DECRQM — Request Mode Status: ESC[?<mode>$p
                case [param, "$", "p"] if param.startswith("?") and param[1:].isdigit():
                    return ANSIModeStatusRequest(mode=int(param[1:]), private=True)

                # \x1b[22;0;0t
                case [_, _, "t"]:
                    # XTWINOPS (Window manipulation) — not implemented
                    return None
                case _:
                    if match := re.fullmatch(r"\[\?([0-9;]+)([hl])", csi):
                        modes = [m for m in match.group(1).split(";")]
                        enable = match.group(2) == "h"
                        tracking: Literal["none"] | MOUSE_TRACKING_MODES | None = None
                        format: MOUSE_FORMAT | None = None
                        focus_events: bool | None = None
                        alternate_scroll: bool | None = None
                        for mode in modes:
                            if mode == "1000":
                                tracking = "button" if enable else "none"
                            elif mode == "1002":
                                tracking = "drag" if enable else "none"
                            elif mode == "1003":
                                tracking = "all" if enable else "none"
                            elif mode == "1006":
                                format = "sgr"
                            elif mode == "1015":
                                format = "urxvt"
                            elif mode == "1004":
                                focus_events = enable
                            elif mode == "1007":
                                alternate_scroll = enable
                        return ANSIMouseTracking(
                            mode=tracking,
                            format=format,
                            focus_events=focus_events,
                            alternate_scroll=alternate_scroll,
                        )
                    return None

        return None

    @staticmethod
    def _strip_osc_terminator(payload: str) -> str:
        for terminator in ("\x1b\\", "\x9c", "\x07"):
            if payload.endswith(terminator):
                return payload[: -len(terminator)]
        return payload

    @classmethod
    def _decode_chrys_osc_payload(cls, payload: str) -> str:
        payload = cls._strip_osc_terminator(payload)
        if not payload.startswith("b64:"):
            return payload
        try:
            return base64.b64decode(payload[4:], validate=True).decode("utf-8")
        except binascii.Error, UnicodeDecodeError:
            return ""

    def on_token(self, token: tuple[str, str]) -> Iterable[ANSICommand]:
        match token:
            case ["c0", "\x07"]:
                yield ANSIBell()

            case ["c0", "\x08" | "\r" as control]:
                yield self.C0_CURSOR_COMMANDS[control]

            case ["c0", "\t"]:
                yield ANSIHorizontalTab()

            case ["c0", "\n" | "\x0b" | "\x0c"]:
                yield ANSINewLine()

            case ["c0", "\x0e"]:
                yield ANSICharacterSet(dec_invoke=self.INVOKE_G1_INTO_GL)

            case ["c0", "\x0f"]:
                yield ANSICharacterSet(dec_invoke=self.INVOKE_G0_INTO_GL)

            case ["c0", _]:
                # Unimplemented C0 controls and DEL have no printable form.
                # Consume them here so they can never enter ANSIContent.
                pass

            case ["osc", osc]:
                body = self._strip_osc_terminator(osc[1:])
                kind, _, payload = body.partition(";")
                match kind:
                    case "8":
                        link = body.rsplit(";", 1)[-1]
                        self.style += Style(link=link or None)
                    case "2025":
                        current_directory = self._decode_chrys_osc_payload(payload)
                        if current_directory:
                            self.current_directory = current_directory
                            yield ANSIWorkingDirectory(current_directory)
                    case "2026":
                        command = self._decode_chrys_osc_payload(payload)
                        if command:
                            self.last_command = command
                            yield ANSIShellCommand(command)

            case ["csi", csi]:
                sgr_params = csi[1:-1]
                if csi.endswith("m") and not any(c not in "0123456789;:" for c in sgr_params):
                    if (sgr_style := self._parse_sgr(sgr_params)) is None:
                        self.style = NULL_STYLE
                    else:
                        self.style += sgr_style
                        # Special case to use widget background rather
                        # than theme background
                        if sgr_style.background is not None and sgr_style.background.ansi == -1:
                            self.style = Style(foreground=self.style.foreground) + sgr_style.without_color
                    yield ANSIStyle(self.style)
                else:
                    if (ansi_segment := self._parse_csi(csi)) is not None:
                        yield ansi_segment

            case ["dec", dec]:
                slot, character_set = list(dec)
                yield ANSICharacterSet(DEC(DEC_SLOTS[slot], character_set))

            case ["dec_invoke", dec_invoke]:
                yield ANSICharacterSet(dec_invoke=self.DEC_INVOKE_MAP[dec_invoke[0]])

            case ["control", code]:
                if (control := CONTROL_CODES.get(code)) is not None:
                    if control == "ri":  # control code
                        yield ANSICursor(delta_y=-1, auto_scroll=True)
                    elif control == "ind":
                        yield ANSICursor(delta_y=+1, auto_scroll=True)
                    elif control == "decsc":
                        yield ANSICursorSave()
                    elif control == "decrc":
                        yield ANSICursorRestore()
                    else:
                        pass
                else:
                    pass

            case ["content", text]:
                yield ANSIContent(text)

            case _:
                pass
