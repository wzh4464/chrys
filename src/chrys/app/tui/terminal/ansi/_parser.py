# Copyright (c) 2026 Chrys. All rights reserved.
# Adapted from toad (https://github.com/batrachianai/toad).

from __future__ import annotations

import io

from chrys.app.tui.terminal.ansi._stream_parser import (
    ParseResult,
    Pattern,
    PatternCheck,
    PatternToken,
    ReadUntil,
    SeparatorToken,
    StreamParser,
)


def character_range(start: int, end: int) -> frozenset:
    """Build a set of characters between to code-points.

    Args:
        start: Start codepoint.
        end: End codepoint (inclusive)

    Returns:
        A frozenset of the characters..
    """
    return frozenset(map(chr, range(start, end + 1)))


_CONTROL_CHARACTER_READ = ReadUntil[tuple[str, str]](*map(chr, range(0x20)), "\x7f")


class FEPattern(Pattern):
    FINAL = character_range(0x30, 0x7E)
    INTERMEDIATE = character_range(0x20, 0x2F)
    CSI_TERMINATORS = character_range(0x40, 0x7E)
    OSC_TERMINATORS = frozenset({"\x07", "\x9c"})
    DSC_TERMINATORS = frozenset({"\x9c"})

    def check(self) -> PatternCheck:
        sequence = io.StringIO()
        store = sequence.write
        store(character := (yield))

        match character:
            # CSI
            case "[":
                CSI_TERMINATORS = self.CSI_TERMINATORS
                while (character := (yield)) not in CSI_TERMINATORS:
                    store(character)
                store(character)
                return ("csi", sequence.getvalue())

            # OSC
            case "]":
                last_character = ""
                OSC_TERMINATORS = self.OSC_TERMINATORS
                while (character := (yield)) not in OSC_TERMINATORS:
                    store(character)
                    if last_character == "\x1b" and character == "\\":
                        return ("osc", sequence.getvalue())
                    last_character = character
                store(character)

                return ("osc", sequence.getvalue())

            # DCS
            case "P":
                pass  # DCS (not implemented)
                last_character = ""
                DSC_TERMINATORS = self.DSC_TERMINATORS
                while (character := (yield)) not in DSC_TERMINATORS:
                    store(character)
                    if last_character == "\x1b" and character == "\\":
                        break
                    last_character = character
                store(character)
                return ("dcs", sequence.getvalue())

            # Character set designation
            case "(" | ")" | "*" | "+" | "-" | "." | "/":
                if (character := (yield)) not in self.FINAL:
                    return False
                store(character)
                return ("dec", sequence.getvalue())

            case "n" | "o" | "~" | "}" | "|" | "N" | "O":
                return ("dec_invoke", sequence.getvalue())

            # Line attribute
            case "#":
                store((yield))
                return ("la", sequence.getvalue())
            # ISO 2022: ESC SP
            case " ":
                store((yield))
                return ("sp", sequence.getvalue())
            case _:
                return ("control", character)


class ANSIParser(StreamParser[tuple[str, str]]):
    """Parse a stream of text containing escape sequences in to logical tokens."""

    def parse(self) -> ParseResult[tuple[str, str]]:
        ESCAPE = "\x1b"

        while True:
            token = yield _CONTROL_CHARACTER_READ
            if isinstance(token, SeparatorToken):
                if token.text == ESCAPE:
                    token = yield self.read_patterns("\x1b", fe=FEPattern())
                    if isinstance(token, PatternToken):
                        # `PatternValue` admits `bool` for patterns that only signal a match;
                        # every `FEPattern.check()` arm returns a `TokenMatch` tuple.
                        yield token.value  # ty: ignore[invalid-yield]
                else:
                    yield "c0", token.text
                continue

            # A `StreamRead` yield always resumes with a `Token`; only a ParseType yield is
            # echoed back by the driver, and those paths `continue` above.
            yield "content", token.text  # ty: ignore[unresolved-attribute]
