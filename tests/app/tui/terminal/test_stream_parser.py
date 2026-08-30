# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the vendored terminal stream parser."""

from chrys.app.tui.terminal.ansi._stream_parser import ParseResult, StreamParser


class _FiniteParser(StreamParser[str]):
    def parse(self) -> ParseResult[str]:
        yield self.read(1)
        yield "complete"


def test_feed_stops_when_finite_parser_exhausts_with_unconsumed_text() -> None:
    parser = _FiniteParser()

    assert list(parser.feed("ab")) == ["complete"]
    assert parser._gen is None
