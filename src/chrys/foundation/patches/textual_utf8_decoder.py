# Copyright (c) 2026 Chrys. All rights reserved.

"""Patch: use ``errors="replace"`` for UTF-8 incremental decoders in Textual drivers.

Problem
-------
The input threads in ``LinuxDriver``, ``LinuxInlineDriver``, and
``WebDriver`` create a strict UTF-8 decoder (the default).  If any
invalid byte sequence arrives from the terminal (e.g. a locale mismatch,
raw binary leaking into stdin, or a broken paste), the decoder raises
``UnicodeDecodeError`` and crashes the input thread, killing all
keyboard/mouse handling.

Solution
--------
Pass ``errors="replace"`` so invalid bytes are replaced with U+FFFD
instead of raising.  The application stays responsive; garbled input is
visibly wrong but non-fatal.
"""

from chrys.foundation.patches.patcher import FilePatch, register

_OLD = 'utf8_decoder = getincrementaldecoder("utf-8")().decode'
_NEW = 'utf8_decoder = getincrementaldecoder("utf-8")(errors="replace").decode'

register(
    FilePatch(
        package="textual",
        module_file="drivers/linux_driver.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Use errors='replace' for UTF-8 decoder in LinuxDriver",
    ),
    FilePatch(
        package="textual",
        module_file="drivers/linux_inline_driver.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Use errors='replace' for UTF-8 decoder in LinuxInlineDriver",
    ),
    FilePatch(
        package="textual",
        module_file="drivers/web_driver.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Use errors='replace' for UTF-8 decoder in WebDriver",
    ),
)
