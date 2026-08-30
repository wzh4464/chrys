# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for ACP stdio wire discipline."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading

from tests.support.paths import REPO_ROOT

_INITIALIZE_RESPONSE_TIMEOUT_SECONDS = 45


def _drain_lines(stream, sink: queue.Queue[str]) -> None:
    try:
        for line in stream:
            sink.put(line)
    finally:
        sink.put("")


def test_acp_stdio_stdout_contains_only_json_rpc(tmp_path) -> None:
    home = tmp_path / "home"
    env = os.environ.copy()
    env["HOME"] = os.fspath(home)
    # The child is a fresh process, so the in-process config-dir guard cannot
    # reach it and ``HOME`` is all it has to go on — except on Windows, which
    # derives the config directory from these two instead. Without them the
    # child boots against the developer's real ``%APPDATA%\chrys`` and runs its
    # startup migrations there.
    env["USERPROFILE"] = os.fspath(home)
    env["APPDATA"] = os.fspath(tmp_path / "appdata")
    env["NO_PROXY"] = "[::1]"
    proc = subprocess.Popen(
        [sys.executable, "-m", "chrys.app.cli.app", "acp"],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    stdout_lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=_drain_lines, args=(proc.stdout, stdout_lines), daemon=True)
    reader.start()
    stderr_chunks: list[str] = []
    stderr_reader = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read()),  # type: ignore[union-attr]
        daemon=True,
    )
    stderr_reader.start()
    try:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        }
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        try:
            # A cold interpreter imports the full ACP stack before it reads
            # stdin. Under xdist CPU pressure that can exceed 20 seconds even
            # though the server is healthy, so retain a bounded but CI-safe
            # startup budget below the test's global timeout.
            line = stdout_lines.get(timeout=_INITIALIZE_RESPONSE_TIMEOUT_SECONDS)
        except queue.Empty:
            raise AssertionError(
                "ACP server did not write an initialize response "
                f"within {_INITIALIZE_RESPONSE_TIMEOUT_SECONDS}s (returncode={proc.poll()})"
            ) from None
        assert line, "ACP server closed stdout before sending initialize response"
        response = json.loads(line)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response

        proc.stdin.close()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        reader.join(timeout=5)
        stderr_reader.join(timeout=5)

    assert proc.returncode == 0
    remaining: list[str] = []
    while True:
        try:
            extra = stdout_lines.get_nowait()
        except queue.Empty:
            break
        if extra and extra.strip():
            remaining.append(extra)
    assert remaining == []
    assert "NO_PROXY is invalid" in (stderr_chunks[0] if stderr_chunks else "")
