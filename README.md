```
 ▄   ▄▄▄▄
 ▀██████▀ █▄
   ██     ██    ▄
   ██     ████▄ ████▄██ ██ ▄██▀█
   ██     ██ ██ ██   ██▄██ ▀███▄
   ▀█████▄██ ██▄█▀  ▄▄▀██▀█▄▄██▀ ▄
                       ██
                     ▀▀▀
```

[![CI](https://github.com/0x7c13/chrys/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/0x7c13/chrys/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/0x7c13/chrys?style=flat-square)](https://github.com/0x7c13/chrys/releases)
![platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-yellow?style=flat-square)

> **Work in progress.** Interfaces, file formats, and defaults still change between
> releases. Found a rough edge? [Open an issue](https://github.com/0x7c13/chrys/issues).

**Chrys is an agent that lives in your terminal, and a platform for building your own.**

A full-screen terminal UI where the agent reads and writes files, runs shell commands,
searches your codebase, and delegates to sub-agents, with every tool call rendered as its
own card. The same engine runs headless in scripts, speaks the Agent Client Protocol so
editors can drive it, and serves the same UI to a browser.

Two things set it apart:

- **Flexible, and quick to drive.** Swap models with **F4** and agents with `#` mid-conversation,
  mixing providers and specialised agents inside one session. The terminal UI is the point, not a
  compromise: mouse-aware, fully scrollable, with an embedded shell and a modal editor a keystroke away.
- **Agents are configuration, not code.** An agent is a YAML file listing its instructions,
  tools, skills, MCP servers, and sub-agents. Building a new one means editing a file,
  not forking the project.

Providers wired up today: Anthropic, OpenAI, DeepSeek, and GLM, plus any OpenAI-compatible
endpoint including local models.

## Install

There is no PyPI release yet; `pip install chrys` fetches an unrelated project. Take a binary
from [Releases](https://github.com/0x7c13/chrys/releases) instead. No Python required, and
nothing is downloaded at runtime — the binary carries its own interpreter and every
dependency, so it works on an air-gapped machine.

```bash
VER=0.18.1                      # check Releases for the current one
ASSET=chrys-macos-aarch64       # or macos-x86_64, linux-x86_64, linux-aarch64
curl -fsSLO https://github.com/0x7c13/chrys/releases/download/v$VER/$ASSET-v$VER-offline.tar.gz
tar xzf $ASSET-v$VER-offline.tar.gz
chmod +x ./chrys                # the archive does not carry the exec bit
./chrys --version               # the first run unpacks itself, then starts instantly
./chrys install                 # copy into ~/.local/bin
```

Windows ships `chrys-windows-x86_64-v$VER-offline.zip` containing `chrys.exe`: unzip it, run
`.\chrys.exe install`, then open a new terminal. If `chrys` still isn't found on macOS or
Linux, `~/.local/bin` isn't on your `PATH` and the installer prints the line to add.
Already on Python 3.14? The published wheel works too, with the `[tui]` extra.

The Linux binaries need glibc 2.17 on x86_64 (CentOS 7, Ubuntu 14.04) and glibc 2.18 on
aarch64 — Alpine and other musl-only distributions are not covered.

## Point it at a model

Chrys ships agent profiles but no model profiles: you bring your own. Press **F4** in the
TUI to create and activate one, or write the file yourself.

```yaml
# ~/.chrys/models/my-gpt.yaml   (%APPDATA%\chrys\models\ on Windows)
name: GPT-5.6
provider: openai          # or anthropic, deepseek-openai, glm-openai
model_id: gpt-5.6
api_key: sk-...           # or leave blank and export OPENAI_API_KEY
max_context_tokens: 1000000
max_output_tokens: 64000
```

```bash
chmod 600 ~/.chrys/models/*.yaml    # profiles hold API keys; chrys won't chmod for you
chrys models                        # '*' marks the active profile
```

The profile id defaults to the filename stem. Set `base_url` to reach any OpenAI-compatible
endpoint, and `api_style: responses` for OpenAI or DeepSeek reasoning models. Keys belong in
the profile or in your shell. Chrys also reads `~/.chrys/.env`, whose values override the
shell; `chrys acp` additionally loads the `.env` of the directory it is launched in. Keep
keys out of any `.env` a repo might commit.

## Run it

```bash
chrys                                 # the TUI
chrys run "Say hello" --agent Code    # headless; approves every tool call silently
chrys acp --agent Code                # ACP stdio server, for editors that speak it
chrys serve --auth-required           # the TUI in a browser (localhost, password-gated)
chrys agents                          # what's available
chrys models
```

In the TUI the input bar is most of the interface: `/` for slash commands, `@` for fuzzy
file mentions, `#` to switch agents, `!` to drop into an embedded terminal, and Ctrl+O for
a modal editor with Vim, Emacs, and standard keymaps. Function keys open the pickers:
F1 sessions, F2 agents, F4 models, F9 themes.

Tool calls need your approval by default. You can hand that decision to an LLM judge, or
turn it off. `/diff` reviews what the agent changed, and `/rollback` rewinds a turn — the chat
*and*, after showing you the diff, your working tree.

## Make it yours

An agent is one YAML file. Here is the bundled `QA` agent, trimmed:

```yaml
name: QA
display_name: "Q&A Agent"
description: "Read-only conversational assistant for answering questions about code, architecture, and usage"
instructions: |
  You are a knowledgeable Q&A assistant. You answer questions about code, architecture,
  APIs, conventions, and usage by exploring the codebase and citing what you find...
tools:
  builtins: [filesystem.read, search, shell, sleep, doc_converter, todo]
sub_agents:
  max_total_concurrency: 3
  agents:
    - profile: Explore
      tool_name: explore_agent
      max_concurrency: 3
      tool_description: Read-only codebase exploration; returns a cited summary.
memory:
  files: [AGENTS.md]          # injected into the system prompt
approval:
  default: auto
  overrides:
    shell: require
```

Drop a file of the same shape into `~/.chrys/agents/` and it shows up in the picker and in
`chrys agents`. Reuse a built-in's name and yours shadows it. The same file can attach MCP
servers, [Agent Skills](https://agentskills.io/specification) directories, or an external
ACP agent such as Claude Code or Codex as one of your sub-agents. **F2** opens an in-app
editor for all of it. Lifecycle hooks are configured by hand, in `~/.chrys/hooks/hooks.yaml`
or a project's `.chrys/hooks/hooks.yaml`.

The [`examples/contextgraph-memory`](examples/contextgraph-memory) profile shows how to attach
a validated ContextGraph experience graph as bounded, read-only, untrusted MCP context.

Built-in agents cannot be deleted. Their **Reset** action restores the bundled profile while
preserving Skills, MCP, and Memory settings. When any preserved setting differs from the
bundle, Chrys must keep a full shadow YAML; later bundled changes to other fields remain
masked until the agent is reset again.

Chrys configuration and session state live under `~/.chrys`, or `%APPDATA%\chrys` on
Windows. Not everything is there: the launcher unpacks its Python runtime into the platform
data and cache directories, `chrys install` puts the binary in `~/.local/bin`
(`%LOCALAPPDATA%\chrys\bin` on Windows), and agents write to the workspace you point them at.

## Hacking on chrys

[uv](https://github.com/astral-sh/uv) is the only prerequisite; it provisions Python 3.14
for you.

```bash
git clone https://github.com/0x7c13/chrys.git
cd chrys
uv sync --extra all        # not bare `uv sync`, not `--all-extras`
./scripts/fetch_rg.sh      # downloads vendored ripgrep; Windows: .\scripts\fetch_rg.ps1
uv run pre-commit install
uv run chrys
```

The source is five tiers with one-way dependencies, enforced by a test that fails the build
if a lower tier reaches upward:

| tier | owns |
| --- | --- |
| `app` | CLI, TUI, ACP server, installer |
| `orchestration` | engine, sub-agents, session host |
| `service` | LLM clients, tools, MCP, skills, hooks, approval, profiles, compaction |
| `kernel` | provider-agnostic agent runtime |
| `foundation` | settings, event bus, platform, vendored ripgrep |

CI gates lint, formatting, types, a three-OS test run, and a wheel build. Reproduce it
before you push:

```bash
LINT="src/ tests/ scripts/calibrate_gc_freeze.py scripts/gc_freeze_calibration_math.py"
uv run ruff check --no-fix $LINT
uv run ruff format --check $LINT
uv run ty check --error-on-warning src/chrys
uv run pytest -m "not integration and not gc_calibration"
```

Two traps: `ruff check` autofixes unless you pass `--no-fix`, and a bare `uv run pytest`
picks up real-network integration tests that CI skips. Use `-n0` to debug a flake.

**Read [AGENTS.md](AGENTS.md) before your first PR.** It is written dense for coding agents,
but it is still the fastest way to learn the conventions and invariants this README leaves
out.

## License

MIT. Copyright (c) 2026 Chrys.
