# 长程套件：下一个 session 的执行清单

用途：新 session 打开本文件，按顺序读文件、做任务、勾选。所有决定已在 spec 里定案，不要重新讨论。

## 0. 先读什么（按顺序，约 20 分钟）

- [x] `docs/superpowers/specs/2026-09-03-long-horizon-suite-design.md`：设计。重点第 3 节架构图、D2（路由）、D3（长程 track = 完整 RC + 并行定位 + 委派 pass）、D4/D5（memory）、D6（内置 profile）、D7（定位走 chrys 模型）、第 8 节"已确认的决定"。
- [x] `docs/superpowers/plans/2026-09-03-long-horizon-suite.md`：36 个 task 的实施计划。每个 task 自带文件、接口、测试、验证命令。
- [x] `AGENTS.md`：全文。尤其"Layering DAG"、"Conventions & lint footguns"、"Top gotchas"里的 system reminder / 换 profile / turn marker 规则、"Testing rules"。
- [x] `~/.claude/projects/-Users-zihanwu-Public-codes-chrys/memory/chrys-long-horizon-suite-plan.md`：三轮用户决定的摘要。

## 1. 环境准备

- [x] `git fetch origin --prune`；确认存在 `origin/feature/pact`、`origin/feature/requirement-clarification`、`origin/feature/semantic-graph`（`origin/feature/events-ledger` 不用）。
- [x] `uv sync --extra all && ./scripts/fetch_rg.sh`。
- [x] 需要对照上游分支源码时，在会话暂存目录建 worktree（不要放进仓库）：

```bash
SP=<scratchpad>; mkdir -p $SP/wt
for b in pact semantic-graph requirement-clarification; do git worktree add --detach $SP/wt/$b origin/feature/$b; done
# 用完：git worktree remove --force $SP/wt/<b>; git worktree prune
```

- [x] CI 四件套别名（每个 task 结束前跑）：

```bash
uv run ruff check --no-fix src/ tests/ && uv run ruff format --check src/ tests/ && uv run ty check --error-on-warning src/chrys
uv run pytest -m "not integration and not gc_calibration"
```

- [x] 规则：不 commit / 不 push，除非用户明确说。**实际执行**：用户以 `/goal 执行全部，直到完成` 授权了逐 task commit；36 个 task 全部 commit 在本地 `integration/long-horizon-suite` 上，**未 push 任何分支**。
- [x] 执行方式：用户选择了本 session 批量（`/goal 执行全部，直到完成`），未拆子代理。

## 2. 已经勘察过的代码锚点（做对应 task 时先读这些）

| 主题 | 文件与行号 | 用途 |
| --- | --- | --- |
| fresh turn 入口 | `src/chrys/orchestration/engine/run/runner.py:138-232` | `run_fresh`；RC 在此分流，长程 track 也在此分流（Task 28） |
| 消息 admission | `src/chrys/orchestration/engine/run/coordinator.py:287-470` | `_admit_user_message`；路由器插在 `_handle_user_prompt_submit_decision` 之后、`prepare_user_contents` 之前（Task 18） |
| profile 切换 | `src/chrys/orchestration/engine/state/controls.py:213-255` | `on_profile_switch`：rebuild permit + `soft_restart_with_rebuild_permit`，历史保留（Task 18） |
| ACP 客户端 switch | `src/chrys/app/acp/session_manager.py:869-895` | `switch_agent` 如何等 `ProfileSwitched`（Task 18/20 参考） |
| after_turn / hooks | `src/chrys/orchestration/engine/run/turn_hooks.py:75-130`、`src/chrys/service/hooks/events.py:16-133` | 只用于理解现状；本套件不加 hooks.yaml |
| MCP overlay | `src/chrys/orchestration/engine/engine.py:1138-1144`、`src/chrys/orchestration/engine/build/builder.py:588-640`、`src/chrys/orchestration/sub_agents/tools.py:502-516` | 主 agent / sub-agent 读 `profile.tools.mcp` 的位置（Task 6） |
| ACP sub-agent 注册与深度 | `src/chrys/orchestration/sub_agents/tools.py:735-760`（`register_acp`）、`:1195-1235`（`_resolve_acp_spec`，`CHRYS_ACP_SUBAGENT_DEPTH`）、`src/chrys/service/acp_client/spawn.py:28-66` | Task 23（max_depth）、Task 25（`command: chrys` 自解析） |
| 引擎生命周期 | `src/chrys/orchestration/engine/engine.py:810-830`（session_end）、`:1235`（shutdown）、`:1430`（`_save_current_session`）、`:1814`（`_run_and_save`） | Task 11 watcher 接线 |
| 运行时元数据 | `src/chrys/service/session/runtime_metadata.py:33-76` | Task 9 水位线 |
| Settings 声明 | `src/chrys/foundation/config/settings.py:712-760`（示例字段）、`src/chrys/foundation/config/spec.py:335-372`（`spec()` 签名） | Task 5/14/24/26 |
| Profile schema / loader | `src/chrys/service/profiles/agents/schema.py`（`AcpAgentConfig` 第 14 行起、`AgentProfile` 第 340 行起）、`loader.py` | Task 13/23 |
| 内置 profile | `src/chrys/service/profiles/agents/builtins/{Code,QA,Explore,General,Plan}.yaml` | Task 25 |
| 系统提醒中间件 | `src/chrys/service/agent_middleware/system_reminder.py:967`（`queue_hook_reminders`）、`:985`（`set_profile_switch`） | Task 29/30 reminder 注入 |
| 上下文 provider | `src/chrys/service/context/manager.py`、`providers/memory.py` | 了解 system prompt 组装；本套件不往 system prompt 放每轮数据 |
| 一次性 LLM side call 范式 | `src/chrys/service/approval/judge.py:436-530`、`src/chrys/service/llm/clients.py:393`（`create_client`）、`src/chrys/service/llm/responses.py`（`get_final_response`） | Task 16/26/28 |
| memory 现有模块 | `src/chrys/service/memory/contextgraph_mcp.py:350-408`、`contextgraph_deposit.py:150-232`、`contextgraph_repository.py`、`_contextgraph_repository_worker.py` | Task 7/8 |
| 分层测试 | `tests/architecture/test_layering.py`、`tests/orchestration/engine/test_protocol_compliance.py` | 每次给 host 加属性都要过 |
| TUI slash 命令 | `src/chrys/app/tui/screens/main/commands.py:426-760`、`src/chrys/app/tui/widgets/chrome/commands.py:62-100` | Task 19 |
| ACP ext method | `src/chrys/app/acp/server.py:486-560` | Task 20 |
| ContextGraph 本地 | `~/codes/ContextGraph`（`agent_memory/memory.py:279 learn`、`writer.py:22 RawTrajectory`）、`Public/codes/ContextGraph-chrys`（`apply_chrys`，`docker-compose.yml` neo4j 7687/7690） | Task 12/33 |

合入后才存在、做 M4 前必读：

| 主题 | 文件 | 用途 |
| --- | --- | --- |
| RC 设计 | `docs/design/requirement-clarification-integration.md`、`docs/design/requirement-clarification-guide.md`（合入时从仓库根移过来） | 第 5 节执行过程、第 9 节 artifact 树、`06-pact-input` |
| RC 工作流 | `src/chrys/orchestration/engine/run/requirement_clarification.py`（`clarification_only` 分支约第 455-500 行；ΔR reminder 注入约第 517 行；`_run_fresh_standard` 调用点 186/221/261/289） | Task 27 扩展点插入位置 |
| RC 服务层 | `src/chrys/service/requirement_clarification/service.py:50-330`（`clarify`、`generate_pact_input`、`validate_pact_runtime_input`）、`model.py:399-700`（`ChrysClarificationModel`）、`artifacts.py:249-303`（`save_pact_generation`）、`types.py:276-320`（PACT schema） | Task 26/27/29/30 |
| PACT 设计 | `docs/design/chrys-pact-integration.md` | 第 3 节 launch 契约（run-request JSON、`.pact-io` 路径规则）、第 7 节 profile 配置 |
| PACT 代码 | `src/chrys/pact/cli.py:22-90`、`role_runner.py:99-140`（`_default_host_factory`、`_derive_turn_settings`）、`server.py:239`（`parse_launch_request`） | Task 24 |
| semantic-search | `src/chrys/service/semantic_search/pipeline.py`、`skill/scripts/_localization_agent.py`（497 行：`LocalizationAgent.run` 循环、`OpenAIChatClient` 待删）、`_localization_tools.py`（五个工具）、`SKILL.md` | Task 26 移植 |

## 3. 按里程碑执行

每个 task：先读上表对应锚点 → 写失败测试 → 实现 → 跑该 task 的验证命令 → 跑 CI 四件套 → 勾选。

### M0 集成分支（Task 1–4）
- [x] Task 1：`git switch -c integration/long-horizon-suite origin/main`；合入 RC 时剔除 `dist/archived-binaries/`（98 MB），文档移到 `docs/design/`。
- [x] Task 2：合入 pact；`pyproject.toml` 三处变化都保留（`pact-core` 依赖、`[tool.uv.sources]`、`allow-direct-references`）；`uv lock`。
- [x] Task 3：`git rebase --onto integration/long-horizon-suite e0cd89fa <semgraph 分支>` 只带两个 commit；DeepSWE 脚本与报告移到 `evaluation/semantic_search/`。
- [x] Task 4：合入 `codex/contextgraph-memory`；全量 CI 绿；AGENTS.md 增加三个子系统条目。

### M1 memory 全局暴露（Task 5–7）
- [x] Task 5 settings（3 个 memory 字段）→ Task 6 `service/memory/overlay.py` + engine/builder 两处应用 → Task 7 MCP `instructions` 文本，删除 `examples/contextgraph-memory/hooks/hooks.yaml`。

### M2 空闲写回（Task 8–12）
- [x] Task 8 `writeback.py` 水位线 → Task 9 `runtime_metadata` 持久化 → Task 10 `MemoryWritebackWatcher`（可注入时钟）→ Task 11 engine 接线 + `MemoryWritebackCompleted` → Task 12 `chrys memory doctor|sweep|init`（`--import` 入口留着，dump 由用户提供）。

### M3 路由（Task 13–22）
- [x] Task 13 profile `routing`（含 `target_profile`）→ Task 14 settings `routing_mode` + `routing_tiebreaker_model_profile` → Task 15 双语启发式 + 五档 + `readiness.py` → Task 16 受保护裁决（`guard.py`、`json_extract.py` 从 judge 抽出）→ Task 17 事件 → Task 18 `TurnRouter` 在 admission 阶段决定并切 profile → Task 19 TUI `/longrun /quick /route` + 公告 → Task 20 ACP ext method → Task 21 `chrys run --route` → Task 22 `chrys debug router` + 60 条双语校准样本门禁。
- [x] i18n：Task 19 新增 `msg()` 后跑 `extract → update → compile → check` 两遍并更新 `tests/foundation/i18n/test_catalog_artifacts.py` 的 oracle。

### M4 长程工作流（Task 23–33）
- [x] Task 23 `AcpAgentConfig.max_depth` → Task 24 `pact-agent --verify-from-settings` + 角色 host `routing_mode=off` → Task 25 内置 `ChrysPact`、`LongHorizon`，`Code` 只加 `target_profile`。
- [x] Task 26 定位 LLM 循环移入进程（`ChrysLocalizationModel`），便宜模型 setting `semantic_search_model_profile`，skill 打包。
- [x] Task 27 RC 六个扩展点 + `NoopExtensions`（RC 现有测试必须原样绿）。
- [x] Task 28 `LongHorizonExtensions`：定位与 clarify 并行（读 S0 视图）+ `SideCallClientCache`。
- [x] Task 29 合并：repair reminder = ΔR + 定位表；`brief.md`；PACT 提示。
- [x] Task 30 repair 之后的委派 pass（`.pact-io`、run-request reminder、第三次 executor pass）。
- [x] Task 31 turn marker `_chrys_route`（baseline + campaign）→ Task 32 amendment/interrupt/`/quick` 降级/CLI 定位 → Task 33 示例与 e2e（需要本地 Neo4j 与真实模型；初始图留空）。

### M5 收尾（Task 34–36）
- [x] Task 34 沉淀语义（PACT `completed` 才算 success；problem_statement 取澄清需求单）→ Task 35 记忆先验（可选）→ Task 36 文档与已知缺口。

## 4. 不要做的事

- 不加 hooks.yaml 的静态 hook；写回只走引擎 watcher 和 `chrys memory sweep`。
- 不把每轮数据放进 system prompt 或持久化历史；用 `SystemReminderMiddleware.queue_hook_reminders`。
- 不在 `Code` 上挂 `chrys_pact`；增强工具只在 `LongHorizon`。
- 不跳过 P0/repair；长程 track 就是完整 RC 加扩展点。
- 不在脚本里自建 OpenAI 客户端；所有 LLM 调用走 `service/llm/clients.create_client`。
- 不合入 `feature/events-ledger`；不把 98 MB 二进制和 DeepSWE 报告带进 `src/`。

## 5. 待用户提供

- [x] **ContextGraph 初始图**：用户指定 CAPBench selected-Harbor 图（`bolt://127.0.0.1:7705`）。已启动本地 Neo4j 5.26 tarball 运行时并接进 `~/.chrys/.env`；`chrys memory doctor` 五项全绿，health `canonical_rules=2525, chrys_trajectories=0`。嵌入模型对齐为 `text-embedding-3-large`（库内向量 3072 维）。记忆先验实测返回相关规则。见 known-gaps §3。
- [x] **`pact.verify_command`**：由我决定，写进仓库 `.chrys/settings.yaml`——CI 四件套按"最便宜的先失败"排序，含 `LANG=en_US.UTF-8` 与那条既有环境失败的 deselect。干净树上 exit 0 / ~130 s，且已验证会对坏代码报错。注意 `project.config_enabled` 默认 False（克隆的仓库不自动获得配置权限），**没有**替用户打开；单次运行可用 `CHRYS_PACT_VERIFY_COMMAND` 覆盖。
- [x] 执行方式：本 session 批量。

## 6. 完成一个 task 后

- [ ] 在 `docs/superpowers/plans/2026-09-03-long-horizon-suite.md` 里勾掉该 task 的步骤。
- [ ] 遇到与 spec 不符的现实（签名不同、行号漂移），改 plan 里的对应 task，并在本清单末尾"偏差记录"追加一行。

## 偏差记录

| 日期 | Task | 偏差 | 处理 |
| --- | --- | --- | --- |
| 09-03 | 全局 | 本机 `LANG=zh_CN.UTF-8`，TUI 测试断言英文文案，直接跑 pytest 会有约 173 个假失败 | 所有 pytest/i18n 命令一律加 `LANG=en_US.UTF-8` 前缀 |
| 09-03 | 全局 | `tests/service/skills/test_runner.py::test_stopped_script_returns_promptly` 在 `origin/main` 上同样挂死（Python 3.14 asyncio `os.waitpid` 子进程 watcher，macOS 环境问题） | 已用 `origin/main` worktree 对照确认为既有失败；全量跑加 `--deselect` |
| 09-03 | Task 1 | RC 分支未跑过当前 main 的门禁：`msg()` 写在 dict 字面量里、6 条词条未进 POT/PO、`format_message(MessageDef)` ty 报错、两处冗余 `cast`、`model_lock.py` 未 format | 全部就地修复；i18n 流水线跑两遍、补 zh-Hans 翻译、oracle 1812→1818 |
| 09-03 | Task 1 | main 的 `test_stream_stall.py` 用 `object.__new__(Executor)` 造桩，RC 新增的 `_requirement_phase` 未被 git 视为冲突（语义冲突） | 6 处桩补 `executor._requirement_phase = ""` |
| 09-03 | Task 1/3 | `tests/architecture/trajectory_wait_manifest.json` 因新代码与行号漂移失效 | `uv run python -m tests.support.trajectory_wait_inventory` 重新生成（分类由 AST 确定性推导） |
| 09-03 | Task 2 | 计划预期 `app.py`/`AGENTS.md`/`README.md`/`pyproject.toml`/`uv.lock` 冲突，实际全部自动合并成功，`uv lock` 无漂移 | 无需处理；已核对 pyproject 三处新增都在 |
| 09-03 | Task 3 | semantic-graph 分支违反当前 main 两条门禁：`pipeline.py` 子进程缺 `stdin=`、`chrys locate` 未调 `bootstrap_runtime` | 补 `stdin=subprocess.DEVNULL`；给 `locate.py` 加 `_prepare_runtime()`（照 `profiles.py` 范式） |
| 09-03 | Task 3 | `deepswe_runner.py` 移出 `scripts/` 后 `parents[1]` 指错 | 改为 `parents[2]` |
| 09-03 | Task 5 | 持久化 setting 必须带 `label`，三个 memory 字段都要 `msg()` + zh-Hans 翻译 + oracle（1818→1821）；面板无对应行，故登记进 `DEFERRED_KEYS` | 已办 |
| 09-03 | Task 6 | 计划只说在 builder 的注册处应用 overlay；实际 `effective_sub_agents` 指纹循环也要应用，否则改 memory 设置不会让 sub-agent 指纹失效 | 两处都应用 |
| 09-03 | Task 8 | `WritebackOutcome.deposited` 改为只记真正写入的 turn（无工具动作的 turn 只推进水位线），避免事件里报出并未写入的条数 | 已办 |
| 09-03 | Task 9 | 计划把 `WATERMARK_KEY` 定义在 `writeback.py`；改为定义在 `runtime_metadata.py`（与其它 state key 同处），`writeback.py` 再导出 | 已办 |
| 09-03 | Task 12 | 计划要 chrys 自己幂等建四个索引；实际 schema 归 ContextGraph 的 `Neo4jStore.init_schema` 所有，重复声明会在上游改动时静默分叉 | 改为经隔离 worker 新增 `init_schema` op 委托给上游 |
| 09-03 | 评审 | `/code-review high` 报 7 条：5 条已修（见 `a81e602d`）；wheel 未打包 semantic-search skill 归 Task 26；restore 不删非空目录经核实不是缺陷（被 gitignore 的文件本就在回滚范围外） | 已办 |
| 09-03 | 评测 | `deepswe_runner._run_locate` 不把 `--timeout` 传给 `chrys locate`，内层吃 120 s 默认值，推理模型必超时 | 新增 `--localization-timeout` 并透传（`fc5f6392`） |
| 09-03 | Task 14 | `routing.mode` 原计划 project 可设；`always` 会强制每轮一条 PACT campaign，是成本升级向量，且三值枚举没有 tighten-only 比较器 | 改为 `ProjectMerge.DENY` |
| 09-03 | Task 18 | 计划说路由插在 `_handle_user_prompt_submit_decision` 之后；实际必须在 `reserve_prompt_admission` **之前**——切 profile 会 bump build generation，先占的 admission slot 会失效 | 放在预留之前，切换后 `continue` 重走全部闸门 |
| 09-03 | Task 18 | 路由异常会吞掉用户这一轮消息 | `_route_turn` 整体兜底：记日志并回落标准链路 |
| 09-03 | Task 18 | `SubAgentTools.tool_names` 是既有方法不是属性（计划写的是属性） | 直接调用 `tool_names()`，测试桩同步 |
| 09-03 | Task 18 | `pact_verify_command`（原属 Task 24）被 readiness 提前需要 | 提前加入，`Risk.DANGEROUS` + `SAFE_DEFAULT` |
| 09-03 | Task 19 | 我给 memory/routing 新设的几个键标了 `Apply.LIVE` 但没有热应用路径 | `idle_seconds` 改 RELOAD；其余在 settings 面板登记为"每次使用时读取" |
| 09-03 | Task 22 | 建校准集时暴露词表三处缺口：无 `everything`、无 `clean up` 这类词组、无 清理/整理 | 补齐后 precision 1.00 / recall 1.00 / strong 零误报 |
| 09-03 | Task 25 | YAML 1.1 把裸 `off` 解析成 `False`，`routing.mode: off` 在 profile 里会变成布尔 | loader 接受 `False` 作为 `"off"` |
| 09-03 | Task 26 | `ruff check --unsafe-fixes` 把 skill CLI 脚本里 406 行 `print()` 当作 T20 违规删掉——破坏性 | `git checkout` 还原脚本目录，只重跑预期的编辑脚本，并给该路径加 `T20`/`PERF` per-file-ignores 与 ty override |
| 09-03 | Task 26 | skill 原在 `.agents/skills/`，不进 wheel（也是评审第 6 条） | 移到 `src/chrys/service/semantic_search/skill/`，核对 wheel 里 15 个文件；`augment_requirement.py`/`analyze_augmented_run.py` 与 DeepSWE harness 移到 `evaluation/semantic_search/` |
| 09-03 | Task 28 | `MagicMock` 会自动生成 `aclose`，`SideCallClientCache` 的关闭测试永远"通过" | 改用真实 double 类；关闭优先 `aclose`，失败只记日志不抛 |
| 09-03 | Task 28 | 计划写的 `Kind.CHOICE` / `Risk.HIGH` 不存在 | 改 `Kind.ENUM` / `Risk.DANGEROUS`（后者要求显式 `InvalidPolicy.SAFE_DEFAULT`） |
| 09-03 | Task 31 | 给 host 加 `_last_route` / `_long_horizon_campaign` / `insert_turn_marker(extra=)` / `history_state` 后，`tests/orchestration/**` 与 `tests/service/session/test_responses_history.py` 的大量 host double 需要同步（`test_protocol_compliance.py` 会挡） | 逐个补齐；`test_resume.py` 有一处插错缩进已修正 |
| 09-03 | Task 33 | e2e 需要目标仓库的 `pact.verify_command` 和本机 Neo4j，两者都还没有，**未能端到端跑过** | 脚本 `bash -n` 通过；能不依赖模型/图跑的两条命令（`chrys debug router` 两种输入、`chrys memory doctor`）真实执行并把输出贴进 `examples/long-horizon/README.md`，同时写明 e2e 未跑及原因 |
| 09-03 | Task 35 | 计划标为"可选"；实现时确认记忆先验在图不可达/未配置/无需求文本时必须完全静默（不 Warning、不改阶段、不发查询） | 已办；先验限 3 条 / 2000 字符，与澄清证据共享 plan prompt 预算 |
| 09-03 | Task 36 | AGENTS.md 的 semantic-search 条目在 Task 26 之后过期（skill 路径、"不进 wheel"的告示） | 一并更新，并补 CodeGraph 安装需固定 sha256、`T20`/`PERF` 不可 `--unsafe-fixes` 两条 |
| 09-03 | 评审二轮 | 第一轮 code review 早于 M3–M5 的代码；对新代码补做一轮，发现 4 条（全部已修并补测试） | 见下四行 |
| 09-03 | 评审二轮 | `probe_workspace_readiness` 每轮都跑 `git status --porcelain`（5 s 超时），而 `git_dirty` 只有 `chrys debug router` 读；且探测排在 `routing.mode == "off"` 早退之前 | `git_dirty` 拆成 `probe_git_dirty()` 由 CLI 直接调；readiness 改为惰性、只有长程分支付代价，标准链路完全不碰文件系统（`4dcbbfc7`） |
| 09-03 | 评审二轮 | 记忆先验在 `pact_input_hints()` 里同步查 Neo4j——阻塞事件循环，外层 `asyncio.timeout` 对阻塞调用无效 | 召回移到 `on_clarification_start`，`to_thread` + 15 s 预算，与澄清并行；渲染变成纯函数（`5fb5a711`） |
| 09-03 | 评审二轮 | `MemoryWritebackWatcher.stop()` 只 suppress `CancelledError`，await 一个已异常死掉的 task 会把异常抛进会话关闭路径 | 改为记日志（`4dcbbfc7`） |
| 09-03 | 评审二轮 | `on_revision` 清了定位候选但留下按旧需求文本召回的先验 | 一并清空（`faa226f5`） |
| 09-04 | 评审三轮 | 09-03 派出的 5 个评审 teammate **全部死于 API Error 529**（22:58），运行时仍报 `running`、`TaskOutput` 永远 not_ready | 从 transcript 抢救出 31 条 findings，停掉死进程，按 CLAUDE.md 的 529 规则在主线程逐条核实（不再派并行子代理） |
| 09-04 | 评审三轮 | 沉淀水位线按位置计数；压缩折叠后水位线倒退，会把没沉淀过的 turn 标记为已沉淀（长会话静默丢数据） | 改为按 turn marker 全局编号，水位线只进不退（`f83a1bbe`） |
| 09-04 | 评审三轮 | `_redact` 先截断后匹配，跨越预算边界的凭据以可读前缀写入 Neo4j | 先在超出预算的窗口脱敏再截断（`fcb9a2e5`） |
| 09-04 | 评审三轮 | 三个 MCP 工具是同步的，FastMCP 直接在事件循环上跑；stdio 读循环停摆，连取消都收不到 | 改 async + `to_thread`（`fcb9a2e5`） |
| 09-04 | 评审三轮 | `_sanitize(None)` 返回字面量 `"None"`，穿过所有空值判断 | None → `""`（`fcb9a2e5`） |
| 09-04 | 评审三轮 | 澄清中止的三条路径只发 phase 事件，TUI 永远转圈、headless 在半修复工作区上退出 0 | 三条路径都投递终止 `Error`（`84ab04b7`） |
| 09-04 | 评审三轮 | `_execute` 吞掉 `CancelledError`，`asyncio.timeout` 因此永不触发——两条 `except TimeoutError` 与 `repair_timed_out` 都是死代码（连带让上一条修复的 P0 超时路径也不可达） | 只吞自己的 interrupt，其余重新抛出（`bf2d4ca9`） |
| 09-04 | 评审三轮 | 澄清阶段 executor 空闲，Stop 最长 1800 s×2 才生效 | Stop 取消 side-call gather；待处理的 Stop 直接跳过 PACT 生成（`4ea502bd`） |
| 09-04 | 评审三轮 | `collect_base_evidence` 在事件循环上跑十几个 git/ripgrep 子进程，每轮三次 | 专用单线程池 + 不等待关闭（`50083b47`） |
| 09-04 | 评审三轮 | 悬空符号链接让 `chrys locate` 整体失败（已复现） | 守护 `stat`（`4dd19e24`） |
| 09-04 | 评审三轮 | `deepswe_runner --repo-root` 在用户自有仓库里 `git clean -fdx` + 强制 checkout，删除 `.env`/虚拟环境/未提交修改 | 改为从这些仓库本地克隆；已手工验证源仓库完好（`4dd19e24`） |
| 09-04 | 评审三轮 | ACP 客户端断开后 `wait_closed()` 无超时，进程最多多活 30 分钟 | 传输关闭路径限 30 s（控制平面线程是 daemon）（`c8933d15`） |
| 09-04 | 评审三轮 | 对照组补丁用 `git diff HEAD`，遗漏新建文件，系统性让 A/B 失真 | 先 `git add --intent-to-add --all`（`070eb3e0`） |
| 09-04 | 评审三轮 | 写入路径硬依赖 embedding key，而 env.example 说"可选"、doctor 只说读会降级 | 三处措辞统一说明写入必需（`070eb3e0`） |
| 09-04 | 评审三轮 | 降级路径上同一段答复被渲染两次（provisional + final） | 终止事件带 `repeats_provisional`，两端跳过重绘但仍关闭 turn（`91408bf8`） |
| 09-04 | 评审三轮 | `candidates/proposal-N` 按位置编号，与 `investigations/` 的真实 proposer 编号错位 | `ClarificationResult` 携带每条提案的 proposer 编号（`24c19002`） |
| 09-04 | 评审三轮 | 定位缓存忽略 `mode`，`fallback` 结果被当作 `llm` 结果复用 | mode 纳入缓存判定（`9b394739`） |
| 09-04 | 评审三轮 | reviewer 的 `.pact-io` 决策路径若是目录会抛错 → 整个已完成的 turn 被记为 `spawn_failed` 且丢弃 final_text | 非文件路径按"没写决策"处理（`9b394739`） |
| 09-04 | 评审三轮 | 评测两处未守护的 `json.loads`（断点续跑的 result.json、被杀死的 verifier reward.json）会终止整批 | 分别降级为该任务失败（`9b394739`） |
| 09-04 | 评审三轮 | artifact 目录在仓库内时，指纹把自己的产物算进去 → 缓存永久失效，每次重跑全量索引+LLM（已复现前后对比） | 指纹排除 artifact 目录（`011e32bf`） |
| 09-04 | 评审三轮 | `verified_evidence` 上限 30 < 提案 schema 允许的 36，合法提案被当作失败 proposer 丢弃 | 上限由提案 schema 推导（`01de905a`） |
| 09-04 | 评审三轮 | `accepts_amendments` 判定与 `accept_amendment` 之间隔着三个 await；期间阶段推进则消息被静默丢弃 | 补发 `UserInjectResult(consumed=False)`（`01de905a`） |
| 09-04 | 评审三轮 | 回滚先扫描后写回 `.gitignore`，等于问"修复自己"哪些文件该回滚；残留文件还会让随后的 `matches()` 误报冲突 | 先写回快照文件再决定多余项（`8e792765`） |
| 09-04 | 评审三轮 | 空目录清理走全树，删掉本就存在的空目录，还会进 `.git` 删 `refs/tags` | 只清理本次删除造成的空目录（`8e792765`） |
| 09-04 | 评审三轮 | 取消通知失败会把已取消的运行报成 `refusal` | 与失败路径一致地抑制（`34de9614`） |
| 09-04 | 评审三轮 | 3 条判定为**不宜盲修**：role 运行时孤儿、重试不更新图、崩溃恢复轮次无 mutation 账本 | 写进 `docs/design/long-horizon-known-gaps.md` §1b、1c、1d，含机制与运维含义 |
| 09-04 | 评审三轮 | 1 条核实后不是缺陷：repair 阶段抑制 `Error` 是刻意的（工作流提升 P0 并交付有效答复）；其"回滚也失败时什么都不投递"的子情形是真缺陷，已单独修复 | 保留原行为，子情形见 `84ab04b7` |
| 09-04 | 重做 | 用户要求前两个模块**尽量复用上游** `feature/requirement-clarification-localization`（squash、88 文件、含 enrichment 预检与 SnapshotReadTools）；两个模块整体取上游，共享引擎文件保留本分支并手工补入上游新增，本分支对这两个模块的修复逐条重放（上游已独立修好的四条不重放）；in-process 定位模型删除，长程 track 改走上游 `localize_requirement_async` | `f44f614c` 合并提交；全量 19451 通过 |
| 09-04 | e2e | 合并后首个端到端跑通（P0/repair/`.pact-io` 全过），但会话产物暴露三处：简报读 `05-outcome/`（finalize 之后才写）→ 写着"(clarification produced none)"；委派 reminder 在调用 `_run_fresh_standard` **之前**入队，挂在 repair pass 的 current state 上，委派 pass 没看到（轨迹：3 次 read + 1 次 zsh，`campaign: null`）；本地定位被默认 120 s 预算掐断 | 简报先读 `03-clarification/deliverable/`；reminder 经 `before_execution` 以 `for_next_turn=True` 入队；smoke 脚本导出 900 s 预算（`8d87fdcc`） |
| 09-04 | e2e | smoke 脚本用 shell 环境变量判断图是否配置，而图在 `~/.chrys/.env` 里由 chrys 自己加载 → 记忆那半在图真正活着的机器上被跳过 | 改问 `chrys memory doctor --json`（`1d2e5939`）；写回已手工验证：会话 1 轮待沉淀 → 沉淀 → 图中 `chrys_trajectories` 0→6 |
| 09-04 | DeepSWE | `runs/first20-lh` 以 `--run-long-horizon` 启动（11:07），模型配置与基线一致，定位预算 1800 s；远端无 `pact.verify_command`，按设计不委派 campaign。第一题 P0 约 15–20 min（推理模型），预计每题 45–60 min | 每 10 min 核对；无需为 `8d87fdcc` 重启（它只影响 campaign 路径） |

### 09-04 远端首轮跑批（lh + enrich，各 10 分片 × 2 题）暴露的偏差

按"每 10 分钟核对一次"逐条抓到并修复，全部带回归测试，提交号见 git log：

- **`--route long-horizon` 从未生效**（`66a19526`）：新会话的 `chrys run` 不调 `host.start()`，而
  `RouteOverride` 的订阅在 `start()` 里注册，override 无人接收；远端 12 个会话 10 个走了标准轨，
  本地 8 轮 e2e 之所以走长程只是因为 prompt 本身 `strong_long_horizon`。修后 10/10 `source: override`。
- **campaign 子进程收不到 verify 命令**（`11ab6e6c`）与**模型 profile**（`66a19526`）：ACP 子进程环境是白名单，
  `command: chrys` 的自启动子进程现在显式收到 `CHRYS_PACT_VERIFY_COMMAND` / `CHRYS_MODEL_PROFILE`。
- **两条 run 的澄清 100% 降级**（`75f1e811`）：lh 树里是 rsync 过去的 macOS arm64 vendored `rg`
  （Linux 上 `Exec format error`），enrich 树没有任何 rg；`SnapshotReadTools` 的 grep/glob 全部失败。
  `find_rg` 现在用 `--version` 探测候选是否真能执行。服务器两棵树都放了 Linux musl 版 rg。
- **Manager 决策被 ```json 围栏挡住**（`75f1e811`）：角色回复整体是一个围栏时先去壳再交给 pact_core。
- **117 次工具调用的 investigation 被整个丢弃**（`75f1e811`）：记录保留尾部 100 条而不是校验失败。
- **role 宿主关闭超时**（`14797d3e`）：Manager 调了图记忆工具，宿主里自启了 MCP server 又要在结束时写回，
  5 s 宽限期不够。role 宿主现在不带记忆（委派方会沉淀 campaign 结果），宽限期 30 s。
- 运维教训：`pkill -f` 会碰到同机其他用户的 deepswe 进程（改为 `-u "$USER"`）；清理目录时误删过
  chrys-home（改为只清任务目录）；rsync 整棵树会带上本机的 vendored 二进制（排除 `vendor/ripgrep/*`）；
  Python 重定向到文件要 `PYTHONUNBUFFERED=1` 否则分片日志为空。
- 13:05 起 ali-server 的 sshd 在密钥交换后不再应答（TCP 22 通、ping 正常），与 20 个并发 agent 同时
  运行相符，疑为内存/IO 压力；后台每分钟重试，进去后先看 `free`/`vmstat` 再决定是否降并发。

### 09-04 下午（限流后的 lh 跑批、PR #696、LoLBench 接入）

- **P0 跑满预算整轮无回复**（`6b638913`）：drizzle/koota 两题 P0 用满 5400 s 后工作流按"中断"结束、rc=1，
  工作区里 90 分钟的修改被丢掉。现在预算只结束 P0：中断后等它停稳，把部分基线当作 P0（带 warning），澄清与修复照常。
- **选择器漏评即整体失败**（`07a27a21`）：mashumaro 题选择器两轮评审了 29 个候选中的 24 个，剩下 5 个没评就把整次
  澄清判为 `selector_failed`、晋升 P0。现在漏评的候选记为 reject 并记 warning；未知/重复 id 仍然失败。
- **sweep 的仓库归属读错键**（`a771bb68`）：会话信封记录的是 `meta.primary_cwd`，sweep 读 `meta.cwd`，
  所以所有被扫描沉淀的会话都标成 "general"，按仓库召回自然找不到。报告生成器同样修正。
- **调度器计数把 grep 自己算进去**：`ps | grep -o "…--offset [0-9]*"` 会匹配 grep 进程自己的参数（`[0-9]*` 可空），
  3 个在跑被数成 4，排队的分片永远起不来；改为 `grep deepswe_runner.py | grep -o -- "--offset [0-9][0-9]*"`。
  同类教训：`pkill -f` 的模式若出现在调用它的 ssh 命令行里，会把自己的会话杀掉——模式要用 `^bash /abs/path` 锚定。
- **Windows CI**（PR #696）：`/usr/bin/git` 写死（改 `shutil.which`）；用文本写入后比较字节数（改 `write_bytes`）；
  PACT worktree 路径在 pytest 临时目录下超过 Windows 限制（`'$GIT_DIR' too big`，POSIX-only）；rg 探针测试用
  shell 脚本当二进制（POSIX-only）。macOS 的 `test_stopped_script_returns_promptly` 是上游自带的环境性用例，
  在上游 main 本地同样失败，与本分支无关。
- 远端 Neo4j 曾在 IO 风暴中静默退出（日志停在 Started），`neo4j status` 在无 JDK PATH 的非交互 ssh 里会误报
  "We cannot execute"；核对时先 `export PATH=$HOME/lhs/jdk21/bin:$PATH`。
- **LoLBench 接入的"前 20 题"选错**（`fe3dea0a`）：`gen_instances.py` 按目录名排序取前 20，而 DeepSWE runner
  按 `tasks/manifest.json` 顺序；两套 20 题几乎不重叠。改为 manifest 顺序后重跑。容器内 agent 以 root 写
  `agent_out/`，普通用户删不掉，用 alpine 容器删。
- lh 被杀过的运行留下的工作区会让 `--resume` 后的重跑撞 `index.lock`（langchain 就是这样失败的）；
  排队分片对应的 5 个旧工作区已清理，langchain 的失败记录删掉让它重跑。

## 7. 交付状态（09-03 收尾）

- 36 个 task 全部完成并 commit 在本地 `integration/long-horizon-suite`（`origin/main..HEAD` 共 96 个
  commit）。**未 push**。
- 全量 CI 每个 task 与每条修复后都跑过：最终 `19373 passed, 11 skipped`，只 deselect 那条既有环境失败。
- `/code-review high` 的 7 条 findings：5 条已修（`a81e602d`）、1 条由 Task 26 修掉（wheel 打包）、
  1 条经核实不是缺陷（restore 不删非空目录）。
- 收尾时对 M3–M5 新代码补做第二轮评审（第一轮早于这些代码）：4 条全部已修并补测试，
  见偏差记录"评审二轮"五行。
- 第三轮：09-03 派出的 5 个评审 teammate 全部死于 API 529，从 transcript 抢救出 31 条 findings。
  去重后 30 条：**26 条已修并带回归测试**，3 条判定不宜盲修、写入 known-gaps（role 运行时孤儿、
  重试不更新图、崩溃恢复轮次无 mutation 账本），1 条核实后不是缺陷（repair 阶段抑制 `Error` 是刻意的
  ——工作流会提升 P0 并交付一个有效答复；其中真正有问题的"回滚也失败"子情形已单独修复）。
  见"评审三轮"各行。
- DeepSWE 前 20 题定位基线（Task 26 之前的 subprocess 路径）：R@1 0.135 / R@5 0.386 / R@all 0.477，
  20/20 至少命中一个 gold 文件。注意事项与已知缺口见
  `docs/design/long-horizon-known-gaps.md` §5。
- §5 的两项已于 09-04 闭环：初始图接的是 CAPBench selected-Harbor（2525 条 canonical rule），
  verify_command 由我决定并验证。`chrys debug router` 对强带需求现在给出
  `pact_ready=True` / `plan localization=True clarification=True pact=True`——整条长程链路可达。
