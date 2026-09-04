# Chrys 长程任务增强套件：设计文档

日期：2026-09-03
状态：待用户评审（未实施）
基线：`SELab-Leibniz/chrys` `main` = `15c436e7`（v0.21.1 squash）

## 1. 目标

把四条能力分支合入 chrys，并让它们在"长程任务"上协同工作：

| 能力 | 来源分支 | 状态 |
| --- | --- | --- |
| PACT campaign（Worker/Reviewer/Planner/Manager 治理式长程执行） | `origin/feature/pact`（3 commits） | 已是外部 ACP agent `chrys pact-agent` |
| Semantic graph 代码定位 | `origin/feature/semantic-graph`（有效 2 commits，历史无关） | `chrys locate` + `chrys run --semantic-localization` 预处理 |
| Requirement clarification（S0→P0→ΔR→repair，产出 PACT 输入） | `origin/feature/requirement-clarification`（29 commits） | profile 级开关 `requirement_clarification.enabled` |
| ContextGraph 记忆（Neo4j 检索 MCP + 经验沉淀） | 本仓库 `codex/contextgraph-memory`（3 commits） | 静态 `after_turn` hook 写入 |

用户提出的硬性要求：

1. 路由识别到长程任务时自动走增强链路；也可用 slash command 强制路由。
2. 通过 ACP 调用我们的能力（PACT 已是 ACP agent；保持一个进程边界）。
3. ContextGraph 要有动态经验沉淀与召回。
4. PACT 场景下所有 agent（Primary、sub-agent、PACT 各角色）都暴露 memory MCP。
5. 不用静态 hook：是否调记忆由模型自主决定；但每条 session 空闲 1 小时后自动写回图。
6. 图部署在每台本地 host。

不在范围内：`origin/feature/events-ledger`（分析/标注功能，与本需求无关，不合入）。

## 2. 现状要点（决定设计的事实）

- **分层**：`app → orchestration → service → kernel → foundation`，`tests/architecture/test_layering.py` 强制。pact 分支新增同级 tier `pact`（tier 4）。
- **fresh turn 入口**：`orchestration/engine/run/runner.py::TurnRunner.run_fresh`。RC 分支在此按 profile 开关分流到 `RequirementClarificationWorkflow`；长程工作流也从这里分流。路由本身更早，在 `run/coordinator.py::TurnCoordinator._admit_user_message`：那里 FSM 尚未运行、run task 未创建，可以先切换 profile 再开 turn。
- **外部 ACP sub-agent**：profile 带 `acp:` 段即被强制 `sub_agent_only`，由 `orchestration/sub_agents/tools.py::SubAgentTools.register_acp` 注册成工具 `_invoke_acp(prompt)`。子进程 env 带 `CHRYS_ACP_SUBAGENT_DEPTH`（`service/acp_client/spawn.py`），目前**没有最大深度限制**。
- **PACT 契约**（`docs/design/chrys-pact-integration.md`）：Primary 写 `.pact-io/chrys-pact/<request-id>/{goal-contract.json,initial-plan.json}`，向 ACP agent 发一条 JSON prompt `{"schema":"chrys-pact/run-request/v1","contract_path":…,"plan_path":…}`；角色是**进程内** `ChrysSessionHost`（`pact/role_runner.py::_default_host_factory`），使用 `--agent` 指定的 profile，继承其 tools/MCP/skills；`idle_timeout_seconds: 0`。
- **RC 服务层可独立复用**：`service/requirement_clarification/service.py::ClarificationService.clarify / generate_pact_input / validate_pact_runtime_input`，`model.py::ChrysClarificationModel`，`snapshot.py::WorkspaceSnapshotter`，`artifacts.py`（含 `save_pact_generation` 写 `06-pact-input/`）。RC 已支持 `reuse_workspace_as_p0` + `clarification_only`，但那条路径会以 P0 收尾并结束 turn，不能在同一 turn 内继续委派 PACT。
- **Semantic graph**：`service/semantic_search/pipeline.py::localize_requirement` 以子进程跑 `.agents/skills/semantic-search/scripts/*.py`；脚本目录用 `Path(__file__).parents[4]/.agents/...` 解析，**在 wheel/二进制里会失效**；产物默认写到 `<repo>/.semantic-search/`。
- **MCP overlay 已存在**：`AgentEngine(mcp_overlay=…)` / `ChrysSessionHost(mcp_overlay=…)`，`engine._profile_with_mcp_overlay` 只作用于主 agent；sub-agent 在 `sub_agents/tools.py:502` 直接读 `profile.tools.mcp`。
- **Hooks**：13 个事件（`session_start/…/session_end`），无 idle/定时事件；`after_turn` 由 `run/turn_hooks.py` 触发。用户明确不要静态 hook。
- **Profile 切换**：发布 `AgentProfileSwitch` 事件 → `engine._on_profile_switch` 软重启且保留历史；ACP 有 `session/switch_agent` ext method。
- **Slash command**：`app/tui/screens/main/commands.py::MainSlashCommandRegistry.build()` + `SlashCommandActionPort`；ACP 用 `server.py::ext_method` 扩展；headless 用 `chrys run` 参数。
- **Settings**：`foundation/config/settings.py` dataclass 字段 + `spec(key=…, env=…, apply=…, group=…, kind=…, risk=…, project_merge=…)`。
- **Memory 现状**：`service/memory/contextgraph_mcp.py` 暴露 `team_memory_health/query/record`（Bolt 直连 Neo4j，RRF 融合 canonical rule + fragment 两路）；`contextgraph_deposit.py` 读 `session.json`、按 turn 提取 steps、经 `_contextgraph_repository_worker.py` 用 ContextGraph checkout 的 venv 调 `AgentMemory.learn(RawTrajectory)`；`RawTrajectory(instance_id, repo, success, steps, problem_statement, trajectory_id)`。
- **ContextGraph 本地部署**：`~/codes/ContextGraph`（main）与 `Public/codes/ContextGraph-chrys`（`apply_chrys`）；`docker-compose.yml` 提供 neo4j（127.0.0.1:7687 / 7690）、`memory-server`（8010）、litellm（4000）。

## 3. 总体架构

```text
                          ┌──────────────── Primary chrys session ────────────────┐
user ── TUI / ACP / run ──► TurnCoordinator._admit_user_message                  │
                          │        │                                              │
                          │        ▼                                              │
                          │   TurnRouter ──(守卫 > /longrun 覆盖 > profile > 继承 > 启发式/LLM 裁决)
                          │        │ standard                     │ long_horizon │
                          │        │                              ▼              │
                          │        │              AgentProfileSwitch → LongHorizon（历史保留）
                          │        ▼                              ▼              │
                          │  TurnRunner.run_fresh          RC workflow + LongHorizonExtensions
                          │  standard pass                 ① P0 baseline (main executor)
                          │  (RC full flow if              ② clarify ∥ localize → ΔR+定位 → PACT 输入
                          │   profile enables it)          ③ repair → P1 (reminder = ΔR + 定位表)
                          │                                ④ delegation pass +   │
                          │                                   run-request reminder│
                          │                                        │             │
                          │   memory MCP (every agent) ◄───────────┤ model calls │
                          │        ▲                               ▼             │
                          │        │                 chrys_pact ACP sub-agent    │
                          └────────┼──────────────────────────┬───────────────────┘
                                   │                          │ one process boundary
                                   │           chrys pact-agent (ACP server, pact tier)
                                   │              └─ PACT CampaignControlPlane
                                   │                   └─ in-process ChrysSessionHost per role
                                   │                        (Worker/Reviewer/Planner/Manager)
                                   │                        memory MCP via overlay  ─────┐
                                   ▼                                                      ▼
                     ┌──────────── local host ContextGraph ────────────┐   role sessions flush
                     │ Neo4j (Bolt 127.0.0.1) + ContextGraph checkout  │◄── on shutdown
                     │ read: contextgraph_mcp  write: AgentMemory.learn│◄── primary: idle 1h
                     └─────────────────────────────────────────────────┘    / session end
```

原则：**长程 track 跑完整的 P0 → ΔR → repair；定位与澄清由代码并行执行并合并；是否委派 PACT、是否查记忆由模型决定；所有 LLM 调用走 chrys 自己的模型客户端；写回由引擎定时器执行，不用 hooks.yaml。**

## 4. 设计决策

### D1 分支合入策略

新建集成分支 `integration/long-horizon-suite`（基于 `origin/main`），按顺序合入：

1. `feature/requirement-clarification`（`--no-ff` merge）。**剔除** `dist/archived-binaries/proposer-v10-worktree-20260903/`（98 MB 二进制）与 `evaluation/requirement_clarification/profiles/*.yaml` 中的私有模型配置；`REQUIREMENT-CLARIFICATION-*.md` 移到 `docs/design/`。
2. `feature/pact`（`--no-ff` merge）。冲突点：`app/cli/app.py`、`AGENTS.md`、`README.md`、`pyproject.toml`/`uv.lock`。
3. `feature/semantic-graph`：历史无关，`git rebase --onto origin/main e0cd89fa origin/feature/semantic-graph` 后 merge（只带 `e6d46b7b`、`52301e11`）。DeepSWE 报告目录（`docs/…/code-localization.md` × 20）与 `scripts/deepswe_*.py` 移到 `evaluation/semantic_search/`。冲突点：`app/cli/app.py`、`.gitignore`。
4. `codex/contextgraph-memory`（rebase 到集成分支后 merge）。

每步合入后跑 CI 四件套（ruff check --no-fix / ruff format --check / ty / pytest -m "not integration and not gc_calibration"）。

备选：只 cherry-pick 需要的 commit。不推荐——会丢失同事分支的评审历史，且 RC 29 个 commit 之间有 revert 关系。

### D2 路由 = fresh turn 前的分类 + 一次性覆盖

- `TurnRouter`（`orchestration/engine/run/routing.py`）在 `TurnCoordinator._admit_user_message` 里、`user_prompt_submit` 决定之后、run task 创建之前执行，输出 `RouteDecision(track, band, plan, reason, confidence, source, switched_to)`；`TurnRunner.run_fresh` 只消费这个决定来选择标准 pass 或 `LongHorizonWorkflow`。
- 决策顺序：守卫 > slash/ACP/CLI 一次性覆盖 > profile `routing.mode` > 多轮继承 > 分类。分类先跑双语启发式打分（篇幅、步骤链、范围词×变更动词、验收标准、多路径、任务原型），落入五档置信带：`strong_standard / lean_standard / uncertain / lean_long_horizon / strong_long_horizon`。只有 `uncertain` 档才发一次 LLM 裁决，裁决受每 session 限流（20 次）、熔断（连续 5 次失败停 60 s，半开恢复）、5 s 超时、150 token 上限保护，失败一律回落 `standard`。裁决模型默认用本 session 的 active model，因此天然服从 `CHRYS_MODEL_LOCK`。
- 档位映射到分级 plan：`lean_long_horizon` 跑完整 RC（P0 → ΔR ∥ 定位 → repair），不委派 PACT；`strong_long_horizon` 在 repair 之后再加一次 PACT 委派 pass。两档都不便宜，所以进入 lean 档的门槛已高于 opencode 的对应档位，并要求 strong 档零误报。第二维度"工作区就绪度"（`pact.verify_command` 已配置、`chrys_pact` 工具可用）只做 PACT 的否决门，不参与打分；不采用 opencode auto-router 的"仓库并行度取 min"，因为长程 track 的价值在治理与验证而非并行。
- 多轮继承：上一轮决定在指纹未变、未过 30 分钟、无原型翻转、非琐碎跟进时复用；含 PACT 委派的 `strong` 决定永不继承，每次委派都要重新分类或显式覆盖。`/route reroute` 强制重算。
- 可解释与可校准：`chrys debug router "<prompt>"` 干跑打印信号、分数、档位、就绪度；60 条双语标注样本做 CI 门禁（长程精确率 ≥ 0.90，`strong` 档零误报，召回 ≥ 0.50）；每次决定作为 `TurnRouted` 事件进 session 的 `trajectory/events.jsonl`，不写用户仓库。
- 公告与降级：路由到长程时在聊天区/ACP 流打一行 `→ Routing: long-horizon · <reason>`；准备阶段内 `/quick`（或 ACP `session/route_override`）可把本 turn 降级为标准 pass。
- 路由后切换 profile（用户决定）。路由在 admission 阶段（`TurnCoordinator._admit_user_message`，FSM 未运行、run task 尚未创建）完成；长程决定且 `routing.target_profile` 非空时，先经 `AgentProfileSwitch` 同一路径软重启到目标 profile（历史保留，与 `#` 切换一致），再开 turn。内置 `Code` 只带 `routing: {mode: auto, target_profile: LongHorizon}`；增强工具（`chrys_pact`）只在内置 `LongHorizon` 上。不自动切回：后续 turn 留在 `LongHorizon`，按它自己的路由配置决定 standard/long_horizon track；`#Code` 手动切回。全局开关 `routing.mode` 可整体关闭。
- 嵌套防护：`CHRYS_ACP_SUBAGENT_DEPTH > 0` 或全局 `routing.mode=off` 时路由器恒返回 `standard`。PACT 角色 host 的派生 settings 固定 `routing_mode="off"`。

### D3 长程 track = 完整 RC 流程 + 并行定位 + repair 后的 PACT 委派

用户决定保留 P0 与 repair：长程 track 直接复用 `RequirementClarificationWorkflow`（`orchestration/engine/run/requirement_clarification.py`），通过新增的扩展点 `RequirementWorkflowExtensions`（`run/workflow_extensions.py`）挂上 `LongHorizonExtensions`（`run/long_horizon.py`）。RC 本体的分支逻辑不改，只在六个位置插入钩子。

1. 打开 turn、保存 H0、冻结 S0、主 executor 生成 P0：与 RC 完全相同，P0 以 provisional 消息展示。
2. 澄清阶段两路并行（用户要求）：`asyncio.gather` 同时跑 RC 的 `ClarificationService.clarify()`（3 proposals + selector + ΔR + 需求单）和定位分支 `localize_requirement()`。定位读的是 S0 冻结视图而不是被 P0 改过的 live 工作区，产物写到 `<session_dir>/long_horizon/turn_<n>/semantic-search/`。两路各自超时，一路失败不取消另一路；中断、amendment、`/quick` 同时取消两路。
3. 合并：repair 的 system reminder = ΔR + 定位表（不可信，编辑前须核实，≤ 8 条）；`generate_pact_input()` 的 Initial Plan prompt 附带定位提示作为不可信证据；`task brief`（原始需求 + 澄清需求单 + 定位表 + 当前 baseline + 降级说明）写到 `<session_dir>/long_horizon/turn_<n>/brief.md`，Initial Plan 的 constraints 引用它，主 agent 与 PACT 角色都能读。
4. repair：RC 原样执行，产出 P1；失败则 RC 照常提升 P0。
5. 委派 pass：`06-pact-input/` 已生成时，把它复制为 `<workspace>/.pact-io/chrys-pact/<request-id>/…`，用代码拥有的 reminder（brief 路径与摘要、当前 baseline、精确的 run-request JSON、行为规则）再跑第三次 executor pass；模型调用 `chrys_pact`，campaign 从已含 P1（或 P0）的工作区开始，负责验证与完成剩余工作。P1 文本以 `requirement_phase="repair"` 的中间消息展示，第三次 pass 的输出才是本 turn 的最终回答。没有 PACT 输入或 `plan.pact=False` 时不跑第三次 pass，P1 即最终回答。
6. finalizer 检查委派 pass 是否调用了 `chrys_pact`；`routing.long_horizon.require_pact: true` 且未调用时发 `Warning(code="long_horizon_delegation_skipped")`，不强制重跑。turn marker 记录 `baseline`（p1/p0/none）与 campaign 结果。

降级：定位失败 → repair reminder 只含 ΔR，brief 无定位表，继续；澄清 degraded → RC 提升 P0，不生成 PACT 输入，无委派 pass；repair 失败 → RC 提升 P0，若 PACT 输入存在仍可委派（brief 标明 baseline=p0）；委派 pass 失败或中断 → 以 P1/P0 文本收尾，不丢结果。

为什么让 PACT 在 repair 之后接管：RC 的 P0 → ΔR → repair 给出一份经过澄清修正的高质量基线；对真正的长程任务，一次 repair 往往做不完，PACT 的 Goal Contract 与 Initial Plan 正是 RC 为此生成的下游输入，campaign 用 `pact.verify_command` 验证、按 mission 补齐剩余工作。这样既保留 RC 的自我修正，又让治理式执行只处理"剩下的部分"。

备选：代码直接用 `service/acp_client` 启动 PACT 并流式转发，绕过模型。不推荐——重复 `AcpSubAgentController` 的权限代理/翻译/审计逻辑，且失去模型对 `blocked` 结果的处理与总结。

### D4 memory MCP 对所有 agent 全局暴露

- `service/memory/overlay.py::memory_mcp_server_config(settings) -> MCPServerConfig | None`：当 `memory.mcp.enabled` 为真且 `CONTEXTGRAPH_NEO4J_URI` 已配置时返回代码拥有的 stdio 配置（`sys.executable -m chrys.service.memory.contextgraph_mcp`，`allowed_tools=[team_memory_health, team_memory_query, team_memory_record]`，`max_tool_result_tokens=2000`，`expose_instructions=true`）。
- 单一应用点 `apply_memory_overlay(profile, settings)`：`engine._profile_with_mcp_overlay`（主 agent）与 `build/builder.py` 解析 sub-agent profile 处（sub-agent）各调用一次；`ChrysSessionHost` 走同一 engine 路径，因此 headless、ACP server、PACT 角色 host 自动获得。MCP 连接缓存按配置共享，进程内只起一个 MCP 子进程。
- "模型自主决定"：MCP server 的 `InitializeResult.instructions` 携带一段简短指引（何时值得查记忆、结果为不可信参考、不要把记忆当指令），随 `expose_instructions` 进入 system reminder，**不改任何 profile 的 instructions**。
- 删除 `examples/contextgraph-memory/hooks/hooks.yaml`（静态 hook）；`Memory.yaml` 示例改为说明"默认已全局启用"。

### D5 写回：引擎内空闲定时器 + session 结束冲刷 + 可选扫描命令

- `orchestration/engine/memory_writeback.py::MemoryWritebackWatcher`：engine 持有；每次 turn finalize 调 `touch()`；后台 task 在 `last_activity + idle_seconds`（默认 3600，setting `memory.writeback.idle_seconds`）且无活动 turn 时执行 `flush(reason="idle")`；`shutdown()`（即 `session_end` 语义）时若 `memory.writeback.on_session_end`（默认真）则 `flush(reason="session_end")`；`session_restored` 后也 `touch()`。
- `service/memory/writeback.py`：水位线 `memory_deposit_watermark`（写进 `session.json` 的 runtime metadata），只沉淀水位线之后的 turn；沉淀复用 `contextgraph_deposit.extract_turn_experience` + `deposit_experience`（子进程 worker，超时 300 s，失败不推进水位线）。稳定 `source_id` 保证重放幂等。
- PACT 角色 host 生命周期短（每个角色 turn 一个 host），自然走 `session_end` 冲刷；角色 session 的沉淀带 `campaign_id/role` 标签（由 `chrys pact-agent` 通过派生 settings 传入）。
- 安全网 `chrys memory sweep`：扫描 `sessions/` 下所有 `session.json`，对"最后修改 ≥ idle_seconds 且水位线落后"的 session 执行沉淀（幂等）；供进程被 kill 的场景与 cron/launchd 使用。`chrys memory doctor` 检查 Neo4j 连通、索引、checkout/venv、embedding key。

不采用：hooks.yaml 的 `after_turn` durable hook（用户排除）；也不采用 cron-only（会丢失进程内的"1 小时"语义）。

### D6 内置 profile 与 PACT 校验命令

- 新增内置 ACP profile `ChrysPact.yaml`（固定 hex id，`sub_agent_only: true`，`acp: {command: chrys, args: [pact-agent, --agent, Code, --verify-from-settings], result_mode: last_segment, idle_timeout_seconds: 0, max_depth: 1}`）。`command: chrys` 在 `_resolve_acp_spec` 中解析为当前可执行文件（PyApp 二进制或 `python -m chrys`）。
- `chrys pact-agent --verify-from-settings`：读取新 setting `pact.verify_command`（project 层可覆盖，`Risk.HIGH`，受 project trust 域约束）；为空且无 `--allow-unverified` 时启动失败（fail closed）。
- 新增内置主 agent profile `LongHorizon.yaml`（固定 hex id）：instructions 以 `Code` 为底并追加 "Long-horizon delegation" 一节，工具与 `Code` 相同，sub-agents 为 Explore/Plan/General + `{profile: ChrysPact, tool_name: chrys_pact, max_concurrency: 1}`，`routing: {mode: auto}`（`target_profile` 留空）。内置 `Code.yaml` 只加 `routing: {mode: auto, target_profile: LongHorizon}`。`QA` 不变。用户也可以直接 `#LongHorizon` 或 `CHRYS_DEFAULT_AGENT=LongHorizon` 跳过切换。
- `AcpAgentConfig.max_depth`（新字段，默认 1）：`register_acp` 在 `CHRYS_ACP_SUBAGENT_DEPTH >= max_depth` 时跳过注册并发 Warning，杜绝 PACT 角色再启 campaign。

### D7 Semantic graph 打包修正

- 把 `.agents/skills/semantic-search/` 迁到包内 `src/chrys/service/semantic_search/skill/`，`_SKILL_SCRIPT_DIR` 改用 `importlib.resources`；`hatch` 打包包含脚本与 `schemas/`。
- 内置 profile 通过 `skills.paths` 默认挂载该 skill，模型在标准 track 也能自行调用 `run_skill_script`。
- `chrys run --semantic-localization` 保留，但改为设置 `TurnPlan.localization=True` 走 in-turn 预处理（reminder 注入），不再拼接到用户 prompt；`chrys locate` 不变。
- 定位的 LLM 循环（五工具 DFS/BFS）从子进程脚本移入 chrys 进程：`service/semantic_search/localization_model.py::ChrysLocalizationModel` 用 `service/llm/clients.create_client` 建 fresh Agent（仿 RC 的 `ChrysClarificationModel`），脚本只保留确定性阶段（索引、图归一化、CodeGraph、fallback 排序、报告渲染），不再自建 OpenAI 客户端、不再读 model profile YAML、不再持有 api_key。
- 模型选择：setting `semantic_search.model_profile`（空 = session 的 active model）指向 chrys 注册的任一 model profile，填便宜模型即可降本；路由裁决同理用 `routing.tiebreaker_model_profile`。两者都经 `create_client`，`CHRYS_MODEL_LOCK` 自动生效：锁不匹配时定位退化为确定性 fallback，不发模型请求。
- 不多开客户端：engine 持有 session 级 `SideCallClientCache`（按 model profile id 复用 `create_client` 实例），路由裁决、定位、澄清 side call 共用，session 结束时关闭。

### D8 经验沉淀增强

`TurnExperience` 增加 `route`、`campaign`（从 turn marker 的 `_chrys_route` 与 `chrys_pact` 工具结果解析）。`success` 判定：标准 turn 沿用 `status == ok`；长程 turn 以 PACT canonical `completed` 为准（PACT 已做验证）。`problem_statement` 优先取 `05-outcome/clarified-requirement.md`。可选（P2）：澄清前用 `contextgraph_mcp._do_query` 做一次代码侧召回，把 Top-3 策略作为"不可信先验"附给 Initial Plan 生成 prompt。

### D9 配置面

Settings（`foundation/config/settings.py`）：

| 字段 | key | env | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `routing_mode` | `routing.mode` | `CHRYS_ROUTING_MODE` | `auto` | `off/auto/always`，全局上限 |
| `memory_mcp_enabled` | `memory.mcp.enabled` | `CHRYS_MEMORY_MCP` | `true` | 需 `CONTEXTGRAPH_NEO4J_URI` 才生效 |
| `memory_writeback_idle_seconds` | `memory.writeback.idle_seconds` | `CHRYS_MEMORY_WRITEBACK_IDLE_SECONDS` | `3600` | `0` 关闭定时器 |
| `memory_writeback_on_session_end` | `memory.writeback.on_session_end` | `CHRYS_MEMORY_WRITEBACK_ON_END` | `true` | |
| `routing_tiebreaker_model_profile` | `routing.tiebreaker_model_profile` | `CHRYS_ROUTING_TIEBREAKER_MODEL_PROFILE` | `""` | 空 = active model；填便宜模型 profile id |
| `semantic_search_model_profile` | `semantic_search.model_profile` | `CHRYS_SEMANTIC_SEARCH_MODEL_PROFILE` | `""` | 空 = active model；填便宜模型 profile id |
| `pact_verify_command` | `pact.verify_command` | `CHRYS_PACT_VERIFY_COMMAND` | `""` | project 层可设，`Risk.HIGH` |

Profile schema（`service/profiles/agents/schema.py` + loader）：

```yaml
routing:
  mode: off | auto | always          # 默认 off；内置 Code 与 LongHorizon 为 auto
  target_profile: ""                 # 长程决定后切换到的 profile；内置 Code 设为 LongHorizon
  classifier: heuristic | llm | both # 默认 both
  min_confidence: 0.7                # LLM 裁决进入 lean 档的最低置信度（≥ 0.85 进 strong）
  inherit: true                      # 多轮继承
  stale_after_seconds: 1800
  long_horizon:
    localization: true
    clarification: true
    pact_tool: chrys_pact
    require_pact: false
acp:
  max_depth: 1                       # 新字段
```

ContextGraph 连接仍用 `~/.chrys/.env` 的 `CONTEXTGRAPH_*`（已有）。

### D10 事件与 UI

- 新事件（`foundation/events/types.py`）：`RouteOverride(track, one_shot, reroute, plan_localization)`（前端→引擎命令）、`TurnRouted(track, band, plan, reason, confidence, source, inherited, pact_ready, tiebreaker_failure, can_downgrade)`、`LongHorizonPhaseChanged(phase, detail, terminal)`、`MemoryWritebackCompleted(turns, reason, ok)`。
- TUI：`/longrun [text]`、`/quick [text]`（运行中也接受，准备阶段内即降级）、`/route [show|off|auto|always|reroute]`；聊天区一行路由公告，状态栏显示 LH 阶段（复用 RC 的 phase 展示路径）。
- ACP：ext method `session/route_override {sessionId, track, reroute?}`（准备阶段内等同降级）；`chrys/session_runtime` 载荷加 `route`；`TurnRouted` → `chrys/turn_routed` 通知。
- CLI：`chrys run --route auto|long-horizon|standard`；`chrys debug router "<prompt>" [--json] [--full] [-C DIR]` 干跑。

## 5. 长程一轮的时序

```text
user text ──► admit ──► TurnRouter（守卫 > 覆盖 > profile > 继承 > 启发式/裁决）
   ├─ long_horizon 且 target_profile=LongHorizon ──► AgentProfileSwitch 软重启（历史保留）
   ▼
RequirementClarificationWorkflow + LongHorizonExtensions（在 LongHorizon profile 上）
   1 open turn, save H0, freeze S0
   2 [RC initial_implementation]  main executor → P0（provisional）
   3 [RC clarification ∥ localizing] ┌─ clarify: 3 proposals + selector → ΔR → 需求单
                                     └─ localize_requirement(S0 view) → semantic-search/   (并行，各自超时)
   4 [merging]           repair reminder = ΔR + 定位表 ; brief.md ; generate_pact_input(hints=定位) → validate → 06-pact-input/
   5 [RC repair]         main executor (history=H0, workspace=P0, reminder) → P1（中间消息，requirement_phase=repair）
   6 [delegating]        .pact-io/chrys-pact/<request-id>/ ← 06-pact-input ; queue run-request reminder
                         main executor 第三次 pass (memory MCP 可用) ──► chrys_pact(run-request JSON)
                                                          └─ chrys pact-agent (ACP) → campaign from P1 → roles (memory MCP, 可读 brief) → summary
   7 finalize            TurnRouted/phase terminal, route+baseline+campaign 写入 turn marker, watcher.touch()
   ...
idle 1h ──► watcher.flush("idle") ──► deposit turns > watermark ──► AgentMemory.learn ──► watermark 前进
```

## 6. 错误处理与降级矩阵

| 情况 | 行为 |
| --- | --- |
| LLM 分类超时/失败 | `standard`，发 Warning |
| 定位脚本缺失/超时/模型不可用 | 跳过定位；repair reminder 只含 ΔR，brief 无定位表，继续 |
| 澄清 degraded 或 PACT 输入校验失败 | RC 提升 P0；无 run-request，无委派 pass；artifacts 保留 |
| P0 失败或被中断 | RC 语义：不澄清、不 repair，按失败/中断结束；定位若已启动则取消 |
| repair 失败/超时 | RC 提升 P0；PACT 输入存在时仍可委派，brief 标明 baseline=p0 |
| 委派 pass 失败或中断 | 以 P1/P0 文本作为最终回答，phase=degraded |
| `routing.target_profile` 不存在或切换被拒 | 本轮降级 standard，发 `route_profile_switch_failed` |
| `CHRYS_MODEL_LOCK` 与定位模型不匹配 | 定位退化为确定性 fallback，不发模型请求 |
| `ChrysPact` 不可用（无 verify 命令且未 allow-unverified） | sub-agent 注册失败 → 路由器把 `plan.pact=false`，仍做定位+澄清 |
| PACT 返回 `blocked/active` | 模型如实汇报；turn 正常结束；沉淀 `success=false` |
| 用户在澄清/repair 阶段发消息 | RC 的 amendment 语义：revision 递增，重新澄清并同时重跑定位 |
| 用户在委派 pass 发消息 | 普通 injection 进入第三次 pass |
| 用户中断 | 停止当前阶段，按 interrupted finalize |
| 进程崩溃于准备阶段 | 恢复时不自动续跑；artifacts 保留供重发 |
| Neo4j 不可达 | MCP 工具返回健康度错误；写回失败不推进水位线，`sweep` 可补 |

## 7. 测试策略

- 架构：layering（新模块归属）、`test_protocol_compliance`（host 新增 `_route_override`、`_memory_watcher`）、i18n 目录 oracle。
- service：路由分类器（双语启发式用例表、档位边界、就绪度否决；LLM 裁决用 mock client，限流/熔断用注入时钟）、60 条标注样本的校准门禁、overlay（设置开/关、env 缺失）、writeback 水位线与幂等、`memory sweep` 扫描。
- orchestration（engine-driven，`agent_engine` fixture）：`/longrun` → `TurnRouted` → `ProfileSwitched(LongHorizon)` 且历史保留 → RC 的 P0/澄清/repair 阶段事件 + 定位并行完成 → repair reminder 含 ΔR 与定位表 → 第三次 pass 的 reminder 含 brief 与 run-request → fake `chrys_pact` 工具被调用；RC 现有用例在 `NoopExtensions` 下不变绿；降级矩阵每行一个用例；并行性用挂起分支计时断言；`SideCallClientCache` 只建一次 client；idle watcher 用可注入时钟；shutdown 冲刷；`max_depth` 拒绝嵌套。
- app：TUI slash 命令（`App.run_test`）、ACP ext method、`chrys run --route`。
- 集成（`@pytest.mark.integration`）：`examples/long-horizon/e2e_smoke.sh`：本地 Neo4j + 真实模型，跑一条 `/longrun`，断言 `.pact-io` 生成、campaign 摘要、1 h 定时器缩短后写回并可召回。

## 8. 已确认的决定（2026-09-03 用户反馈）

1. 独立内置 `LongHorizon` profile；路由到长程后在 admission 阶段切换 profile，不自动切回（D2、D6）。
2. 空闲 1 h 写回之外，session 正常结束（TUI 退出、`chrys run` 结束、ACP `session/delete`、PACT 角色 host 关闭）也冲刷一次；可用 `memory.writeback.on_session_end=false` 关闭。已确认。
3. PACT 委派由模型调用 `chrys_pact` 完成，准备与合并阶段由代码完成。已确认。
4. 长程 track 跑完整的 P0 → ΔR → repair（用户决定，推翻此前的澄清-lite方案）；PACT 在 repair 之后以委派 pass 接管验证与完成（D3）。标准 track 上完整 RC 仍由 `requirement_clarification.enabled` 控制。
5. `feature/events-ledger` 不合入；RC 分支的 98 MB 二进制与 DeepSWE 报告不进 `src`。已确认。
6. 每台 host 用 ContextGraph `apply_chrys` 分支的 docker-compose 起本地 Neo4j（`chrys memory init`）；初始图留空，`chrys memory init --import <dump>` 预留导入入口，dump 由用户稍后提供。
7. 定位的 LLM 循环移入 chrys 进程，走 chrys 自己的模型客户端；`semantic_search.model_profile` / `routing.tiebreaker_model_profile` 可指向便宜模型；session 级客户端缓存，不多开 client；model lock 自动生效（D7）。
8. 定位与 RC 的澄清阶段并行执行，结果合并进 repair reminder、task brief 与 PACT Initial Plan 提示（D3）。

## 9. 范围外 / 后续

- 图跨 host 同步或导出导入。
- PACT 语义级取消/恢复、多 campaign 并行（PACT MVP 未实现）。
- 交互式澄清（向用户提问）——RC 当前是仓库证据驱动、静默的。
- events-ledger 分支。
