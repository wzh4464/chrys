# Chrys 需求澄清使用与产物指南

本文是面向用户、下游 Agent 和集成代码的操作入口。它说明如何开启 Chrys 的需求澄清、运行时由谁生成
什么，以及应从哪里读取最终需求单和 PACT 输入。算法、恢复和编排的深层设计见
[`REQUIREMENT-CLARIFICATION-INTEGRATION.md`](REQUIREMENT-CLARIFICATION-INTEGRATION.md)。

## 1. AI 快速规则

如果你是第一次接触该功能，先遵守以下规则：

1. 功能只在原生 Chrys AgentProfile 中通过 `requirement_clarification.enabled: true` 开启。
2. 当前流程是静默的 repository-grounded clarification，不会向用户提问或等待确认。
3. 最终完整需求单读取 `05-outcome/clarified-requirement.md`。
4. 只想比较原始需求与补充信息时，读取 `05-outcome/clarified-requirement-delta.md`。
5. PACT 消费者读取 `06-pact-input/goal-contract.json` 和 `initial-plan.json`，并先确认
   `generation.private.json` 的 `status` 为 `generated`、内容 hash 匹配。
6. `candidates/`、`decision/`、transcript 和根目录 `*.private.json` 是私有审计/恢复材料，不是最终需求。
7. 不要编辑 `workflow.json`、`h0.private.json`、`s0/` 或 `p0/`；它们由恢复状态机拥有。

## 2. 如何开启和运行

功能默认关闭。在 TUI 中按 **F2** 编辑一个原生 Agent profile，或编辑
`~/.chrys/agents/<agent>.yaml`（Windows 为 `%APPDATA%\chrys\agents\`），加入：

```yaml
requirement_clarification:
  enabled: true
```

完整可选配置：

```yaml
requirement_clarification:
  enabled: true
  strategy: legacy-v1-stabilized
  reuse_workspace_as_p0: false
  clarification_only: false
  clarification_timeout_seconds: 1800
  initial_timeout_seconds: 5400
  repair_timeout_seconds: 5400
```

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `enabled` | `false` | 是否对 fresh turn 启用需求澄清 workflow |
| `strategy` | `legacy-v1-stabilized` | 澄清算法；可切到 `legacy-v1-exact` 复现实验历史 v1 |
| `reuse_workspace_as_p0` | `false` | 为 `true` 时跳过 initial trial，把当前 workspace 当作 P0 |
| `clarification_only` | `false` | 为 `true` 时落盘澄清与 PACT 后以 P0 收尾，不启动 repair |
| `clarification_timeout_seconds` | `1800` | clarification 和 PACT 生成各自使用的 side-phase 超时 |
| `initial_timeout_seconds` | `5400` | P0 initial trial 超时 |
| `repair_timeout_seconds` | `5400` | P1 repair 超时 |

然后选择该 profile，正常使用 TUI，或执行：

```bash
chrys run "<requirement>" --agent <profile-name>
```

每个启用该功能的新 turn 都自动运行完整流程。外部 ACP agent profile 不能启用该能力；`acp:` profile
会被 loader 拒绝。PACT 文件只会生成并落盘，Chrys 当前不会自动启动 PACT。

## 3. 工作原理

图例：

```text
[MAIN]  主 Chrys Agent / Executor，可使用 profile 配置的 tools 和 skills
[SIDE]  内部 fresh headless Agent + active LLM，只读；调查为普通文本，定稿为 structured output
[RULE]  确定性 Python 规则，不调用 LLM
[SNAP]  WorkspaceSnapshotter
[WRITE] owner-only 原子落盘
```

```text
用户需求 / amendments
          |
          v
+------------------------------+
| [RULE] RequirementRevision   |----> 01-input/
+--------------+---------------+
               v
+------------------------------+
| [SNAP] 冻结 turn-start S0    |
+--------------+---------------+
               v
+------------------------------+
| [MAIN] Initial Trial         |----> 02-initial-trial/
| 根据原始需求生成 baseline P0 |
+--------------+---------------+
               v
+------------------------------+
| [RULE] 收集 S0 evidence      |
| frozen repo + bounded history|
+--------------+---------------+
               v
+------------------------------+
| [SIDE] 3 个 Proposal Agent   |----> candidates/
| 1 ownership/interface        |
| 2 data/control/state         |
| 3 compatibility/error        |
+--------------+---------------+
               v
+------------------------------+
| [SIDE] Selector Agent        |
+--------------+---------------+
               v
+------------------------------+
| [RULE] 清洗并渲染 delta      |----> decision/ + sources/
| 去重 / confidence / 字符上限 |
+--------------+---------------+
               v
+------------------------------+
| [RULE] 拼装完整澄清需求单    |----> deliverable/
| 原始需求 + amendments + delta|
+--------------+---------------+
               v
+------------------------------+
| [SIDE] Goal Contract Agent   |
| 用户 authority -> goal + AC  |
+--------------+---------------+
               v
+------------------------------+
| [SIDE] Initial Plan Agent    |
| contract + S0 + proposals    |
+--------------+---------------+
               v
+------------------------------+
| [RULE] PACT pair validation  |----> 06-pact-input/
| shape / coverage / refs / DAG|
+--------------+---------------+
               v
+------------------------------+
| [MAIN] Fresh Repair          |----> 04-repair/
| P0 files + H0 history + delta|
+--------------+---------------+
               v
+------------------------------+
| [RULE] Outcome               |----> 05-outcome/
| 成功采用 P1，失败恢复 P0     |
+------------------------------+
```

### 3.1 Main Agent、Side Agent、Skill 和 Rule 的边界

| 工作 | 执行者 | LLM | Skills | 工具权限 |
| --- | --- | ---: | ---: | --- |
| P0 initial trial | `[MAIN]` Executor | 是 | 当前 profile skills | 当前 profile tools |
| 三个 proposals | `[SIDE]` fresh Agent | 是 | 不加载 | 同一 session 先调查再定稿；controller 按 focus 审计 search/read、相邻实现面和 evidence 引用，覆盖不足可反馈续跑一次 |
| selector | `[SIDE]` fresh Agent | 是 | 不加载 | 对每个封闭 candidate ID 逐项记录 select/reject 和私有理由，不能改写或新增陈述；首轮全拒会带反馈复核一次 |
| Goal Contract | `[SIDE]` fresh Agent | 是 | 不加载 | read/search；prompt 限定用户 authority |
| Initial Plan | `[SIDE]` fresh Agent | 是 | 不加载 | `filesystem.read`, `search` |
| selection 清洗、delta、需求单 | `[RULE]` | 否 | 否 | 无模型工具调用 |
| PACT shape/coverage/DAG 校验 | `[RULE]` | 否 | 否 | 无模型工具调用；模型重试后仅对漏覆盖 AC 追加复述 Goal Contract 的确定性 mission |
| P1 repair | `[MAIN]` Executor | 是 | 当前 profile skills | 当前 profile tools |
| hash、manifest、落盘 | `[RULE]` | 否 | 否 | owner-only filesystem write |

这些 side agents 是 Chrys 内部的 `Agent.run(stream=False)`，不是 `codex` CLI 子进程，也不是外部
`chrys run` 进程。它们使用本轮锁定的 active ModelProfile，各自拥有无状态 route session，只能读取 S0；
不能读取或修改 P0、运行 shell、执行测试或访问网络。

### 3.2 两种澄清策略

`legacy-v1-stabilized` 是默认生产策略：保留历史 v1 的三个 focus、3 proposer + 1 selector 和最终清洗
阈值，但把调查与 structured synthesis 拆开。调查阶段不提供 proposal schema，controller 要求真实 search/read
轨迹，并分别覆盖 ownership→extension、source→consumer、boundary→compatibility/test。structured synthesis
阶段禁用工具；每个 `current_repo` evidence 必须引用实际读取过的文件，placeholder 会被拒绝。覆盖不足回到
调查阶段续跑，只有 schema/表达错误才在 synthesis 阶段重试。三个 proposer 中只要一个有效即可继续，这个
“一票即可进入 selector”的门槛保留了历史 Bandit 胜题出现过的 `[0, 0, 1]` 候选形态。

`requirement_complete` 仍是合法结果，但不是首轮免调查出口：它必须至少形成 search→read 证据链，并额外读取
第二个相关实现面或执行第二次定向搜索；若读取过两个实现面，最终 proposal 必须引用两者。系统不设置 token
下限，token 只作为诊断指标；低 token 本身不失败，无证据轨迹的快速空结果才会被判定为 proposer collapse。

`legacy-v1-exact` 是复现实验用兼容策略：使用历史 v1 的 prompt、proposal schema、单次 proposer tool loop
和直接 selector 输出，不强制工具调用、不共享 proposer session，也不附加后来的两阶段语义门。它仍使用
当前的 S0 隔离、artifact tree、fallback 和 PACT 落盘，因此不会回滚这些产品能力。模型服务随时间变化仍
可能造成随机差异；该选项保证的是源码协议等价，而不是保证某道题必胜。

## 4. 产物在哪里

默认 session storage 位于：

```text
macOS/Linux: ~/.chrys/sessions/<session-id>/
Windows:     %APPDATA%\chrys\sessions\<session-id>\
```

设置 `CHRYS_SESSION_ROOT_DIR=<root>` 后，session storage 位于 `<root>/sessions/`。每个澄清 turn 的目录是：

```text
<session_dir>/requirement_clarification/turn_<n>/
```

完整结构：

```text
turn_<n>/
├── workflow.json
├── h0.private.json
├── initial_implementation.private.json
├── clarification.private.json
├── summary.json
├── s0/                              # 临时；正常终态删除
├── p0/                              # 临时；正常终态删除
│
├── 01-input/
│   ├── requirement.md
│   ├── workspace-snapshot.json
│   └── manifest.json
├── 02-initial-trial/
│   ├── response.json
│   └── transcript.private.json
├── 03-clarification/
│   ├── candidates/
│   │   ├── proposal-1.private.json
│   │   ├── proposal-2.private.json
│   │   └── proposal-3.private.json
│   ├── investigations/
│   │   ├── proposal-1.private.json
│   │   ├── proposal-2.private.json
│   │   └── proposal-3.private.json
│   ├── decision/
│   │   └── selection.private.json
│   ├── sources/
│   │   └── delta.md
│   └── deliverable/
│       ├── clarified-requirement.md
│       └── manifest.json
├── 04-repair/
│   └── attempts/revision-<n>/
│       ├── response.json
│       └── transcript.private.json
├── 05-outcome/
│   ├── final-response.md
│   ├── clarified-requirement.md
│   ├── clarified-requirement-delta.md
│   └── summary.json
└── 06-pact-input/
    ├── goal-contract.json
    ├── initial-plan.json
    └── generation.private.json
```

## 5. 每个文件的意义

### 5.1 根目录：兼容与恢复

| 文件 | 生成者 | 意义 | 下游是否应直接消费 |
| --- | --- | --- | --- |
| `workflow.json` | `[RULE]` phase 状态机 | workflow、phase、revision、terminal、S0/P0 引用 | 仅监控/恢复 |
| `h0.private.json` | `[RULE]` serializer | P0 前的完整 history checkpoint | 否 |
| `initial_implementation.private.json` | `[RULE]` serializer | 兼容旧恢复逻辑的 P0 transcript | 否 |
| `clarification.private.json` | `[RULE]` aggregate | status、empty reason、investigations、proposals、selection、delta、usage、warnings | 仅审计/evaluation |
| `summary.json` | `[RULE]` | `05-outcome/summary.json` 的根目录兼容副本 | 兼容 consumer 可读 |
| `s0/` | `[SNAP]` | 实现前 workspace 的冻结恢复/澄清视图 | 否；正常结束删除 |
| `p0/` | `[SNAP]` | baseline workspace 恢复点 | 否；正常结束删除 |

### 5.2 `01-input/`：用户 authority 与冻结输入

| 文件 | 意义 |
| --- | --- |
| `requirement.md` | 用户第一条需求逐字保存；后续 amendments 按 revision 顺序追加 |
| `workspace-snapshot.json` | S0/P0 snapshot ID、manifest hash、大小和条目数；不保存正文 |
| `manifest.json` | 当前 revision、消息数量和 requirement 文件路径 |

### 5.3 `02-initial-trial/`：P0 baseline

| 文件 | 意义 |
| --- | --- |
| `response.json` | P0 的 provisional 文本、revision 和状态 |
| `transcript.private.json` | P0 完整私有 history 与 provider session 信息 |

### 5.4 `03-clarification/`：生成过程与唯一 deliverable

| 文件 | 意义 |
| --- | --- |
| `candidates/proposal-*.private.json` | 三个 proposal；含 confidence、basis、risk、evidence |
| `investigations/proposal-*.private.json` | 默认 stabilized 策略下每个 proposer 的工具轨迹、调查/定稿次数、验证错误与 completed/failed 状态；exact 策略不生成这一层 |
| `decision/selection.private.json` | selector 对每个 candidate ID 的 raw 逐项审查（select/reject、私有理由）与服务端物化后的 cleaned 输出；confidence 沿用 proposal；可以合法地拒绝全部候选 |
| `sources/delta.md` | 实际准备注入 repair 的 `Repository implementation guidance` |
| `deliverable/clarified-requirement.md` | 唯一完整澄清需求单：原始需求 + amendments + delta |
| `deliverable/manifest.json` | revision、计数、引用、warnings 和需求单 SHA-256 |

`investigations/proposal-<n>.private.json` 使用 investigation v2 私有格式，记录 focus、coverage status、
required coverage、search queries、inspected paths、每次工具调用的参数摘要、结果 hash/长度、proposer token
统计以及 validation errors。它用于证明探索真实发生，不是下游需求 authority。

不要把 proposal 或 selection 单独当作用户需求。只有 `deliverable/clarified-requirement.md` 是该阶段完整的
人类可读需求单。

### 5.5 `04-repair/`：每个 revision 的 P1 尝试

| 文件 | 意义 |
| --- | --- |
| `attempts/revision-<n>/response.json` | repair 输出及 `succeeded/failed/timed_out/interrupted/invalidated_by_amendment` |
| `attempts/revision-<n>/transcript.private.json` | 对应 repair attempt 的完整私有 transcript |

### 5.6 `05-outcome/`：最终采用结果

| 文件 | 意义 | 推荐用途 |
| --- | --- | --- |
| `final-response.md` | 用户最终得到的 P1，或失败时提升的 P0 | 查看最终 Agent 回答 |
| `clarified-requirement.md` | 原始需求 + amendments + delta，与 clarification deliverable 内容一致 | 下游 Agent 的完整需求输入 |
| `clarified-requirement-delta.md` | 只包含第一条原始需求 + delta，明确排除 amendments | baseline/clarified 对比实验 |
| `summary.json` | outcome、fallback 原因、上述路径及内容 hash | 自动化索引和完整性检查 |

如果没有产生 delta，两个需求单仍会生成。合法空澄清会说明需求已完整或候选被拒绝；澄清失败则明确写明
`degraded`，不会伪装成“需求已完整”。

### 5.7 `06-pact-input/`：静默生成的 PACT 输入

| 文件 | 意义 |
| --- | --- |
| `goal-contract.json` | closed-shape `pact-runtime/goal-contract/v1`：goal、hard AC、non-goals |
| `initial-plan.json` | closed-shape `pact-runtime/initial-plan/v1`：constraints、missions、AC mapping、DAG |
| `generation.private.json` | `generated/failed/skipped` 状态、revision、usage、warnings 和两个文件 hash；澄清 degraded 时为 `skipped` |

PACT Initial Plan 首次未通过 coverage/reference/DAG 校验时会携带确定性错误反馈重试一次。如果第二次
输出唯一的问题仍是漏覆盖 acceptance criterion，RULE 会为每个漏项追加一个直接复述 Goal Contract 的
mission；它不引入新的完成义务。未知引用、重复 ID、环或其他非法结构仍会失败。最终生成失败
仍不会使 repair 失败；`generation.private.json` 会落盘，但两个 canonical JSON 可能不存在或不应消费。
Chrys 只生成这些输入，不自动运行 PACT。
澄清结果和 delta 会先落盘，随后 PACT 使用新的 side-agent 实例生成；PACT 失败只更新 warning/status，
不会清空 delta、改变 selector 结果或阻止已决定的 repair。

## 6. 下游 AI 应该读取哪个文件

```text
要实现完整澄清需求
  -> 05-outcome/clarified-requirement.md

要分析澄清相对原始需求增加了什么
  -> 05-outcome/clarified-requirement-delta.md

要查看最终 Agent 回答
  -> 05-outcome/final-response.md

要启动 PACT
  -> 先检查 06-pact-input/generation.private.json
  -> 再读取 goal-contract.json + initial-plan.json

要审计为什么生成某条 guidance
  -> 03-clarification/candidates/ + decision/

要恢复 session
  -> 让 Chrys 自己读取根目录 workflow/H0/P0；外部 AI 不要修改
```

## 7. 回退、amendment 和生命周期

- P0 失败或被中断：不运行 clarification/repair，按原 turn 状态结束。
- clarification proposal/selector 失败：安全提升 P0。
- PACT 生成或写盘失败：记录 warning，现有 delta/repair 流程继续。
- delta 为空：不启动 repair，提升 P0，但仍生成 outcome 需求单。
- repair 失败、超时或被中断：恢复 P0 workspace/history，再提升 P0。
- clarification/repair 期间收到 amendment：revision 增加，`01-input/requirement.md` 刷新；旧 repair 失效后
  从 P0 重新生成 clarification 和 repair。
- 正常终态：删除大体积 `s0/`、`p0/`，保留结构化 metadata、需求单和审计文件。
- session rollback：删除目标 turn 之后的 requirement-clarification artifact directories。
- 所有产物均位于 owner-only session 目录，并通过原子写入保存。

## 8. 代码与验证入口

| 职责 | 代码 |
| --- | --- |
| Profile schema/loader | `src/chrys/service/profiles/agents/schema.py`, `loader.py` |
| Typed outputs | `src/chrys/service/requirement_clarification/types.py` |
| Prompts | `src/chrys/service/requirement_clarification/prompts.py` |
| Read-only side agents | `src/chrys/service/requirement_clarification/model.py` |
| Proposal/selection/PACT validation | `src/chrys/service/requirement_clarification/service.py` |
| Snapshot | `src/chrys/service/requirement_clarification/snapshot.py` |
| Artifact layout | `src/chrys/service/requirement_clarification/artifacts.py` |
| Turn workflow | `src/chrys/orchestration/engine/run/requirement_clarification.py` |

主要测试：

```bash
uv run pytest \
  tests/service/requirement_clarification \
  tests/orchestration/engine/test_requirement_clarification_workflow.py \
  tests/integration/test_requirement_clarification_evaluation.py \
  -n0
```
