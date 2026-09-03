# Chrys 需求澄清合入说明

## 1. 文档状态与范围

本文记录 `feature/requirement-clarification` 分支中已经落地的产品实现，描述当前代码的真实行为。
实施前方案仍保存在工作区上层的 `PLAN-0831.md`；若方案与本文不一致，以本文和当前代码为准。

本次合入没有把 `variant_clarification` 的实验 runner 原样嵌入 Chrys，而是保留其 Route A
语义，将产品能力拆入 Chrys 既有的 profile、service、orchestration、event、TUI、session 和
trajectory 边界。DeepSWE 评估、批量调度和报表仍未进入 `src/chrys` 发行路径；它们现在作为独立实验层放在
[`evaluation/requirement_clarification/`](evaluation/requirement_clarification/README.md)，通过公开的
`chrys run` 入口验证这里描述的产品行为。

当前功能是“内部、仓库证据驱动的需求补充”，不是“主动向用户提问并等待回答”的交互式澄清。

## 2. 核心语义

一次开启需求澄清的新 turn 严格按以下顺序运行：

```text
用户需求 Rᵥ
  → 保存历史 H0，并冻结实现前工作区 S0
  → 普通 Chrys agent 根据 Rᵥ 生成 baseline P0
  → 冻结 P0
  → 澄清 side calls 只根据 Rᵥ + H0 有限投影 + S0 生成 ΔR
  → 保留 P0 文件、恢复 H0 历史，以 Rᵥ + ΔR fresh repair
  → 成功时提交 P1；失败时恢复并提升 P0
```

这里有四条不可破坏的隔离约束：

1. P0 必须先于 ΔR；P0 和 ΔR 不并行。
2. ΔR 不能读取 P0 文件、P0 transcript、repair 状态或评测结果。
3. repair 从 P0 文件系统开始，但不继承 P0 transcript 或 provider continuation handle。
4. 用户原文需求 Rᵥ 始终是最高权威；ΔR 只能补充仓库中有证据的实现约束，不能创造新需求。

## 3. 代码分层

| 层 | 主要文件 | 职责 |
| --- | --- | --- |
| Agent profile | `service/profiles/agents/schema.py`, `loader.py` | 声明和加载 `requirement_clarification.enabled` |
| 领域类型 | `service/requirement_clarification/types.py` | revision、phase、proposal、selection、result 的类型契约 |
| 快照 | `service/requirement_clarification/snapshot.py` | 捕获、验证、恢复和销毁 S0/P0 |
| 确定性证据 | `service/requirement_clarification/evidence.py` | 从冻结源码和可达祖先中构造有界 evidence packet |
| Prompt 策略 | `service/requirement_clarification/prompts.py` | 固定 3+1 策略、schema、上限和版本号 |
| 模型边界 | `service/requirement_clarification/model.py` | 创建 fresh、无状态、只读的澄清 agent |
| 澄清服务 | `service/requirement_clarification/service.py` | 并行 proposals、selector、清洗和 ΔR 渲染 |
| 私有产物 | `service/requirement_clarification/artifacts.py` | owner-only workflow record、H0/P0 transcript 和私有结果 |
| Turn 编排 | `orchestration/engine/run/requirement_clarification.py` | Route A 状态机、repair、回退和 amendment |
| Turn 入口 | `orchestration/engine/run/runner.py` | 在 fresh turn 上按 profile 开关选择普通或澄清路径 |
| 用户输入协调 | `orchestration/engine/run/coordinator.py` | 将 clarification/repair 期间的新输入转为 requirement amendment |
| Executor 展示语义 | `orchestration/engine/executor.py` | 标记 P0 provisional、保存/恢复历史、提升 fallback |
| Event/TUI | `foundation/events/types.py`, `app/tui/screens/main/` | phase 状态、provisional P0 和最终消息展示 |
| Session 恢复 | `service/session/lifecycle.py` | 崩溃后验证工作区并安全提升完整 P0 |
| Rollback | `orchestration/engine/rollback.py` | 回滚 turn 时裁剪对应 workflow artifacts |

## 4. 开启方式

功能默认关闭，只通过原生 AgentProfile 开启：

```yaml
requirement_clarification:
  enabled: true
```

首版只公开 `enabled`。proposal 数量、selector 数量、prompt、阈值和输出上限由代码及
`STRATEGY_VERSION` 共同拥有，不允许不同 profile 静默改变实验语义。

ACP agent profile 当前不能开启该能力；loader 会直接拒绝这种配置。普通原生 profile 未开启时，
`TurnRunner.run_fresh()` 完整走原有 `_run_fresh_standard()`，不进入任何新阶段。

## 5. 一次 turn 的详细执行过程

### 5.1 建立 H0 和 S0

workflow 在普通 agent 写文件前完成以下操作：

- 创建本轮 `workflow_id` 和 revision 1，revision 中保留用户原文；
- 保存 Executor 的完整 H0 checkpoint；
- 从历史中提取最多 40 条 user/assistant 文本，截断到最后 12,000 字符，作为非权威背景；
- 排除带 Chrys 内部 kind 标记的历史内容，避免把内部产物再次喂给澄清器；
- 对所有 workspace roots 和显式 reference files 捕获 S0；
- Git root 同时冻结 turn-start HEAD 及其可达祖先 bundle；非 Git root 只提供冻结文件视图。

快照恢复层保存完整字节，澄清视图则是只读、有界视图：

- 每个 root 最多 50,000 个条目；
- 总恢复内容最多 512 MiB；
- 大于 50 MiB 的单文件不暴露正文，只暴露元数据；
- 符号链接以链接本身记录，不允许借其逃出 workspace root；
- 读取过程中发现文件身份、大小或 mtime 变化会判定快照不稳定并失败；
- VCS 元数据、依赖、缓存和常见构建目录按 Chrys 既有 exclude 规则排除。

如果 H0/artifact 初始化或 S0 捕获失败，本轮发布 degraded warning，然后退回普通 Chrys turn；此时
不会产生一个残缺的澄清流程。

### 5.2 生成 P0

workflow 复用当前 Executor 调用 `_run_fresh_standard(..., finalize=False)`：

- 使用原 AgentProfile 的 instructions、tools、skills、approval 和 active model；
- 执行过程与普通 Chrys turn 相同；
- 只是暂缓逻辑 turn 的最终提交；
- P0 的最终文本被标记为 `is_provisional=True`，TUI 显示为 baseline candidate，但不会结束 turn。

P0 成功后会保存：

- `p0_text`；
- P0 完整 history checkpoint；
- P0 私有 transcript；
- P0 workspace snapshot；
- P0 阶段消费的用户 injections。

P0 没有产生 diff 也仍然可以进入澄清。只有 P0 技术失败或被用户中断时，workflow 才不会启动
ΔR，并直接按失败/中断状态 finalize。

### 5.3 生成 ΔR

`ClarificationService` 先从 S0 构造有界的确定性 evidence packet：

- 从 Rᵥ 提取最多 10 个高信号搜索词；
- 在每个冻结 root 中搜索当前 frozen HEAD 或非 Git 文件视图；
- Git root 可额外搜索 frozen HEAD 的有限可达祖先；
- 搜索数量、每词命中、祖先 commit 数量和 evidence 总字符数均有固定上限。

随后并行执行三个独立 proposal：

1. ownership、公共接口与扩展点；
2. 数据流、控制流、状态和消费者；
3. 兼容性、错误、边界与集成陷阱。

三个 proposal 返回严格 Pydantic schema。每个候选携带 category、statement、confidence、basis、
contract cell、decision impact、evidence 和 risk；这些属于私有选择材料，不直接给 repair agent。

proposal 完成后执行一个 selector。selector 最多选择 5 条 guidance point。确定性清洗还会移除：

- 空内容和重复内容；
- 已经包含在用户原需求中的复述；
- 带占位符或过程性元数据的内容；
- 置信度低于 0.75 的条目；
- 超过 5 条或使最终文本超过 1,400 字符的条目。

最终公开给 repair 的 ΔR 只有简短的 `Repository implementation guidance` 列表，不包含 proposal
证据、风险、置信度或 selector 推理。

### 5.4 澄清 side call 的隔离

每个 proposal 和 selector 都通过 `ChrysClarificationModel` 创建一个新的 Agent：

- 使用本轮锁定的 active `ModelProfile`；
- 使用独立的 route session id，并把主 session id 作为 parent；
- 显式关闭 provider store/continuation/background 能力；
- 不继承 P0 provider conversation handle；
- workspace 指向 S0 的冻结 view，而不是 live P0 workspace；
- 只加载 `filesystem.read` 与 `search` 类别中的只读工具；
- 不开放 shell、写文件、测试或网络工具；
- 强制结构化响应；
- token usage 汇入主 workflow 的 side-call usage。

因此，clarifier 无法通过正常 Chrys 能力看到或修改 P0。当前的全局 model lock 同样作用于这些
side calls；如果模型 wire identity 不匹配，client 会在网络请求前失败，随后本流程按 P0 fallback。

### 5.5 在 P0 上 fresh repair

只有当 ΔR 非空且 live workspace 仍与 P0 snapshot 完全一致时才启动 repair。启动时：

- workspace 保持 P0，不恢复成 S0；
- Executor history 恢复到 H0；
- SessionHistoryManager 重新绑定恢复后的 history；
- 清理 P0 的 intermediate text 和 consumed-injection 运行状态；
- 重置 Executor counters 和 batch id；
- requirement phase 切换为 `repair`；
- 将 Rᵥ 设置为本轮用户消息；
- 通过内部 system reminder 注入 ΔR。

repair reminder 明确告知 agent：baseline 已经存在，必须直接检查当前文件，只做满足原需求和
repository guidance 所需的修复，不能假设 P0 transcript 或 rationale。

repair 成功后，最终 history 是 H0、Rᵥ 和 repair transcript；P0 transcript 不进入后续模型历史。
workflow 只在 repair 完成后统一 finalize，P1 是正常情况下唯一的 final answer。

## 6. 用户 amendment 和 interrupt

workflow 仅在 `clarification` 和 `repair` 阶段接受 requirement amendment：

- 用户消息按原文追加到 `RequirementRevision.messages`；
- revision number 递增；
- 消息继续经过 Chrys 原有的 prompt admission、image rejection 和 hook 流程；
- clarification 期间 revision 改变时，旧 selector 结果被丢弃并重新执行澄清；
- repair 期间收到 amendment 时，当前 Executor 被中断；
- workflow 恢复 P0 workspace 和 P0 history，再用新 revision 重新生成 ΔR 和 repair。

用户 interrupt 会设置 workflow stop 标记并中断活动 Executor。如果 P0 已完整存在，后续逻辑优先
保留 P0；如果中断发生在 P0 生成期间，则不进入澄清。

## 7. 回退和冲突语义

| 情况 | 当前行为 |
| --- | --- |
| runtime、artifact 或 S0 初始化失败 | 发布 warning，退回普通 Chrys turn |
| P0 失败 | 不执行 ΔR，按失败状态 finalize |
| P0 被中断 | 不执行 ΔR，按中断状态 finalize |
| P0 transcript 或 P0 snapshot 无法保存 | 提升已经完成的 P0 |
| proposal/selector/schema 失败 | 提升 P0 |
| 最终 ΔR 为空 | 提升 P0 |
| repair 前 workspace 不再匹配 P0 | 不覆盖现场，标记 conflicted 并提升 P0 文本 |
| repair 失败或被中断 | 恢复 P0 workspace/history，提升 P0 |
| amendment 使 repair 失效 | 恢复 P0，再按新 revision 重算 |
| P0 恢复本身失败 | 标记 conflicted，保留可调查现场 |

“提升 P0”表示清除失败 repair 的 Executor 状态，把保存的 `p0_text` 作为一次真正的 final
`AgentMessage` 发布，然后执行普通 turn finalization。它不是根据 benchmark 分数决定的逐任务回退。

## 8. Event 与 TUI 展示

新增的主要事件语义是：

- `RequirementClarificationPhaseChanged`：携带 workflow id、phase、revision、detail 和 terminal；
- `AgentMessage.is_provisional`：标记已经完整但尚不能结束逻辑 turn 的 P0；
- `AgentMessage.requirement_phase`：区分 initial implementation 与 repair；
- presentation accept/reject 事件：提交或撤销 response attempt 中的 provisional segments。

TUI 会显示当前阶段状态：冻结 workspace、构建初始实现、澄清仓库需求、修复初始实现和最终化。
proposal、selector、私有证据及其推理不会显示到普通聊天区。

## 9. 私有产物、清理与恢复

每个 workflow 的 owner-only 产物目录为：

```text
<session_dir>/requirement_clarification/turn_<n>/
```

运行中可能包含：

```text
workflow.json
h0.private.json
initial_implementation.private.json
clarification.private.json
summary.json
s0/
p0/
```

`workflow.json` 原子记录 phase、terminal、revision、配置 fingerprints 和快照引用。proposal、selection、
usage 和 ΔR 保存在 `clarification.private.json`，不会混入普通 session history。

正常终态会销毁 S0/P0 的大体积恢复快照。Session rollback 会删除被回滚 turn 之后的 workflow
artifacts。

当前崩溃恢复策略是“安全提升完整 P0”，不是从中断点继续运行 ΔR/repair：

1. session restore 查找最新的非终态 workflow；
2. 只有 P0 snapshot 和 live workspace 字节一致时才继续；
3. 从私有 P0 transcript 恢复 history；
4. 清空 provider service session id，避免续接废弃的远端响应；
5. 将该 turn 作为已恢复 P0 提升，并把 workflow 标记为 recovered；
6. 如果 workspace 已变化或 artifact 无效，则不修改 workspace，标记 recovery conflict。

## 10. 当前明确未实现的部分

以下内容不在现有 Chrys 产品路径中：

- 自动生成澄清问题并阻塞等待用户回答；
- Route B 或先生成 ΔR 再生成 P0；
- baseline/candidate 批量 runner；
- fixed-P0 配对实验；
- DeepSWE、SWE-ReBench 或其他 benchmark adapter；
- verifier 调度、统计显著性、成本报表和实验重试策略；
- 根据评测分数决定是否采用 P1；
- ACP agent profile 上的需求澄清。

实验层仍应放在独立的 `evaluation/requirement_clarification/` 一类目录，通过稳定产品接口调用，
不应把 benchmark 特有逻辑写回 turn workflow。

## 11. 主要提交

```text
1f825a6 feat: add requirement clarification foundation
2b4b8ad feat: integrate clarification into turn execution
a8c4baf feat: recover interrupted clarification workflows
c23ab12 fix: isolate clarification history background
```

后续的 `6ff9b3c feat: add fail-closed model lock` 是运行时模型约束；它不是澄清算法的一部分，但会
统一约束主 agent、clarification side calls、repair、sub-agent、judge 和其他 LLM client。

## 12. 现有验证入口

需求澄清的主要测试位于：

```text
tests/service/requirement_clarification/test_profile.py
tests/service/requirement_clarification/test_snapshot.py
tests/service/requirement_clarification/test_service.py
tests/service/requirement_clarification/test_recovery.py
tests/orchestration/engine/test_requirement_clarification_workflow.py
```

这些测试验证的是产品语义和故障边界，不包含真实模型实验或 benchmark 评分。

## 13. TUI 友好的可视化速查

本节使用纯文本图，不依赖 Mermaid、浏览器或图片渲染，可直接在 Chrys TUI 和普通终端中阅读。

### 13.1 合入架构

```text
                     AgentProfile
          requirement_clarification.enabled
                           |
                           v
+---------------------------------------------------+
| TurnRunner                                        |
|                                                   |
| enabled = false --------> Original Chrys Turn     |
| enabled = true  --------> Clarification Workflow  |
+---------------------------------------------------+
                           |
                           v
+---------------------------------------------------+
| RequirementClarificationWorkflow                  |
|                                                   |
| Owns: H0 / S0 / P0 / revision / phase / fallback |
+---------------------------------------------------+
       |                 |                  |
       v                 v                  v
+-------------+   +----------------+   +-------------+
| Snapshotter |   | Clarification  |   | Executor    |
|             |   | Service        |   |             |
| S0 / P0     |   | 3 proposals    |   | P0 / repair |
| capture     |   | 1 selector     |   | finalization|
| restore     |   | render Delta-R |   |             |
+-------------+   +----------------+   +-------------+
       |                 |                  |
       +-----------------+------------------+
                         |
                         v
+---------------------------------------------------+
| Chrys Existing Infrastructure                     |
|                                                   |
| EventBus | TUI | Session | History | Approval     |
| Tools    | Usage | Trajectory | Rollback           |
+---------------------------------------------------+
```

### 13.2 完整 turn 时序

```text
User sends requirement R1
            |
            v
+-------------------------+
| Phase: SNAPSHOT         |
|                         |
| Save history H0         |
| Freeze workspace S0     |
+-------------------------+
            |
            v
+-------------------------+
| Phase: INITIAL_IMPL     |
|                         |
| Normal Chrys execution  |
| R1 + normal tools       |
| produces baseline P0    |
+-------------------------+
            |
            |  TUI shows:
            |  Baseline candidate (provisional)
            v
+-------------------------+
| Freeze P0               |
|                         |
| Save P0 workspace       |
| Save private transcript |
+-------------------------+
            |
            v
+---------------------------------------------------+
| Phase: CLARIFICATION                              |
|                                                   |
| Inputs: R1 + bounded H0 text + frozen S0          |
|                                                   |
|       +------------+                              |
|       | Proposal 1 |-- ownership / interfaces     |
|       +------------+                              |
|       +------------+                              |
|       | Proposal 2 |-- data / control flow        |
|       +------------+                              |
|       +------------+                              |
|       | Proposal 3 |-- errors / compatibility     |
|       +------------+                              |
|             |                                     |
|             +-----------+-----------+             |
|                         v                         |
|                   +----------+                    |
|                   | Selector |                    |
|                   +----------+                    |
|                         |                         |
|                         v                         |
|                 validated Delta-R                 |
+---------------------------------------------------+
            |
            v
+---------------------------------------------------+
| Phase: REPAIR                                     |
|                                                   |
| Workspace = P0                                    |
| History   = H0                                    |
| Prompt    = R1 + Delta-R                          |
|                                                   |
| P0 transcript is NOT inherited                    |
+---------------------------------------------------+
            |
            v
+-------------------------+
| Phase: FINALIZING       |
|                         |
| Commit repair history   |
| Publish final P1        |
| Clean S0/P0 snapshots   |
+-------------------------+
            |
            v
          P1 final
```

核心顺序可以压缩为：

```text
P0 is generated first
        |
        |  P0 is hidden from clarification
        v
Delta-R is generated only from S0
        |
        |  Delta-R is injected into a fresh history
        v
Repair starts from P0 files
```

### 13.3 信息可见性矩阵

```text
+----------------+----------+----------+----------+
| Information    | P0 Agent | Clarifier| Repair   |
+----------------+----------+----------+----------+
| User R         | YES      | YES      | YES      |
| History H0     | YES      | bounded  | YES      |
| Frozen S0      | live     | YES      | ancestor |
| P0 files       | creates  | NO       | YES      |
| P0 transcript  | creates  | NO       | NO       |
| Private props  | NO       | YES      | NO       |
| Final Delta-R  | NO       | creates  | YES      |
| Shell/write    | YES (*)  | NO       | YES (*)  |
| Network tools  | profile  | NO       | profile  |
+----------------+----------+----------+----------+

(*) 仍然受原 AgentProfile 和 approval policy 管理。
```

Clarifier 和 repair 读取的是两个不同的 workspace view：

```text
Live workspace
     |
     +---- snapshot before P0 ----> S0 read-only view
     |                                  |
     |                                  +--> Proposal agents
     |                                  +--> Selector agent
     |
     +---- P0 modifies here
                  |
                  +----------------------> Repair agent
```

### 13.4 TUI 展示

```text
User prompt
    |
    v
[Freezing the turn workspace...]
    |
    v
[Building an initial implementation...]
    |
    v
+---------------------------------------+
| Baseline candidate (provisional)      |
|                                       |
| P0 response text                      |
+---------------------------------------+
    |
    v
[Clarifying repository requirements...]
    |
    v
[Repairing the initial implementation...]
    |
    v
+---------------------------------------+
| Final answer                          |
|                                       |
| P1 response text                      |
+---------------------------------------+
```

P0 的 presentation event 不会提前终止逻辑 turn：

```text
is_provisional = true
is_final       = false
phase          = initial_implementation
```

### 13.5 用户 amendment

```text
Initial requirement
       R1
        |
        v
  Clarification rev=1
        |
        | user injects R2
        v
   Rv = [R1, R2]
   revision = 2
        |
        +--> discard stale Delta-R rev=1
        |
        +--> regenerate Delta-R rev=2
        |
        +--> repair from clean P0
```

如果 amendment 出现在 repair 期间：

```text
Repair rev=1 running
        |
        | user amendment R2
        v
Interrupt repair rev=1
        |
        v
Restore P0 workspace
Restore P0 checkpoint
        |
        v
Regenerate Delta-R rev=2
        |
        v
Start repair rev=2
```

### 13.6 P0 fallback

```text
                         +--> Delta-R empty --------+
                         |                          |
                         +--> Clarifier failed -----+
                         |                          |
P0 completed ------------+--> Schema invalid -------+--> Promote P0
                         |                          |
                         +--> Repair failed --------+
                         |                          |
                         +--> Repair interrupted ---+
                                                    |
                                                    v
                                      P0 becomes final answer
```

```text
S0 failed  ---> degrade to original Chrys turn
P0 failed  ---> finish as failed
P0 success ---> always retain P0 as fallback
```

### 13.7 崩溃恢复

```text
Process crashes after P0
          |
          v
Session restore
          |
          v
Find non-terminal workflow.json
          |
          v
Load saved P0 snapshot
          |
          +--- live workspace != P0 ---> CONFLICTED
          |                              leave files untouched
          |
          +--- live workspace == P0 ---> restore P0 history
                                         clear remote continuation
                                         promote P0 safely
```

### 13.8 产品与实验边界

```text
variant_clarification
        |
        | concepts adapted
        v
+-------------------------------------+
| Chrys product implementation        |
|                                     |
| S0 -> P0 -> Delta-R -> repair       |
| profile / events / TUI / recovery   |
+-------------------------------------+

+-------------------------------------+
| Experiment layer: NOT implemented   |
|                                     |
| fixed-P0 paired runs                |
| baseline vs candidate               |
| DeepSWE evaluation                  |
| verifier / reports / statistics     |
+-------------------------------------+
```
