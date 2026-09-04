# 长程套件已知缺口

本文记录 long-horizon suite（auto router + 需求澄清 + 语义定位 + PACT campaign + ContextGraph 记忆）
交付时**已知且刻意留下**的缺口。它不是 bug 列表，而是边界说明：下面每一条都是当前代码的真实行为，
读到它的人不应该假设相反的能力存在。

设计与实现见 [`requirement-clarification-guide.md`](requirement-clarification-guide.md) §3.3、
[`chrys-pact-integration.md`](chrys-pact-integration.md)，以及 `AGENTS.md` 的 Routing / Long-horizon
track / Memory 三条 quick-ref。

## 1. PACT campaign 没有语义取消

`chrys pact` 有 role 级取消（`pact/role_runner.py::cancel_current_turn`，带 cleanup grace），但**没有
campaign 级的"停下并回滚"协议**。

后果：

- 在委派 pass 期间中断该轮，`LongHorizonExtensions` 会把阶段标为 `interrupted` 并终止本轮，ACP 子进程随
  工具调用一起消失；但 campaign 已经写进工作区的改动**留在工作区里**。
- `.pact-io/` 在 `service/mutations/scanner.py::DEFAULT_EXCLUDES` 里，因此 campaign 的输入输出目录
  **不参与 chrys 的 mutation 跟踪与回滚**。中断后需要人工判断 `.pact-io/chrys-pact/<request-id>/` 下的
  残留状态。
- 已经跑完的 mission 无法被"撤销"到 mission 边界；能回到的最近干净点是 repair 结束时的 P1 baseline
  （由 chrys 自己的 snapshot 持有），不是 campaign 的中间态。

想要的行为是一个 campaign 级 cancel token，让 coordinator 在 mission 边界停下并汇报"已完成到哪一步"。
现在没有，所以**中断长程轮次应当视为需要人工检查工作区**。

## 1b. 停不下来的 role 运行时会留在被删掉的 worktree 里

`pact/role_runner.py` 在一个 role turn 的清理超过 `cleanup_grace_seconds`（5 s）时设置
`preserve_runtime`，**跳过** `host.shutdown()`，并由 `_runtime_unresponsive()` **保留**仍在运行的
`turn_task`——这是刻意的：直接 cancel 会卡死在 `ChrysSessionHost` 被 shield 的异步生成器清理里。

代价是：`RoleRuntimeUnresponsive` 向上抛进 pact_core，后者在 `finally` 里
`git worktree remove --force`。于是那个**停不下来的 Chrys agent 仍在以 `ApprovalMode.BYPASS` 运行**，
`cwd` 指向一个刚被 git 强制删除的目录，直到进程结束为止。

没有在本次修复的原因：真正的修法是让 `ChrysSessionHost` 的关闭路径可被有界取消，那是 PACT 运行时的
设计改动，而本机跑不了完整 campaign（缺 `pact.verify_command`，见 §5），改了也验证不了。盲改一个
无法验证的取消路径比记录它更危险。

**运维含义**：看到 `RoleRuntimeUnresponsive` 就应当认为该 campaign 的 worktree 状态不可信，并检查是否
有残留的 chrys 进程。

## 1c. 重试后的经验不会更新图

`contextgraph_deposit` 的沉淀水位线按**全局 turn 号**推进（见 `writeback.py`）。一次 turn 被沉淀之后，
用户重试同一 turn 并成功，turn 号不变（`runner.py` 在重试时不递增计数器），因此水位线已经越过它，
**新的成功轨迹不会被沉淀**——图里留下的是那次失败/中断的版本。

反方向同样不好：若改成按 digest 变化重新沉淀，`source_id` 随之改变，图里就会出现两条覆盖同一份工作的
轨迹（其中一条带 `failed_attempt` 片段），两条都会被 `team_memory_query` 返回。

要正确处理需要 ContextGraph 支持"按 trajectory 身份更新"，属于上游改动；本次刻意不在 chrys 侧用重复
写入来模拟它。

## 1d. 崩溃恢复出来的 P0 轮次没有 mutation 记录

进程在澄清 side-call 期间被杀死后，`_recover_incomplete_requirement_workflow` 会在工作区仍与 P0 一致时
把 P0 的 transcript 提升为一个已完成的轮次。但 P0 pass 是以 `finalize=False` 跑的——**它从不 save**——
所以那一趟的 mutation 行只存在于已经死掉的进程内存里。恢复时 `engine._mutation_tracker` 是从轮次**之前**
的磁盘 state 水合的，随后的 save 又把这份陈旧 tracker 写回去。

后果：那个轮次的改动在工作区里，但账本里没有：diff 界面对该轮次显示为空，`/rollback` 选"撤销改动"会
回退历史却把 P0 的编辑留在工作树里。

没有在本次修复的原因：让它可修需要 P0 pass 中途 save，而引擎里明确写着"主 save 会删除 recovery
sidecar，而 sidecar 在轮次进行中是已提交但未合并的工具交换的唯一持久副本"——用一个更严重的丢失去换这个。
真正的修法是把 mutation 账本与主 save 解耦，属于持久化层的设计改动。

**运维含义**：看到 `requirement_clarification_recovery_conflict` 或"从中断的澄清流程恢复"的提示后，
不要相信该轮次的 diff 视图；用 `git status` 自己看工作树。

## 2. 图不跨主机

ContextGraph 连接来自 `~/.chrys/.env` 里的 `CONTEXTGRAPH_*`，指向**本机部署**的 Neo4j。

- 同一台机器上的所有 chrys 会话（含 sub-agent、PACT role）共享同一张图。
- 两台机器就是两张互不相干的图，没有同步、没有冲突解决、没有合并。跑在另一台主机上的 ACP sub-agent
  写进的是那台机器的图。
- `memory.mcp.enabled` 默认为 `true`，但没有 `CONTEXTGRAPH_NEO4J_URI` 时 overlay 是**惰性**的：不注入
  MCP server，不报错。因此"没配过图的机器"是正常情况而不是故障，所有召回路径在失败时静默返回空。

"团队图谱"目前的含义是"这台机器上这个人的图谱"。多人共享需要一个远端 Neo4j 和一套权限模型，两者都不在
本次交付范围内。

## 3. 初始图谱：已就位（CAPBench selected-Harbor）

不再是空图。本机接的是 CAPBench selected-Harbor 图（2026-08-31 构建），一个 tarball 安装的
Neo4j 5.26 跑在 `bolt://127.0.0.1:7705`：

| 节点 | 数量 |
| --- | --- |
| Fragment | 75 193 |
| Strategy | 2 607 |
| CanonicalRule | 2 525 |
| Trajectory | 1 463 |
| ErrorPattern | 1 076 |

来源是 49 个 CAPBench 任务的 1 463 条 Harbor agent 轨迹，规则由 `deepseek/deepseek-v4-pro` 抽取后按
0.88 阈值去重。建图策略明确写在 manifest 里：**只读 agent 侧证据与 verifier 的 resolved 标志，不读参考
解、隐藏测试或 gold patch**——所以它可以安全地作为先验喂给一个正在解同类问题的 agent。

运行时与接线：

```bash
# 启停（不需要 Docker）
~/Public/codes/CAPBench/contextgraph_selected_harbor_graph/runtime/neo4j-community-5.26.0/bin/neo4j start
~/Public/codes/CAPBench/contextgraph_selected_harbor_graph/runtime/neo4j-community-5.26.0/bin/neo4j stop
```

`~/.chrys/.env` 里的 `CONTEXTGRAPH_*` 指向它。**嵌入模型必须是 `text-embedding-3-large`**：库里的向量
是 3072 维，换一个模型向量通道就只会返回噪声（词法通道仍可用，但融合结果会变差）。

`chrys memory doctor` 五项全绿，health 行为 `canonical_rules=2525, chrys_trajectories=0`——正是初始图
应有的形状：有先验，还没有本机沉淀的经验。

两条仍然成立的注意事项：

1. Neo4j 5.26 只支持 Java 17/21，本机是 Java 26，启动时会打印 unsupported runtime 警告。实测可用
   （读写、向量与全文索引均正常），但这不是受支持配置；真出问题先装一个 JDK 21。
2. 这张图不跨主机（见 §2）。它是这台机器上的图。

## 4. `route` / `campaign_status` 不进图

`contextgraph_deposit.py::TurnExperience` 携带 `route`（`standard` / `long_horizon`）和
`campaign_status`，但 ContextGraph 的 `RawTrajectory` **没有对应字段**。

后果：这两个值只用于在本地决定这条经验的 `success` 标签（跑过 campaign 时以 campaign `completed` 为准，
否则回落到 turn marker），**存进图之后就查不到了**。无法向图提问"给我看所有失败的长程轮次"。

要改这一点需要 ContextGraph 的 schema 扩展，属于上游改动。

## 5. 定位召回率只是基线水平

DeepSWE 前 20 题（deepseek-v4-pro via OpenRouter，reasoning effort high，`chrys locate` 30 分钟超时），
以 gold `solution/` patch 的文件集合为分母做文件级召回：

| 指标 | 值 |
| --- | --- |
| mean R@1 | 0.135 |
| mean R@5 | 0.386 |
| mean R@all | 0.477 |
| 至少命中一个 gold 文件 | 20 / 20 |
| 命中全部 gold 文件 | 1 / 20 |

两条必须一起读的注意事项：

1. **分母偏大**：gold patch 包含 benchmark 自己撰写的测试文件，定位阶段没有理由指向它们，因此真实的
   "该改的实现文件"召回率高于上表。
2. **这是 Task 26 之前的测量**（而 Task 26 的 in-process 模型随后已被上游的 subprocess 定位流水线取代；"after"对照现在是 `deepswe_runner --run-long-horizon` 跑出的整条长程 track，见 `runs/first20-lh`）：跑的是 subprocess 版定位路径，不是后来的 in-process
   `ChrysLocalizationModel`。数值可作为"不比基线差"的参照，不能当作当前实现的评估结果。

结论：候选位置是**提示**，不是答案。这也是它在 delta 之后、在 Initial Plan prompt 里、在 brief 里都被标为
`untrusted; verify before editing` 的原因。任何把它当权威来源的下游改动都是错的。

## 6. 路由是启发式加一次裁决，校准集是我们自己的

`service/routing/classifier.py` 是双语启发式打分 + 五个置信带；只有 `uncertain` 带会花一次
LLM 裁决（`service/routing/llm.py`，受 `guard.py` 的每会话调用上限与熔断保护）。

- 校准门禁是 `tests/service/routing/fixtures/calibration.jsonl`（60 条）+ `gate.json`：当前 precision
  1.000 / recall 1.000，强带零误报，8 条真正歧义的样本落在 `uncertain`。
- **这 60 条是我们写的**，不是现场数据。改任何权重都必须重跑这个门禁，但门禁全绿不等于对真实用户输入
  的分布正确。
- `routing.mode` 是 `ProjectMerge.DENY`：仓库不能把用户机器切到 `always`（每轮强制 campaign = 成本升级）。
  同理 `routing.tiebreaker_model_profile` 不可由项目设定——仓库不该决定用户的机器调用哪个模型。
- 判定发生在 hooks 之前（见 `AGENTS.md` Top gotchas）：改写 prompt 的 `UserPromptSubmit` hook**不会**
  改变已经做出的判定。

不确定时用 `chrys debug router "<需求>"` 干跑，用 `chrys run --route standard` / `--route long-horizon`
一键覆盖。

## 7. CodeGraph 是可选外部二进制，默认不装

语义定位的 CodeGraph 阶段依赖一个外部二进制。安装策略默认 `never`，且启用时**必须**给出固定的
`--codegraph-install-sha256`——早期版本会 `curl | sh`，那是一个已修掉的供应链缺口，不要恢复它。

没有 CodeGraph 时该阶段降级，定位继续跑，只是少一路证据。

## 8. 写回只在 idle 与正常结束时发生

`orchestration/engine/memory_writeback.py::MemoryWritebackWatcher` 在会话空闲
`memory.writeback.idle_seconds`（默认 3600）后落盘，`memory.writeback.on_session_end` 默认在正常结束时
再冲一次。

- 崩溃、`kill -9`、机器断电：这些轮次**没有**被存入。watermark 在 session 文件旁边，所以下一次对同一
  session 的 idle 窗口会补上；但如果那个 session 再也不会被打开，就永远补不上。
- 补救手段是手动的：`chrys memory sweep` 扫描 session 并从 watermark 之后继续存入。
- 写回**遇到第一个失败就停下**并保持 watermark 不前进，这样一轮经验宁可重复尝试也不会被跳过；代价是
  一个持续失败的图会让积压一直增长。

## 9. 交付时的环境性测试失败

`tests/service/skills/test_runner.py::test_stopped_script_returns_promptly` 在本机 `os.waitpid` 上挂住。
在 `origin/main` 的干净 worktree 上**同样复现**，因此是先于本套件存在的环境问题，不是本次改动引入的。
全量跑的时候它被 `--deselect` 掉：

```bash
LANG=en_US.UTF-8 uv run pytest -m "not integration and not gc_calibration" -q \
  --deselect "tests/service/skills/test_runner.py::test_stopped_script_returns_promptly"
```

`LANG` 必须显式设成 `en_US.UTF-8`：TUI 测试断言英文文案，`zh_CN.UTF-8` 环境下会产生上百条假失败。
