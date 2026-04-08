# User README（Skill 使用说明）

[English](User_README.md) | [中文](User_README_ZN.md)

本文件用于**快速选 skill**。
仅作导航，不参与执行。

## 1. 简略导航

- 不确定怎么开始：`research-workflow`
- 从网页或 GitHub 收集论文：`papers-collect-from-web` / `papers-collect-from-github-awesome`
- 批量下载与修复 PDF：`papers-download-from-list`
- 分析 PDF 入库：`papers-analyze-pdf`
- 重建索引并检索知识库：`papers-build-collection-index` / `papers-query-knowledge-base`
- 论文对比表：`papers-compare-table`
- 发散研究想法：`research-brainstorm-from-kb`
- 生成方向问题清单：`research-question-bank`
- 将 idea 收敛成可执行方案：`idea-focus-coach`
- 做严格审稿式压测：`reviewer-stress-test`
- 诊断反复出现的 skill 失配：`skill-fit-guard`
- 将 ResearchFlow 迁移到其他领域：`domain-fork`

## 2. 分类与技能说明

### 2.1 知识库构建

- `papers-collect-from-web`
  - 何时用：已经有网页列表，希望批量筛论文。
  - 输入：URL 加关键词和会议约束。
  - 产出：可继续下游处理的 triage 列表。

- `papers-collect-from-github-awesome`
  - 何时用：来源是 awesome 或 curated 仓库。
  - 输入：GitHub 仓库链接。
  - 产出：与本地分析流程对齐的候选清单。

- `papers-download-from-list`
  - 何时用：已有人工筛选后的候选清单，需要落地本地 PDF。
  - 输入：如 `paperAnalysis/*.txt` 之类的候选列表。
  - 产出：下载或修复后的本地 PDF 以及状态结果。

- `pdfs-compress-large-files`
  - 何时用：PDF 过大，不利于存储或传输。
  - 输入：`paperPDFs/`（自动扫描）。
  - 产出：压缩报告和更新后的 PDF。

- `papers-analyze-pdf`
  - 何时用：需要对单篇或批量 PDF 做结构化分析。
  - 输入：本地 PDF 路径。
  - 产出：写入 `paperAnalysis/` 的结构化 Markdown。

- `papers-audit-metadata-consistency`
  - 何时用：怀疑日志与分析笔记不一致。
  - 输入：当前 `paperAnalysis/`。
  - 产出：一致性审计报告和质量问题清单。

- `papers-build-collection-index`
  - 何时用：analysis note 有改动，需要刷新索引。
  - 输入：`paperAnalysis/` 中的 frontmatter。
  - 产出：更新后的 `paperCollection/`。

### 2.2 知识库检索与代码上下文检索

- `papers-query-knowledge-base`
  - 何时用：需要按任务、技术、会议查论文，或做比较总结。
  - 输入：研究问题，可附方向与约束。
  - 产出：基于本地知识库证据的回答。

- `papers-compare-table`
  - 何时用：需要多篇论文的结构化对比表。
  - 输入：论文标题、查询条件或 analysis 路径。
  - 产出：Markdown 或 CSV 格式的对比表。

- `code-context-paper-retrieval`
  - 何时用：准备改代码，想先看相关论文依据。
  - 输入：当前代码上下文或目标模块。
  - 产出：brief 或 deep 两种论文检索结果。
  - 说明：会优先从代码仓库里的 `environment.yml` 等文件检测环境，若检测失败会主动询问。

### 2.3 Paper idea 与研究方案

- `research-brainstorm-from-kb`
  - 何时用：需要发散候选 idea。
  - 输入：问题或方向草案。
  - 产出：结构化 idea 候选与相关工作支撑。

- `research-question-bank`
  - 何时用：在决定做什么前，先摸清某个方向的问题全景。
  - 输入：研究方向与粗略任务描述，可选目标 venue。
  - 产出：写入 `QuestionBank/` 的结构化挑战清单。

- `idea-focus-coach`
  - 何时用：idea 太宽，需要逐步收敛。
  - 输入：初始想法和目标偏好。
  - 产出：聚焦目标、非目标、优先假设和 MVP 实验。
  - 独立使用：不依赖 brainstorm 或 reviewer 的输出。

- `reviewer-stress-test`
  - 何时用：想以 ICLR/CVPR/SIGGRAPH 风格提前压测一个 idea。
  - 输入：idea、roadmap 或完整论文。
  - 产出：major/minor 风险以及对应修复动作。
  - 独立使用：不依赖 focus 的输出。

### 2.4 Pipeline 编排与过程保障

- `research-workflow`
  - 何时用：不确定当前处于哪个阶段。
  - 输入：对当前任务的描述。
  - 产出：阶段判断以及推荐的下一步 skill。

- `notes-export-share-version`
  - 何时用：内部笔记需要对外分享。
  - 输入：待导出的笔记。
  - 产出：去除内部痕迹后的可分享 Markdown。

- `skill-fit-guard`
  - 何时用：某次 skill 输出明显失配，而且这种失配可能反复发生。
  - 输入：这次失配的现象。
  - 产出：可能原因、修订选项，以及是否立即修订的提示。

### 2.5 领域迁移

- `domain-fork`
  - 何时用：想把 ResearchFlow 的架构迁移到其他专业领域，比如前端开发、会计或新闻。
  - 输入：目标领域名称。
  - 产出：在交互确认后，生成完整的领域适配仓库，包括重命名后的 skills、目录结构和 README。
  - 触发方式：仅显式调用。

## 3. 触发策略

所有 skill 都通过 description 匹配或显式调用触发，不存在文件变更后自动运行的机制。下表给出推荐触发模式。

| Skill | 触发模式 | 典型时机 |
|-------|----------|----------|
| `papers-collect-from-web` | 显式 | 用户给出 URL 和主题约束时 |
| `papers-collect-from-github-awesome` | 显式 | 用户给出 GitHub 仓库链接时 |
| `papers-download-from-list` | 显式 / 建议式 | collect 完成后建议执行 |
| `pdfs-compress-large-files` | 建议式 | 检测到下载后的 PDF 超过 20 MB 时建议 |
| `papers-analyze-pdf` | 显式 / 建议式 | download 完成后建议执行 |
| `papers-audit-metadata-consistency` | 建议式 | 批量 analyze 完成后建议执行 |
| `papers-build-collection-index` | 建议式 | analyze 完成后建议执行 |
| `papers-query-knowledge-base` | 显式 / 静默 | 用户查询时显式；作为其他 skill 内部依赖时静默 |
| `papers-compare-table` | 显式 | 用户要求对比时 |
| `code-context-paper-retrieval` | 显式 / 建议式 | 修改模型或方法相关代码前建议执行 |
| `research-brainstorm-from-kb` | 显式 | 用户要求生成研究 idea 时 |
| `research-question-bank` | 显式 | 用户希望先梳理问题版图时 |
| `idea-focus-coach` | 显式 | 用户有模糊 idea，想逐步收敛时 |
| `reviewer-stress-test` | 显式 | 用户有较成型 idea，想先压测时 |
| `research-workflow` | 显式 / 建议式 | 用户不确定下一步时 |
| `notes-export-share-version` | 显式 | 用户希望对外分享笔记时 |
| `skill-fit-guard` | 建议式 | agent 检测到明显且反复的 skill 失配时 |
| `domain-fork` | 显式 | 用户明确要求把 ResearchFlow 迁移到其他领域时 |

触发模式说明：

- 显式：由用户直接调用或由 description 匹配触发
- 建议式：agent 在上下文中建议运行该 skill，待用户确认后执行
- 静默：作为其他 skill 的内部依赖被调用

## 4. 调用方式

- 直接描述目标任务，推荐。
- 直接点名 skill，例如“用 papers-download-from-list”。
- 使用 slash 方式，例如 `/papers-analyze-pdf`。

## 5. 调用安全说明

- 实际路由只依赖 `.claude/skills-config.json` 和各 skill 的 `SKILL.md`。
- 本 `User_README_ZN.md` 只是导航文件，不会影响执行。
- `User_README.md` 也只是导航文件，不在注册表中。
