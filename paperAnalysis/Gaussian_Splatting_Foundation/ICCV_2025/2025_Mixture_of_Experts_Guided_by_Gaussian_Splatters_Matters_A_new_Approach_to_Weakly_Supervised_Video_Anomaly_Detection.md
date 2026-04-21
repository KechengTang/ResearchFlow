---
title: "Mixture of Experts Guided by Gaussian Splatters Matters: A new Approach to Weakly-Supervised Video Anomaly Detection"
venue: ICCV
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - cross-domain-reference
  - gaussian-splatting
  - weakly-supervised-anomaly-detection
  - mixture-of-experts
  - temporal-gaussian-splatting
  - video-understanding
  - status/analyzed
core_operator: 把 anomaly score 时间轴上的峰值渲染成 Temporal Gaussian kernels，替代传统 MIL 的 top-k 弱监督伪标签，再用 anomaly-class-specific experts 与 gate 融合得到更细粒度的时序异常检测。
primary_logic: |
  先用 I3D 与 UR-DMU 提取 snippet 级特征和粗异常分数，
  再从异常分数时间轴上检测峰值并生成 Temporal Gaussian Splatting 伪标签，
  同时训练面向不同异常类别的 experts，
  最后由 gate 结合 task-aware features 与 expert scores 输出帧级异常分数。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_Mixture_of_Experts_Guided_by_Gaussian_Splatters_Matters_A_new_Approach_to_Weakly_Supervised_Video_Anomaly_Detection.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T16:49
updated: 2026-04-18T16:49
---

# Mixture of Experts Guided by Gaussian Splatters Matters: A new Approach to Weakly-Supervised Video Anomaly Detection

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2508.06318](https://arxiv.org/abs/2508.06318) | [Code](https://github.com/snehashismajhi/GS-MoE)
> - **Summary**: 这篇论文并不是 3D/4D 场景重建里的 Gaussian Splatting，而是把“Gaussian splatting”改造成一个时序监督算子。作者在弱监督视频异常检测里，用高斯核覆盖异常峰值附近的整段时间窗，再用 MoE 专家分解不同异常类型，核心收益是更完整的异常时序定位和更细粒度的类别专门化。
> - **Key Performance**:
>   - 在 UCF-Crime 上达到 `91.58% AUC`，较文中上一档最佳方法提升约 `3.56%`，异常视频上的 `AUC_A` 达到 `83.86%`。
>   - 在 XD-Violence 上达到 `82.89 AP / 85.74 APA`，在 MSAD 上达到 `87.72 AUC`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？
弱监督视频异常检测的训练标签通常只有“整段视频是否异常”，但测试时却要输出帧级或 snippet 级异常分数。  
这导致两个老问题一直很难处理：

- MIL 训练通常只盯住 top-k 最异常片段，容易漏掉较弱但真实的异常片段
- 各类异常事件差异很大，用同一个共享检测头容易把复杂异常糊成一个模糊类别

这篇论文要解决的是：

**如何在不引入帧级标注的前提下，让模型既学到更完整的异常时间窗，又能区分不同异常类型。**

### 核心能力定义

- **输入**: 监控视频的 snippet 级特征
- **输出**: 帧级 / snippet 级异常分数
- **强项**: 弱监督条件下的时序定位、复杂异常类别区分
- **弱项**: 并不处理几何重建、视图合成或真实 3D Gaussian 表示

### 真正的挑战来源

- 视频级标签太粗，缺少精确的时间边界
- 异常通常与正常动作交织，top-k 学习很容易只学到最尖锐的峰
- 不同异常类别的时序形态差异大，共享模型会丢掉类别特异性

### 边界条件

- 方法工作在监控视频异常检测场景，不是 3D/4D GS 场景建模
- 依赖训练阶段已有异常类别或可聚类的异常簇
- 高斯核是沿时间轴的监督分布，不是空间中的 3D primitives

## Part II / High-Dimensional Insight

### 方法的整体设计哲学
这篇论文最值得记住的不是 MoE 本身，而是它对弱监督目标的改写：

**与其只把最异常的几个片段推向 1，不如把异常峰值周围的一整段时间窗都变成软监督目标。**

作者把异常分数时间轴上的局部峰值检测出来，然后为每个峰生成一个 temporal Gaussian kernel。这样监督不再是离散 top-k 点，而是覆盖异常持续区间的软分布。

### The "Aha!" Moment

真正的 aha 是：

**Gaussian splatting 在这里不再是几何渲染，而是“把局部置信峰扩散成连续时间支持域”的软定位机制。**

这一步解决了弱监督里一个很核心的问题：很多异常片段的分数没有最高，但它们仍属于同一异常事件。如果只学峰值，模型会把异常事件切碎；如果用 Gaussian kernel 把整段时间窗涂亮，模型就能学到更完整的事件边界。

### 为什么这个设计有效？

- TGS 把孤立峰值变成连续软标签，缓解 top-k 的监督稀疏问题
- class-specific experts 让不同异常类别拥有专门参数，不再被共享模型平均掉
- gate 再把 expert logits 和 task-aware features 融合，保留类别细节同时利用全局异常上下文

### 对我当前方向的价值
它和你主线里的 3D/4DGS 不是同一赛道，但 operator 层面仍有启发：

- 它展示了 “Gaussian splatting” 可以是一个**软分配 / 软扩散 / 软监督**机制，而不只是几何渲染
- 如果以后做 `4DGS_Editing` 的时间编辑掩码传播，这种从峰值向时间窗扩展的高斯化监督值得借鉴
- 如果做 feed-forward Gaussians 或 4D agent 系统，MoE + gate 的结构也适合处理多类型动作、多类编辑意图或多类时序事件

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 关系在“时间掩码传播”而不在表示本体。TGS 可视为一种时间窗软约束，可迁移到编辑触发区间的自动扩展与平滑。
- **与 3DGS_Editing**: 直接关系较弱，但它提醒一个通用原则：局部强响应不等于完整可编辑区域，软分布监督往往比硬阈值更稳。
- **与 feed-forward Gaussians**: 关系体现在 routing 与 specialization。若未来前馈模型要处理多类动态模式，expert decomposition 很可能比单头网络更合适。

### 战略权衡

- 优点: 对弱监督问题下手很准，既补监督稀疏，又补类别多样性
- 代价: 类别专家数量、路由复杂度和推理成本都会上升，且对异常类别划分方式敏感

## Part III / Technical Deep Dive

### Pipeline

```text
video
-> I3D task-agnostic features
-> UR-DMU task-aware anomaly features
-> peak detection on abnormal score timeline
-> Temporal Gaussian Splatting pseudo-labels
-> class-specific expert models
-> gate with bidirectional cross-attention
-> final frame/snippet anomaly score
```

### 关键模块

#### 1. Temporal Gaussian Splatting
作者先在异常分数时间轴上找局部峰值，再按峰宽和峰值差构造 Gaussian kernels。  
这些 kernels 不只覆盖最高分 snippet，还会把同一异常事件中较弱但相关的片段纳入训练目标。

#### 2. Class-Specific Experts
每个 expert 只负责一类异常或一个异常簇。  
这让 “抢劫、打架、纵火、偷窃” 之类具有不同时间模式的事件不再共用同一个决策头。

#### 3. Gate With Task-Aware Features
gate 不只是简单求和 expert 分数，而是把 expert logits 投影后与 task-aware features 做双向 cross-attention，再经过 transformer + MLP 得到最终预测。

### 关键实验信号

- UCF-Crime: `91.58 AUC`，较 VadCLIP 的 `88.02` 高出约 `3.56`
- UCF-Crime 的异常视频指标 `AUC_A` 达到 `83.86`，相对 UR-DMU 的 `70.81` 有明显提升
- XD-Violence: `82.89 AP / 85.74 APA`，其中 APA 优于先前结果
- MSAD: `87.72 AUC`，高于作者复现的 UR-DMU `85.78`
- 组件消融里，完整 GS-MoE 相比 baseline `86.97 AUC` 提升到 `91.58`

### 少量关键数字

- 完整 GS-MoE 在 UCF-Crime 上 `91.58 AUC`
- Gate 版本优于 Soft-MoE，`91.58` 对 `90.14`
- 13 experts 设置下整体系统约 `9.57 FPS`，参数量约 `16.02M`

### 局限、风险、可迁移点

- **局限**: 这不是几何 3D/4D GS 论文，和你主线知识库的关系主要在 operator 可迁移性
- **风险**: 峰值检测如果过早或过密，会把错误片段扩散成伪标签；论文也提到对 subtle anomaly、小目标异常、long peak 仍有失误
- **可迁移点**: peak-to-window Gaussian smoothing、expert specialization、gate-level feature fusion 都可以迁移到时序编辑、动态 mask 传播、动作分段等任务

### 实现约束

- 使用 I3D 提取 `1024` 维 snippet 特征
- expert 为 transformer block + MLP，单 expert 约 `0.5M` 参数
- 训练在单张 RTX A4500 上约 `3` 小时，UCF-Crime 测试约 `55` 秒
- 当前实现里 experts 顺序执行，因此 FPS 不是上限，作者明确指出并行实现会更快

## Local Reading / PDF 参考
![[paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_Mixture_of_Experts_Guided_by_Gaussian_Splatters_Matters_A_new_Approach_to_Weakly_Supervised_Video_Anomaly_Detection.pdf]]
