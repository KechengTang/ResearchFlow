---
title: "MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer"
venue: CVPR
year: 2026
tags:
  - 4DGS_Reconstruction
  - feed-forward
  - monocular-video
  - motion-aware
  - streaming-reconstruction
  - attention-forcing
  - motion-mask-supervision
  - grouped-causal-attention
  - pose-depth-joint-estimation
  - status/analyzed
core_operator: 用训练期 motion-mask 监督的 attention-forcing 把动态物体从静态结构里拆开，再用 grouped causal attention 和 BA-like refinement 把这种分解搬到流式单目 4D 重建里。
primary_logic: |
  输入单目视频流后，先在共享 Transformer 骨干上联合预测深度、位姿、point map 与 motion mask，
  训练时用 attention-forcing 把注意力显式压向静态区域以抑制动态干扰，
  推理时再用 grouped causal attention 支持长序列流式处理，并用 BA-like refinement 轻量回看相机 token，提升时序一致性与位姿稳定性。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2026/2026_MoRe_Motion_aware_Feed_forward_4D_Reconstruction_Transformer.pdf
category: 4DGS_Reconstruction
created: 2026-04-17T10:45
updated: 2026-04-17T10:45
---

# MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://hellexf.github.io/MoRe/) | [arXiv 2603.05078](https://arxiv.org/abs/2603.05078)
> - **Summary**: MoRe 的核心贡献不在显式 4DGS 表示本身，而在于提出一种“训练时显式教模型忽略动态干扰、推理时不再额外依赖动态先验”的流式 4D 重建范式，这对单目长视频尤其关键。
> - **Key Performance**:
>   - 流式相机位姿估计在 `Sintel` 上达到 `0.1474 ATE`，在 `TUM-dynamics` 上达到 `0.0260 ATE`，明显优于其他 streaming 基线。
>   - 流式深度估计在 `KITTI` 上达到 `0.072 AbsRel / 0.966 δ<1.25`，推理速度约 `30.09 FPS`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

MoRe 盯住的是单目动态视频里一个非常顽固的问题：
**运动物体会同时污染相机位姿、几何估计和长序列时序一致性，导致很多静态重建骨干一进动态场景就明显失稳。**

所以它不是只想提高一个 depth 指标，而是想做一个统一的 streaming 4D reconstruction backbone：

- 同时输出 pose、depth、point map、motion mask。
- 能处理长序列流式输入，而不是一次性吃固定长度片段。
- 推理时不额外依赖光流、分割器或测试时优化。

### 核心能力定义

- **输入**：单目视频序列。
- **输出**：每帧深度、相机位姿、点图和 motion mask。
- **擅长**：动态干扰下的位姿稳定、流式推理、跨静动态数据泛化。
- **不直接提供**：显式 dynamic Gaussian 表示；它更像 4D 重建与 4DGS 系统前端的几何/位姿骨干。

### 真正的挑战来源

- 动态区域会破坏静态多视几何假设。
- 长序列会让全局注意力代价过高，不适合流式系统。
- 如果不显式处理动态物体，模型很容易把物体运动错误归因到相机运动。

### 边界条件

MoRe 在训练中依赖 motion mask 标注来教会模型区分动静；测试时虽然不再需要这些 mask，但这也意味着它的上限仍与训练监督质量紧密相关。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

MoRe 的设计其实很干净：

1. 不把 motion mask 当测试时输入，而是当训练时“教注意力怎么看”的老师。
2. 不用全局 attention 硬撑长序列，而是改成适合 streaming 的 grouped causal attention。
3. 不做沉重的 bundle adjustment，而是做 BA-like 的轻量后处理来回看 camera token。

### The "Aha!" Moment

真正的 aha 在于：

**动态场景里，最关键的不是再加一个运动分支，而是先把注意力从“被动态物体牵着走”纠正为“优先信任静态背景”。**

这件事通过 attention-forcing 实现：

- 训练时用 ground-truth motion mask 对注意力分布施加显式约束。
- 模型被迫学会把静态区域当作 pose 和 geometry 的主证据。
- 一旦这种分离学会了，推理时就不必再显式输入 motion mask。

### 为什么这个设计有效？

- pose、depth 和 point map 都受益于同一套“动态抑制”机制，不是局部修补。
- grouped causal attention 允许同一帧内充分聚合空间上下文，同时保持跨帧因果依赖。
- BA-like refinement 不是重优化整条轨迹，而是低成本纠正相机 token，适合在线系统。

### 战略权衡

- **优点**：对 streaming 场景友好，推理快，且在动态数据上比普通静态骨干稳很多。
- **局限**：表征层仍以 point/depth/pose 为主，不是显式 4DGS；如果训练 mask 本身噪声很大，收益会直接受限。

## Part III / Technical Deep Dive

### Pipeline

```text
monocular video stream
-> transformer backbone
-> jointly predict depth / point map / camera pose / motion mask
-> attention-forcing during training aligns attention with static regions
-> grouped causal attention supports streaming inference
-> BA-like refinement revisits duplicated camera tokens for global correction
-> output temporally coherent 4D reconstruction signals
```

### 关键机制

#### 1. Attention-forcing

MoRe 最关键的 operator。训练时利用 motion mask 监督注意力，让模型显式学会“动态物体不该主导 pose 估计”。这是把动态分解注入到 backbone 内部，而不是在输出端再补一个分割头。

#### 2. Grouped causal attention

标准 causal attention 会错误地限制同帧 token 的空间通信。MoRe 允许同一帧内部自由注意，同时保留跨帧的因果顺序，因此既适合时序推理，也不牺牲空间一致性。

#### 3. BA-like refinement

在整段流式处理结束后，模型复制 camera token 再做一次轻量修正。它不像传统 BA 那样昂贵，但能显著降低位姿平移和旋转误差。

### 关键实验信号

- **位姿估计**：流式版本在 `Sintel / Bonn / TUM-dynamics / ScanNet` 上分别达到 `0.1474 / 0.0211 / 0.0260 / 0.0605` 的 ATE，整体优于 streaming 基线，并接近甚至部分优于 full-attention 方法。
- **深度估计**：流式版本在 `Sintel` 达到 `0.254 AbsRel / 0.637 δ<1.25`，在 `KITTI` 达到 `0.072 / 0.966`，说明 motion-aware 设计没有以牺牲深度为代价。
- **消融**：去掉 attention-forcing 后，Sintel ATE 从 `0.147` 退化到 `0.163`；去掉 BA-like refinement 后退化到 `0.155`。
- **流式注意力消融**：不用 grouped causal attention 时，Sintel 深度从 `0.254 AbsRel` 退化到 `0.277`，KITTI 从 `0.072` 退化到 `0.079`。
- **速度**：KITTI 上推理约 `30.09 FPS`，处于最快一档。

### 实现约束

MoRe 训练依赖大规模静态与动态混合数据，并明确承认对 motion mask 质量敏感。它适合做动态重建前端或 4DGS 上游骨干，但若直接当成最终显式 4D 表示，还需要再接高斯化或渲染层。

## Local Reading / 本地 PDF 引用

![[paperPDFs/4DGS_Reconstruction/CVPR_2026/2026_MoRe_Motion_aware_Feed_forward_4D_Reconstruction_Transformer.pdf]]
