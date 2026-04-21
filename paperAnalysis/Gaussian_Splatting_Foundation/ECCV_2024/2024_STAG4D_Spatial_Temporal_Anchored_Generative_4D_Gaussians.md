---
title: "STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians"
venue: ECCV
year: 2024
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 4dgs
  - 4d-generator
  - diffusion
  - temporal-consistency
  - multi-view-consistency
  - densification
  - status/analyzed
core_operator: 用视频帧作为时序锚点、多视图扩散作为空间锚点来初始化近一致的 multi-view temporal sequence，再用 specially-crafted 4D Gaussian generation pipeline 做稳定的 4D 内容优化。
primary_logic: |
  输入文本、图像或视频后，先用 multi-view diffusion 初始化锚定在视频帧上的多视图序列，
  再通过 first-frame temporal anchor 的 self-attention fusion 保持时间一致，
  最后用针对生成任务设计的 4D Gaussian splatting 和 adaptive densification 做 score distillation 优化，得到高保真的 4D 生成结果。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ECCV_2024/2024_STAG4D_Spatial_Temporal_Anchored_Generative_4D_Gaussians.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T15:06
updated: 2026-04-18T15:06
---

# STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2403.14939) / [PDF](https://arxiv.org/pdf/2403.14939.pdf) / [Project Page](https://nju-3dv.github.io/projects/STAG4D)
> - **Summary**: STAG4D 的价值在于把 4D 生成问题拆成“空间锚定 + 时间锚定”。相比只靠一个扩散模型端到端生 4D，它更像是先把多视图和时序初始化稳定住，再交给 4D Gaussian optimization 去细化。
> - **Key Performance**:
>   - 论文强调在渲染质量、时空一致性和生成鲁棒性上优于先前 4D 生成方法。
>   - 支持 text、image、video 多种输入形式。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

4D 生成最难的几件事往往同时出现：

- 多视图一致性
- 时间一致性
- 高斯优化不稳定

STAG4D 针对的正是这三者的联合作用。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

它把 anchor 概念显式化了：

- 视频帧提供 temporal anchor
- multi-view diffusion 提供 spatial anchor

### The "Aha!" Moment

真正的 aha 是：

**4D 生成如果初始化阶段就不一致，后面再怎么 SDS 优化都很难彻底救回来。**

STAG4D 因此先把多视图序列初始化到“几乎一致”的状态，再把 4D Gaussian 优化作为后续提升步骤。

这和很多 4DGS 编辑的实际问题很像：
编辑前如果初始多视图/多时刻表述已经乱了，后面的 refinement 成本会很高。

### 权衡与局限

- 优势：初始化更稳，输入模态更丰富
- 局限：仍是生成导向，不直接提供编辑接口

## Part III / Technical Deep Dive

### Pipeline

```text
text / image / video input
-> multi-view diffusion initialization
-> temporal anchor via first-frame self-attention fusion
-> 4D Gaussian score distillation optimization
-> adaptive densification for stable generation
-> high-fidelity 4D content
```

### 关键技术点

#### 1. Spatial-Temporal Anchors

空间和时间两个锚点分别负责不同一致性来源，这使初始化更可控。

#### 2. Adaptive Densification

作者专门为生成任务改造了 4D Gaussian densification，缓解不稳定梯度问题。

### 对你的价值

如果你之后考虑生成式 4DGS 编辑、从视频或文本先生成再编辑，这篇论文很值得保留，因为它提供了更稳的初始化范式。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/ECCV_2024/2024_STAG4D_Spatial_Temporal_Anchored_Generative_4D_Gaussians.pdf]]
