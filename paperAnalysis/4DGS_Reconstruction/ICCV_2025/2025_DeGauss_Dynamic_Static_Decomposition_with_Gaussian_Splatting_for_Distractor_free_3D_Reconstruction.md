---
title: "DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction"
venue: ICCV
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - dynamic-decomposition
  - static-dynamic-separation
  - wild-reconstruction
  - monocular-video
  - status/analyzed
core_operator: 用 foreground/background Gaussians 与概率 mask 显式解耦动态与静态内容，使动态场景中的干扰物不再污染目标 3D 重建。
primary_logic: |
  输入真实动态场景采集数据，先分别用 foreground Gaussians 建模动态元素、background Gaussians 建模静态内容，
  再通过 probabilistic mask 协调两者的组合和独立优化，
  从而在 highly dynamic、interaction-rich 场景里重建出更干净的 distractor-free 3D 场景。
pdf_ref: paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_DeGauss_Dynamic_Static_Decomposition_with_Gaussian_Splatting_for_Distractor_free_3D_Reconstruction.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T15:06
updated: 2026-04-18T15:06
---

# DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2503.13176) / [PDF](https://arxiv.org/pdf/2503.13176.pdf) / [Project Page](https://batfacewayne.github.io/DeGauss.io/)
> - **Summary**: DeGauss 对你的主线很有意义，因为它把 static/dynamic separation 放到了更真实、更脏的数据场景里，而且目标不是一般意义上的动态重建，而是“去掉干扰之后的可用 3D 场景”。这和很多编辑前处理其实是同一个问题。
> - **Key Performance**:
>   - 论文在多个真实动态数据集上都报告了稳定优于现有方法的结果。
>   - 特别强调在 cluttered、egocentric、interaction-rich 场景中的泛化。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

真实世界动态采集中，经常会出现：

- 走动的人
- 临时遮挡物
- 交互动作带来的局部动态干扰

如果这些干扰直接进统一 3DGS 重建，静态场景会被严重污染。

DeGauss 的目标是：从动态观测中恢复一个干净的、可用的 3D 场景。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

它的方法并不追求复杂启发式，而是回到一个很基础的分解：

- foreground Gaussians
- background Gaussians
- probabilistic mask

### The "Aha!" Moment

真正的 aha 是：

**很多真实场景 3D 重建失败，不是几何不会学，而是静态目标和动态干扰被绑在了一起。**

DeGauss 因此把动态和静态内容建成两个相互独立但可协同的高斯系统。

这对你做 4DGS 编辑尤其有价值，因为很多编辑工作首先需要一个“干净的可编辑底座”，而不是直接在被干扰污染的场景上做操作。

### 权衡与局限

- 优势：真实场景泛化更好，静动分解直接
- 局限：如果前景/背景边界极其模糊，mask 仍可能成为瓶颈

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic real-world captures
-> foreground Gaussians for dynamic content
-> background Gaussians for static content
-> probabilistic mask for coordination
-> independent but complementary optimization
-> distractor-free 3D reconstruction
```

### 关键技术点

#### 1. Decoupled Dynamic / Static Design

方法核心是把动态与静态内容拆开建模，而不是靠鲁棒损失在统一表示里硬扛。

#### 2. Probabilistic Composition

foreground 和 background 不是简单二选一，而是通过概率 mask 共同决定像素解释。

### 对你的价值

如果你后面做真实视频上的 4DGS 编辑，这篇论文非常值得保留，因为它直接回答：
**如何先把可编辑主体从动态干扰里分离出来。**

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_DeGauss_Dynamic_Static_Decomposition_with_Gaussian_Splatting_for_Distractor_free_3D_Reconstruction.pdf]]
