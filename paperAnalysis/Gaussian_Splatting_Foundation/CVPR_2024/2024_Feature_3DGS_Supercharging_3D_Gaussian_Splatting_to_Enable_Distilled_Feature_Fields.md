---
title: "Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields"
venue: CVPR
year: 2024
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - splatted-features
  - foundation-model-distillation
  - clip-features
  - semantic-grounding
  - open-world-segmentation
  - status/analyzed
core_operator: 在每个高斯上显式存储高维语义特征，并通过 feature-aware rasterization 与 2D foundation model distillation，把 3DGS 从 RGB 场景表示扩展成可查询的特征场。
primary_logic: |
  输入多视图图像及对应 2D foundation model 特征，先给每个高斯学习颜色之外的高维语义向量，
  再通过改造后的高斯光栅化和轻量 decoder 处理 RGB 与 feature 在分辨率和通道上的不一致，
  最终得到可用于分割、语言查询和语义编辑的显式 3D 特征场。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/CVPR_2024/2024_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_Fields.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T14:30
updated: 2026-04-18T14:30
---

# Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Shah_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.html) / [PDF](https://arxiv.org/pdf/2312.03203.pdf) / [Project Page](https://feature-3dgs.github.io/)
> - **Summary**: Feature 3DGS 的价值不在于单独做出一个“语义版 3DGS”，而在于它证明了高斯本身可以承载高维 foundation features，这让后续的语言定位、分割、交互式编辑都有了更直接的 3D 作用位点。
> - **Key Performance**:
>   - 论文展示了 novel-view semantic segmentation、language-guided editing 和 segment-anything 等能力。
>   - 与 NeRF 特征场路线相比，它在训练与渲染速度上明显更快，同时特征质量并不更差。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

这篇论文要解决的是：

**3DGS 很快，但原生只会表示颜色；如果没有稳定的 3D 语义特征场，很多编辑、分割和语言查询工作都只能退回到 2D。**

此前基于 NeRF 的 feature field distillation 已经证明“3D 语义场”很有用，但有两个老问题：

- NeRF 训练和渲染都慢
- 隐式特征场容易有连续性伪影，影响特征质量

Feature 3DGS 的目标是把这件事搬到显式高斯表示里。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

作者不是简单给每个高斯多加一串 feature，而是系统地解决了两个不匹配：

- RGB 图像与 foundation feature map 的空间分辨率不同
- 语义特征的通道结构和颜色建模完全不同

因此论文真正提出的是一整套 **feature-aware 3DGS distillation pipeline**。

### The "Aha!" Moment

真正的 aha 是：

**对编辑和语义检索来说，最重要的不只是“渲染得快”，而是“语义特征能不能跟 3D 原语稳定绑定”。**

Feature 3DGS 的回答是：

1. 每个高斯同时存 RGB 与语义特征
2. 用专门的 rasterization / decoder 路径渲染特征图
3. 用 2D foundation models 做蒸馏监督

这样一来，3DGS 就不再只是“可渲染场景”，而是“可查询、可分割、可语言作用”的 3D 特征容器。

这对你研究 4DGS 编辑也很关键，因为很多 4D 编辑瓶颈其实不是“不会生成”，而是 **找不到稳定的 4D 语义锚点**。

### 权衡与局限

- 优势：显式、快速、利于语义查询与交互编辑
- 局限：主要针对静态 3DGS；语义质量上限仍依赖 2D foundation models

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view images
-> 2D foundation model feature extraction
-> per-Gaussian semantic feature learning
-> feature-aware Gaussian rasterization and lightweight decoder
-> explicit 3D feature field
-> segmentation / language-guided editing / SAM-style prompting
```

### 关键技术点

#### 1. Per-Gaussian Features

论文把高维语义特征直接挂在高斯原语上，这让 feature field 变成显式表示，而不是隐式网络里的隐藏状态。

#### 2. Distillation with Architectural Changes

作者强调不能把 feature 监督直接照搬到原始 3DGS 上，因此引入了针对 feature resolution 和 channel consistency 的结构调整。

#### 3. Editing and Query Potential

虽然这篇论文本身不是专门的 3D 编辑论文，但它已经展示了 language-guided editing 和 segment-anything 式交互，这使它在你的 KB 里非常像“语义基础设施论文”。

### 关键信号

- 文中展示的分割与语言引导编辑结果，说明语义特征不是附属产物，而是真能在 3DGS 上工作。
- 更快的训练/渲染速度意味着这条路线更适合后续和 4DGS 结合。

### 对你的价值

对你当前方向，这篇论文最重要的作用是：

- 为 3DGS / 4DGS 的语义 grounding 提供显式思路
- 给后续 point-level localization、language-embedded gaussians、4D LangSplat 这类工作提供上游背景

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/CVPR_2024/2024_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_Fields.pdf]]
