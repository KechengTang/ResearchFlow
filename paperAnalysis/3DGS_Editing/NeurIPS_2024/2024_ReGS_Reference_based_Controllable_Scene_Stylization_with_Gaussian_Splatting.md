---
title: "ReGS: Reference-based Controllable Scene Stylization with Gaussian Splatting"
venue: NeurIPS
year: 2024
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - stylization
  - reference-based
  - controllable-stylization
  - appearance-editing
  - real-time-rendering
  - texture-guided-control
  - view-consistency
  - status/analyzed
core_operator: 不再只优化 3DGS 外观，而是通过 texture-guided control 用颜色梯度定位 texture underfitting 区域，并结构化 densify 负责高斯，再以深度与伪多视角监督维持几何和跨视角一致性。
primary_logic: |
  输入预训练 3DGS 内容场景和内容对齐的参考风格图，
  先通过颜色梯度定位外观欠拟合的局部高斯，
  再将其替换为结构化更密集的小高斯集合以表达高频纹理，
  同时利用深度正则保持原始几何，并用 pseudo-view supervision 与 TCM loss 维持跨视角和遮挡区域的一致风格，
  最终得到可实时渲染的 stylized 3DGS。
pdf_ref: paperPDFs/3DGS_Editing/NeurIPS_2024/2024_ReGS_Reference_based_Controllable_Scene_Stylization_with_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-10T20:11
updated: 2026-04-10T20:11
---

# ReGS: Reference-based Controllable Scene Stylization with Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2024/file/076c1fa639a7190e216e734f0a1b3e7b-Paper-Conference.pdf)
> - **Summary**: ReGS 把 reference-based scene stylization 从 NeRF 迁到了 3DGS，但真正的创新不是“换个表示”，而是承认 3DGS 的离散高斯本身表达不了连续纹理，因此必须显式控制局部高斯布局来补足纹理细节。
> - **Key Performance**:
>   - 在官方 reference-based stylization benchmark 上达到 `Ref-LPIPS 0.202`、`Robustness 31.27`，均优于 `ARF / SNeRF / Ref-NPR`。
>   - 在单张 `A5000` 上实现约 `91.4 FPS` 的实时 stylized view synthesis。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

ReGS 关注的是一种非常具体的编辑任务：

**给定一个内容对齐的参考风格图，如何把其纹理和笔触精确烘焙进 3D 场景，并且保持实时自由视角浏览？**

这和任意风格迁移不一样。这里的目标不是“整体氛围像某位画家”，而是尽量精确地跟随 reference image 中已经对齐到场景内容的局部纹理。

### 核心能力定义

- **输入**：预训练内容 3DGS + content-aligned reference image
- **输出**：可实时渲染的 stylized 3DGS
- **擅长**：高保真 reference-based texture transfer、局部 appearance editing
- **不处理**：几何编辑、对象插入删除、动态 4D 场景

### 真正的挑战来源

- 3DGS 是离散高斯表示，外观与几何高度耦合
- 仅仅固定几何再优化颜色，往往无法表达参考图中的连续高频纹理
- 若强行改动高斯布局，又容易破坏原场景几何和跨视角一致性

### 边界条件

- 需要内容对齐的参考图
- 主要是 appearance / texture stylization，不是语义级对象编辑

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

ReGS 的核心思想可以概括为：

**3DGS 的 stylization 问题，本质上不是“颜色不够准”，而是“负责表达纹理的高斯分辨率不够”。**

因此，作者不是继续沿用 NeRF 时代那种“只优化 appearance”的思路，而是直接介入负责区域的 Gaussian 排布。

### The "Aha!" Moment

真正的 aha 是：

**当高频纹理无法被当前高斯布局表达时，正确做法不是继续硬调颜色，而是找到 texture underfitting 的 responsible Gaussians，并把它们替换成更密的、结构化的小高斯集合。**

作者用颜色梯度来定位这些区域：

- 颜色梯度大，说明优化仍在拼命修补纹理
- 这些高斯往往就是 reference texture 没被表达出来的瓶颈

然后再做 structured densification：

- 用 `9` 个更小的 Gaussian 替换一个 parent Gaussian
- 8 个放在原椭球的八个象限，1 个放在中心

这一步比默认的 positional-gradient densification 更适合 stylization，因为这里缺的是纹理表达力，而不是几何。

### 为什么这个设计有效？

- 颜色梯度能直接反映 texture underfitting，而位置梯度在 stylization 阶段已不敏感
- depth regularization 防止新 densified Gaussians 漂离表面导致几何扭曲
- pseudo-view supervision 和 TCM loss 把参考图中的风格传播到新视角和遮挡区域

### 战略权衡

- **优点**：把 3DGS 的实时渲染优势保住了，同时大幅提升 reference-based 纹理对齐能力
- **局限**：依赖 reference 对齐，且本质仍是外观编辑，不解决更高层语义修改

## Part III / Technical Deep Dive

### Pipeline

```text
pretrained content 3DGS + content-aligned reference image
-> optimize stylized appearance under reconstruction/style losses
-> accumulate color gradients to locate texture-underfitting Gaussians
-> structured densification replaces responsible Gaussians with tiny Gaussian sets
-> depth regularization preserves geometry
-> pseudo stylized views + TCM loss enforce cross-view consistency and occluded-region style spread
-> stylized 3DGS with real-time rendering
```

### 关键模块

#### 1. Texture-Guided Gaussian Control

这是整篇最关键的模块。它不是基于位置梯度，而是基于颜色梯度统计每 `100` 次迭代执行一次 densification。这样 densify 发生在真正缺纹理的地方，而不是几何最难的地方。

#### 2. Structured Densification

作者没有随机复制高斯，而是做结构化替换，保证：

- 新高斯覆盖原始空间范围
- 同时能以更小粒度表达连续纹理变化

#### 3. Depth-based Geometry Regularization + View-consistent Stylization

深度正则防止高斯浮到表面之外，pseudo-view warping 和 TCM loss 则确保风格不会只贴在参考视角上。

### 关键实验信号

- `Ref-LPIPS` 从 `0.339`（Ref-NPR）降到 `0.202`
- `Robustness` 从 `28.11`（Ref-NPR）提升到 `31.27`
- 推理速度从 NeRF 风格方法约 `16 FPS` 提升到 `91.4 FPS`
- 控制实验表明：哪怕只增加 `0.05M` 个新高斯，texture-guided control 也能比默认 density control 更快补齐高频纹理

### 对当前 idea 的启发

这篇论文对你当前方向的意义，主要在 **appearance/material edit 的表示控制**：

- 它给出了一个很清楚的结论：appearance edit 不一定只是 feature change，很多时候还需要局部表示重排
- 如果未来你的 4D 编辑要做局部材质变化，ReGS 的“where to densify”思想很值得迁移
- 另外它还展示了 3DGS 可以自然承接 2D 涂抹式 appearance editing，这对后续交互式编辑界面也有启发

### 实现约束

- stylization 训练约 `3000` iterations
- 使用单张 `A5000` 训练
- 使用 diffuse-only rendering 做 stylization，避免 view-dependent effects 干扰

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/NeurIPS_2024/2024_ReGS_Reference_based_Controllable_Scene_Stylization_with_Gaussian_Splatting.pdf]]
