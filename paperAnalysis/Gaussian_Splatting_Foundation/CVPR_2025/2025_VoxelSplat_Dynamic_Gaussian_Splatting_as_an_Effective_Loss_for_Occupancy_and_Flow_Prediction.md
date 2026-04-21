---
title: "VoxelSplat: Dynamic Gaussian Splatting as an Effective Loss for Occupancy and Flow Prediction"
venue: CVPR
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - occupancy-prediction
  - scene-flow
  - training-loss
  - self-driving
  - semantic-supervision
  - status/analyzed
core_operator: 不把 Gaussian Splatting 当作最终表示，而是把动态 Gaussian 渲染作为 occupancy + scene flow prediction 的训练损失，通过把语义高斯投影回 2D 可见空间并用预测 flow 更新其中心，实现对 3D semantics 和 dynamic flow 的额外监督。
primary_logic: |
  先从 occupancy representation 中解码 sparse semantic 3D Gaussians，
  再将这些高斯投影到 2D camera view 形成可见空间监督，
  同时用预测 scene flow 更新高斯中心并在相邻时刻渲染，
  最终把 Gaussian rendering 作为训练时的 regularization/loss，提升 occupancy semantics 与 flow prediction。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_VoxelSplat_Dynamic_Gaussian_Splatting_as_an_Effective_Loss_for_Occupancy_and_Flow_Prediction.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T16:37
updated: 2026-04-18T16:37
---

# VoxelSplat: Dynamic Gaussian Splatting as an Effective Loss for Occupancy and Flow Prediction

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2506.05563](https://arxiv.org/abs/2506.05563) · [Project Page](https://zzy816.github.io/VoxelSplat-Demo/)
> - **Summary**: VoxelSplat 的思路很特别: 它不把 Gaussian 当最终 4D 场景表示，而是把 dynamic Gaussian rendering 变成 occupancy 和 scene flow 学习的训练损失。这样 Gaussian 在这里充当的是 supervision adapter，把 3D/4D 结构预测重新投影回更可靠的 2D 可见空间。
> - **Key Performance**:
>   - 论文强调在不增加 inference time 的情况下，同时提升 semantic occupancy 和 scene flow accuracy。
>   - 方法可无缝接入多种现有 occupancy models。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

camera-based occupancy prediction 有几个经典难点:

- 很多 3D 标注区域在相机视角下不可见，监督容易误导
- voxel 表示对显式动态建模不自然
- 稀有高速动态事件很难学

VoxelSplat 要解决的是:

**如何给 occupancy + flow prediction 提供更符合可见性和动态性的训练信号。**

### 核心能力定义

- **输入**: occupancy representation 与预测的 scene flow
- **输出**: 更好的 semantic occupancy 与 flow estimation
- **强项**: 作为训练 regularizer 通用、无需改推理结构、适合自动驾驶
- **弱项**: 不提供最终 Gaussian scene asset

### 真正的挑战来源

- 3D supervision 里有大量 camera-invisible 区域
- voxel 格子本身不擅长显式 Laplacian motion
- 仅在 3D 网格上优化，缺少与 2D 可见空间的一致桥梁

### 边界条件

- 核心价值在训练阶段，非推理阶段
- 面向 occupancy / flow perception，而非内容重建
- 强依赖多帧 2D labels 和自驾驶设定

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

VoxelSplat 的哲学是:

**Gaussian 不一定非要做最终表示，也可以作为 3D/4D 结构任务的监督媒介。**

作者把 occupancy features 解码成 sparse semantic Gaussians，再渲染回 2D。  
这样 Gaussian 在这里变成一种 differentiable bridge。

### The "Aha!" Moment

真正的 aha 是:

**与其直接用难以解释的 3D voxel supervision，不如让语义和运动先变成可显式移动、可投影的高斯，再借 2D 可见标签做监督。**

这让监督更贴近真正能被相机看到的内容，也让 flow 学习更显式。

### 为什么这个设计有效

Gaussian 可投影到 2D camera view，因此天然更适合处理 visibility-aware supervision。  
再加上 predicted flow 可以直接更新 Gaussian centers，scene flow 学习也变成了“高斯怎么动”的问题，而不只是 voxel 内部特征漂移。

### 对我当前方向的价值

这篇论文对你当前方向很有意思，因为它拓宽了 Gaussian 的用途:

- 不是只做 scene representation
- 还可以做 differentiable supervision layer

如果你后面要把 Gaussian 接到 perception、planning 或 occupancy pipeline，这篇很值得留档。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 直接关系较弱，但它说明 Gaussian 可作为中间可微接口，不止是最终编辑对象。
- **与 3DGS_Editing**: 一些编辑任务也需要把 3D 操作重新投回 2D 监督或交互空间，这篇的桥接思路值得借鉴。
- **与 feed-forward Gaussians**: 很适合 future feed-forward perception-Gaussian hybrids，把 Gaussian 当作训练时结构正则。

### 战略权衡

- 优点: 训练时增益明显，推理时无额外成本
- 代价: Gaussian 不是最终输出，研究贡献更偏 loss design

---

## Part III / Technical Deep Dive

### Pipeline

```text
occupancy representation
-> decode sparse semantic 3D Gaussians
-> project to visible 2D camera space for semantic supervision
-> move Gaussians with predicted scene flow
-> render adjacent frames for self-supervised flow learning
-> improve occupancy and flow prediction
```

### 关键模块

#### 1. Semantic Gaussians

作者从 3D representation 中解码 sparse semantic Gaussians，使语义监督从 voxel space 获得一个显式载体。

#### 2. 2D Projection Supervision

把 Gaussian 投回 camera-visible 2D space，让 2D 标签重新参与 3D 语义学习。

#### 3. Flow-Updated Gaussians

利用预测 flow 更新 Gaussian centers，再在相邻时刻渲染，形成自监督的 scene flow loss。

### 关键实验信号

- 论文强调不增加 inference time，这说明方法定位清晰: 训练增强而非推理改造
- semantic occupancy 和 flow 同时提升，是其主要证据
- 自驾驶感知里 visibility 问题被明确提出并针对性解决

### 少量关键数字

- 论文主打 zero additional inference cost 下的双任务精度提升

### 局限、风险、可迁移点

- **局限**: 更偏 training trick，不会直接产出可交互的 Gaussian asset
- **风险**: 若 flow 预测本身偏差较大，会反向污染 Gaussian supervision
- **可迁移点**: Gaussian-as-loss、visibility-aware 2D supervision bridge、flow-updated differentiable primitives 都很值得迁移

### 实现约束

- 自动驾驶 occupancy / flow 设定
- 核心在训练阶段
- 兼容现有 occupancy architectures

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_VoxelSplat_Dynamic_Gaussian_Splatting_as_an_Effective_Loss_for_Occupancy_and_Flow_Prediction.pdf]]
