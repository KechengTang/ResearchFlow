---
title: "Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting"
venue: ICLR
year: 2026
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - feed-forward
  - stylization
  - image-guided-editing
  - cross-view-consistency
  - cross-scene-generalization
  - status/analyzed
core_operator: 保留 VGGT 风格几何骨干的 self-attention 路径专门预测深度与相机，再用全局 CrossBlock 把 style image 的 token 通过 cross-attention 注入外观分支，并以 voxel-level 3D style loss 在融合后的 3D 体素特征上对齐风格统计。
primary_logic: |
  输入一张风格参考图和一到多张无位姿内容图，
  先用几何骨干在单次前向中恢复相机、深度和高斯几何参数，
  再让 Style Aggregator 用全局 CrossBlock 把 style token 注入内容特征，并由 style head 预测风格化颜色系数，
  随后通过 Gaussian adapter 合成可渲染的 stylized 3DGS，
  训练时再用 3D voxel-level style loss 对多视图融合后的 3D 特征做统计对齐，以强化跨视角一致性。
pdf_ref: paperPDFs/3DGS_Editing/ICLR_2026/2026_Stylos_Multi_View_3D_Stylization_with_Single_Forward_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-18T18:40
updated: 2026-04-18T18:40
---

# Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://hanzhouliu.github.io/Stylos/) · [GitHub](https://github.com/HanzhouLiu/Stylos) · [arXiv 2509.26455](https://arxiv.org/abs/2509.26455)
> - **Summary**: Stylos 的目标不是“把某个场景重新优化成艺术风格”，而是让 3D 风格化第一次真正具备 feed-forward、零样本、无位姿输入和跨场景泛化能力。它把几何和风格彻底拆成两条路径，让几何保持稳定，再把风格通过 cross-attention 写进外观分支。
> - **Key Performance**:
>   - 在 `Train / Truck / M60 / Garden` 四个场景上，Stylos 在短程和长程一致性指标上都排第一，例如短程 `LPIPS` 为 `0.030 / 0.028 / 0.035 / 0.047`，长程 `LPIPS` 为 `0.051 / 0.074 / 0.083 / 0.139`。
>   - 在艺术性指标上保持强竞争力，同时推理仅需 `0.05s`，明显快于需要 per-scene fitting 的 `StyleGaussian / G-Style / SGSST`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

Stylos 真正瞄准的是 3D 风格化里长期存在但很少被同时解决的三个约束：

**不做 per-scene optimization、输入没有预估位姿、风格结果还要跨视角一致。**

此前方法要么：

- 依赖 NeRF 或 3DGS 的逐场景训练，速度慢；
- 只能处理少量视图；
- 或者风格能迁过去，但跨视角一致性不足，尤其难在更大场景和更多视图上稳定工作。

Stylos 的定位非常清晰：做一个真正可泛化的 single-forward 3D stylization pipeline。

### 核心能力定义

- **输入**：一张风格参考图，以及一张到多张内容视图，且不要求预先给定相机位姿。
- **输出**：可以直接渲染的 stylized 3D Gaussian scene。
- **擅长**：静态场景的全局风格迁移、跨场景泛化、多视图一致风格表达。
- **不擅长**：对象级几何编辑、动态 4D 场景、超高分辨率细节控制。

### 真正的挑战来源

- 风格化如果直接改动几何骨干，很容易破坏深度和相机预测稳定性。
- 传统 2D 风格损失只看单帧统计，不能保证多视图共享的 3D 结构一致。
- 无位姿输入意味着模型既要恢复 geometry，又要做 style transfer，天然容易耦合过重。

### 边界条件

- 论文聚焦静态 3DGS stylization，而不是局部对象级编辑。
- 视图数虽然可以扩展到很多，但作者观察到 batch 中视图数超过 `32` 后质量会逐渐下降。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

Stylos 的设计哲学可以概括成一句话：

**几何必须有一条不被风格污染的稳定主路，风格只能作为条件信息附着在外观路径上。**

于是作者把系统拆成两部分：

- 几何骨干沿用 VGGT 风格的 alternating self-attention，用来做相机、深度和高斯几何。
- 风格聚合器则通过 `CrossBlock` 把 style token 插入内容 token，专门负责颜色和外观。

### The "Aha!" Moment

真正的 aha 是：

**3D 风格一致性问题，不能只在每个视图的 2D feature 上做统计匹配，而要在融合后的 3D 空间里对齐风格统计。**

所以作者没有停留在 image-level 或 scene-level 的 style loss，而是继续往前走了一步，提出 `voxel-level 3D style loss`：

- 先把多视图特征通过可微反投影融合到体素空间；
- 再在这个 3D 体素特征上与 style image 的统计量对齐。

这意味着风格约束不再是“每帧都像风格图”，而是“整个 3D 场景在融合后都遵循同一套风格统计”。

### 为什么这个设计有效？

- 让几何 backbone 只保留 self-attention，可以最大化继承 VGGT 的结构恢复能力。
- 风格注入发生在 `CrossBlock`，因此 style conditioning 是显式、可控、可拔插的。
- 全局 CrossBlock 允许不同视图共享 style-conditioned 信息，显著好于逐视图独立的 frame-only 设计。
- 3D voxel style loss 从根上加强了跨视角一致性，而不仅仅是事后做 view-consistency 约束。

### 战略权衡

- **优点**：速度快、无需 per-scene fitting、无位姿输入、泛化能力强。
- **局限**：本质上仍是 appearance stylization system，几何编辑和局部精修不是它的主战场。

## Part III / Technical Deep Dive

### Pipeline

```text
style image + unposed content views
-> VGGT-style geometry backbone predicts depth, pose, and Gaussian geometry
-> Style Aggregator injects style tokens through CrossBlocks
-> style head predicts stylized SH/color coefficients
-> Gaussian adapter assembles stylized 3DGS
-> render stylized views
-> train with reconstruction, content, clip, TV, and voxel-level 3D style losses
```

### 关键模块

#### 1. Geometry Backbone

几何骨干基本继承 VGGT 的 alternating attention 设计。  
这意味着 Stylos 的几何恢复并不是从零设计一套 stylization-specific backbone，而是直接站在强力 pose-free reconstruction 模型之上。

#### 2. CrossBlock Style Injection

`CrossBlock` 把 style token 作为 `K/V`，内容 token 作为 `Q`。  
作者比较了 `Frame`、`Global` 和混合式设计，发现 **Global CrossBlock** 在几何保真和风格一致性上都更稳。

#### 3. Voxel-level 3D Style Loss

这是整篇最关键的 operator。  
作者先把多视图 VGG 特征融合成 3D 体素特征，再与风格图的均值和标准差做对齐，从而把风格一致性从 2D 图像层提升到 3D 场景层。

### 关键实验信号

- 在 CO3D 消融中，`Global CrossBlock` 比 `Frame` 和混合式更能保住披萨边界和纹理细节。
- 相比 image-level style loss，`3D loss` 在 `ArtScore` 上从 `4.78` 提升到 `9.15`，并带来更稳的短程和长程一致性。
- 在 `Train / Truck / M60 / Garden` 上，Stylos 在短程和长程 `LPIPS / RMSE` 都是最优，说明它并不是只在单帧更“像画”，而是真正提升了多视图一致性。
- 速度上仅 `0.05s`，而且不需要 per-scene training，这对 3D 内容生产非常关键。

### 方法边界与启发

- 它证明了“几何主干冻结式保结构，风格分支单独控制外观”这条路是成立的。
- 对后续 3DGS/4DGS 外观编辑研究，`CrossBlock + 3D style loss` 是很值得复用的一组 operator。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ICLR_2026/2026_Stylos_Multi_View_3D_Stylization_with_Single_Forward_Gaussian_Splatting.pdf]]
