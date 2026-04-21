---
title: "AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction"
venue: CVPR
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - avatar-generation
  - animatable-human
  - single-image-to-3d
  - generative-prior
  - inconsistent-reconstruction
  - status/analyzed
core_operator: 利用 transformer-based video generation model 先从单图生成多视角 canonical-pose 人像与法线，再把这些不完全一致的多视角结果重解释为 4D Gaussian reconstruction 问题，以显式高斯表示吸收视角不一致并获得可实时动画的人体 avatar。
primary_logic: |
  先从单张输入图像生成多视角 canonical pose 图像与 normal maps，
  再将存在不一致的多视角观测转成 4D Gaussian reconstruction 问题，
  通过显式高斯时空建模吸收视角不一致并恢复统一 avatar，
  最终实现单图输入下的 photorealistic、real-time animatable human avatar。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_AniGS_Animatable_Gaussian_Avatar_from_a_Single_Image_with_Inconsistent_Gaussian_Reconstruction.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T16:37
updated: 2026-04-18T16:37
---

# AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2412.02684](https://arxiv.org/abs/2412.02684) · [Project Page](https://lingtengqiu.github.io/2024/AniGS/)
> - **Summary**: AniGS 的关键在于它没有强求生成模型输出完美一致的多视角人像，而是承认单图到多视角 canonical pose 的生成结果天然会不一致，再把这种不一致重新解释成一个 4D Gaussian reconstruction 问题来吸收。这样生成先验和显式高斯重建就形成了互补。
> - **Key Performance**:
>   - 论文强调从 in-the-wild 单图输入即可得到 photorealistic、real-time animatable avatars。
>   - 核心优势是 generalization 与 animation controllability，而不是只做静态 3D 还原。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

单图 animatable avatar 一直卡在两个矛盾上:

- 精细 3D 重建方法常常难以直接动画化
- 纯生成式动画方法虽然易控，但多视角一致性差、每帧推理重

AniGS 要解决的是:

**如何从单张图片出发，既得到高细节 3D avatar，又让它可实时动画，并且能容忍生成多视角中的不一致。**

### 核心能力定义

- **输入**: single in-the-wild human image
- **输出**: 可实时渲染和动画的 Gaussian avatar
- **强项**: 单图人像 avatar、动画性、生成先验 + 显式表示结合
- **弱项**: 主要针对 human avatar，不是通用场景 4D 表示

### 真正的挑战来源

- 单图重建先天有视角歧义
- canonical pose 生成虽然有帮助，但多视角输出往往不完全一致
- 如果强行按一致多视图重建，容易崩；如果只做 2D 动画，又缺全局 3D 一致性

### 边界条件

- 面向人物 avatar，不是开放场景重建
- 依赖预训练 video generation model
- 更偏 animatable human asset creation

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

AniGS 的设计哲学很聪明:

**不要把生成模型输出的不一致当作噪声丢掉，而是把它转成 Gaussian 表示可以吸收的时空变化。**

这意味着 4D Gaussian 在这里不只是动态场景表示，也是一种“处理生成多视角不一致”的缓冲器。

### The "Aha!" Moment

真正的 aha 是:

**单图 avatar 重建里的多视角不一致，可以被解释成一个轻度时变/视变问题，而 4D Gaussian 表示恰好适合承接这种不一致。**

所以作者没有要求完美多视图一致训练数据，而是用 4DGS 去包容这些偏差，并把生成多视角先验转成可动画 3D asset。

### 为什么这个设计有效

生成模型提供:

- canonical pose multi-view prior
- 细节和纹理多样性

Gaussian 表示提供:

- 全局显式 3D 结构
- 实时渲染
- 对不一致的吸收与平滑

两者结合后，比单纯依赖任一方向都更稳。

### 对我当前方向的价值

这篇对你当前方向很有价值，因为它展示了一个很通用的套路:

**生成模型负责补视角和细节，Gaussian 负责把这些结果变成可交互的显式资产。**

对未来 4D 编辑、avatar 驱动和交互式角色系统都很重要。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: avatar 本身就是最典型的 4D 可编辑对象，AniGS 提供的是可动画、高实时的底座。
- **与 3DGS_Editing**: 相比普通 3DGS 人像编辑，这篇更进一步，把可动画性和时序一致性纳入表示目标。
- **与 feed-forward Gaussians**: AniGS 强烈体现出前向生成 + Gaussian 显式重建的组合范式，是 feed-forward Gaussian avatar 的重要参考。

### 战略权衡

- 优点: 单图输入友好，细节和动画性兼顾
- 代价: 强依赖生成模型质量，且领域特定于 human avatar

---

## Part III / Technical Deep Dive

### Pipeline

```text
single human image
-> transformer-based multi-view canonical pose generation
-> generated multi-view images and normals with inconsistencies
-> reinterpret reconstruction as 4D Gaussian problem
-> explicit Gaussian avatar reconstruction
-> real-time animation and rendering
```

### 关键模块

#### 1. Multi-View Canonical Pose Generation

先用生成模型把单图扩展成 canonical pose 下的多视角观测和法线，为后续显式重建提供 richer supervision。

#### 2. Inconsistent Gaussian Reconstruction

作者不假设这些观测完全一致，而是专门设计 4D Gaussian reconstruction 去吸收不一致。

#### 3. Real-Time Animatable Representation

最终目标不是静态 3D 模型，而是可实时动画的人体 avatar，这使其更接近可用资产。

### 关键实验信号

- 论文特别强调 in-the-wild generalization 和 real-time animation
- 4DGS 在这里的价值不只是渲染快，更是对 inconsistent views 的鲁棒性
- avatar 任务比通用 object 任务更强调控制和驱动能力

### 少量关键数字

- 论文以 photorealistic、real-time animation 为核心卖点，重点不是某个单一 benchmark 指标

### 局限、风险、可迁移点

- **局限**: 主要面向人物 avatar，跨类别泛化不一定自然
- **风险**: 如果多视角生成质量太差，4DGS 也难完全吸收不一致
- **可迁移点**: generated-multiview-to-Gaussian asset、inconsistency-tolerant reconstruction、real-time animatable explicit representation 都很值得迁移

### 实现约束

- 单图 human avatar 设定
- 依赖 video generation prior
- 重点是 animatable asset creation

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_AniGS_Animatable_Gaussian_Avatar_from_a_Single_Image_with_Inconsistent_Gaussian_Reconstruction.pdf]]
