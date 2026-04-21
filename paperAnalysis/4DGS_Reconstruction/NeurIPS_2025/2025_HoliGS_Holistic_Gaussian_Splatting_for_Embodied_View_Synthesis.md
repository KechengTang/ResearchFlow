---
title: "HoliGS: Holistic Gaussian Splatting for Embodied View Synthesis"
venue: NeurIPS
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - embodied-view-synthesis
  - monocular-long-video
  - hierarchical-warping
  - invertible-neural-flow
  - status/analyzed
core_operator: 针对长时 monocular embodied videos，将场景分解为 static background 与 time-varying foreground，并对前景组合使用 global rigid transform、skeleton-driven articulation 与 invertible neural flow 的层次形变，以支持大视角变化和多 actor 交互下的 EVS。
primary_logic: |
  先从长时 monocular RGB video 中分离静态背景和动态前景，
  再为动态前景构建完整 canonical shape，并依次施加 global rigid motion、skeleton articulation 与 invertible neural flow 的层次变形，
  最终在大幅相机轨迹变化和多主体交互条件下实现 embodied novel-view synthesis、深度与 mesh 恢复。
pdf_ref: paperPDFs/4DGS_Reconstruction/NeurIPS_2025/2025_HoliGS_Holistic_Gaussian_Splatting_for_Embodied_View_Synthesis.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:26
updated: 2026-04-18T16:26
---

# HoliGS: Holistic Gaussian Splatting for Embodied View Synthesis

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2506.19291](https://arxiv.org/abs/2506.19291)
> - **Summary**: HoliGS 关注的是一个比普通 4DGS 更苛刻的场景: 长时、单目、相机跟着 actor 运动的 embodied view synthesis。它不是只学一个 deformation field，而是把动态前景拆成全局刚体、骨架驱动和局部非刚体三层 warping，从而支撑人、动物、多主体交互以及 egocentric / third-person follow 这种大视角变化。
> - **Key Performance**:
>   - 论文强调在长时 monocular embodied captures 上优于现有 monocular deformable NeRF/GS 方法。
>   - 同时显著降低训练和渲染时间，定位是更可扩展的 EVS 方案，而不是短片段 demo。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

Embodied View Synthesis 和普通动态重建不一样，它要求:

- 相机轨迹本身很激进
- 可能持续数分钟
- 有多人/动物交互
- 需要 actor-centric 视角，比如跟拍、狗视角、第三人称跟随

传统动态 4DGS 或 deformable NeRF 往往只能处理短片段或较弱非刚体运动。  
HoliGS 的目标是:

**让单目长视频里的复杂 embodied camera motion 和多主体交互也能被稳定重建与自由视角查询。**

### 核心能力定义

- **输入**: long monocular RGB videos
- **输出**: embodied novel views、深度、mesh 与 actor-centric visualization
- **强项**: 长时序、大视角变化、多 actor interaction
- **弱项**: 不是轻量通用 4DGS baseline，系统结构较复杂

### 真正的挑战来源

- embodied camera trajectory 变化远大于常规 NVS 设定
- 单目长视频缺乏多视角约束
- 多 actor 交互与动物运动包含强非刚体和遮挡变化

### 边界条件

- 明显针对 EVS 场景设计
- 依赖 foreground canonical shape 与层次化 warping
- 更偏 actor-centric dynamic content，而不是普通静态背景主导场景

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

HoliGS 的方法哲学是:

**复杂 embodied motion 不能被一个单一 deformation 统一吸收，必须按运动层次拆开建模。**

所以作者把动态前景的运动分成:

- global rigid transformation
- skeleton-driven articulation
- subtle non-rigid deformation

这是一种明显的 hierarchical warping 设计。

### The "Aha!" Moment

真正的 aha 是:

**要在单目长视频中稳定重建 actor-centric 视角，必须先构建完整 canonical foreground shape，再让不同尺度和类型的运动依次作用在它上面。**

这样做后，大幅视角变化和局部非刚体形变不再互相干扰。  
global motion 负责大结构，skeleton 负责 articulated body，invertible neural flow 负责剩余细节。

### 为什么这个设计有效

因为 EVS 里的运动天然分层:

- 主体整体在场景里怎么移动
- 身体/骨架怎么转动
- 局部毛发、肌肉、衣物如何细微变形

若全部交给一个 deformation，既难训练也难泛化。  
层次化建模则更符合真实运动结构。

### 对我当前方向的价值

这篇对你当前方向很有启发，因为它把 4DGS 从一般动态重建推进到了 **actor-centric, camera-aware 4D reconstruction**。  
如果你以后做 4D 编辑、行为分析或跟拍式视角生成，这类层次 warping 很值得关注。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: actor-specific removal、dog’s-eye view、third-person follow 这些应用已经和编辑很接近，说明 HoliGS 的表示本身具备强 actor-centric controllability。
- **与 3DGS_Editing**: 3D 编辑通常只改某个主体的空间属性，HoliGS 进一步处理了时间和相机轨迹耦合，是更完整的动态载体。
- **与 feed-forward Gaussians**: 它不是 feed-forward，但 hierarchical warping 和 canonical foreground shape 很适合成为前向 EVS 模型的结构先验。

### 战略权衡

- 优点: 面向更真实的 embodied capture，层次运动建模非常完整
- 代价: 系统复杂，且更偏 actor-centric 场景

---

## Part III / Technical Deep Dive

### Pipeline

```text
long monocular RGB video
-> decompose scene into static background and dynamic foreground
-> build complete canonical foreground shape
-> apply global rigid transformation
-> apply skeleton-driven articulation
-> apply invertible neural flow for subtle non-rigid motion
-> embodied novel-view rendering / depth / mesh
```

### 关键模块

#### 1. Static Background + Dynamic Foreground Split

作者先做经典的 background / foreground 分离，但重点放在前景的完整 canonical 建模。

#### 2. Hierarchical Warping

这是整篇方法的核心。  
全局刚体、骨架驱动和局部 flow 三层组合，使 EVS 的复杂运动被拆成更容易学习的子问题。

#### 3. Invertible Neural Flow

对于 skeleton 无法解释的细节运动，作者再加一层 invertible flow 来吸收细粒度非刚体变化。

### 关键实验信号

- 论文不只看普通 novel view，也展示 actor-specific views、对象移除和行为轨迹可视化
- 这说明其表示不仅可渲染，还可支撑更高层的 embodied applications
- 长时 monocular capture 本身就是它的重要实验难点

### 少量关键数字

- 论文重点强调 relative gains in quality and speed，而不是单一 benchmark 数字

### 局限、风险、可迁移点

- **局限**: 方法重心在 actor-centric embodied scenes，对一般动态街景的直接适配有限
- **风险**: 若 canonical foreground shape 建不好，后续三层 warping 都会受影响
- **可迁移点**: hierarchical warping、foreground-centric canonical modeling、camera-aware dynamic representation 都很值得迁移

### 实现约束

- 长时单目视频
- 依赖层次变形和 foreground canonical shape
- 更偏 EVS than general 4D benchmark

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/NeurIPS_2025/2025_HoliGS_Holistic_Gaussian_Splatting_for_Embodied_View_Synthesis.pdf]]
