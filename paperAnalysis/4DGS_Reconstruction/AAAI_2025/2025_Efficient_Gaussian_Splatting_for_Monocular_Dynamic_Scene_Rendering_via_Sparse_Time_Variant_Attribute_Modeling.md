---
title: "Efficient Gaussian Splatting for Monocular Dynamic Scene Rendering via Sparse Time-Variant Attribute Modeling"
venue: AAAI
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - monocular-dynamic-scene
  - sparse-anchor-grid
  - time-variant-attributes
  - efficient-rendering
  - status/analyzed
core_operator: 用 sparse anchor-grid 承载时不变几何与外观，只对动态 anchors 查询 time-variant attributes，并以 kernel-based motion flow 生成 dense Gaussians，从而同时压缩高斯数量与稳定静态区域。
primary_logic: |
  先用 sparse anchor-grid 初始化场景几何和静态外观，
  再通过无监督静动态筛选仅对动态 anchors 输入 MLP 查询时间相关属性，
  同时用 kernel-based motion flow 从 anchors 派生 dense Gaussians 的运动与渲染属性，
  最终以更少高斯完成单目动态场景的高质量实时渲染。
pdf_ref: paperPDFs/4DGS_Reconstruction/AAAI_2025/2025_Efficient_Gaussian_Splatting_for_Monocular_Dynamic_Scene_Rendering_via_Sparse_Time_Variant_Attribute_Modeling.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:07
updated: 2026-04-18T16:07
---

# Efficient Gaussian Splatting for Monocular Dynamic Scene Rendering via Sparse Time-Variant Attribute Modeling

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2502.20378](https://arxiv.org/abs/2502.20378)
> - **Summary**: 这篇论文的核心不是再提升一点动态高斯表达力，而是抓住 4DGS 里最实际的系统瓶颈: 动态场景为了拟合所有时刻和视角会堆出大量冗余高斯，最终拖慢渲染并让静区抖动。EDGS 通过把时间变化收缩到少量动态 anchors 上，把动态 4DGS 做成更稀疏、更快、也更稳的表示。
> - **Key Performance**:
>   - 论文强调在两个真实动态场景数据集上同时提升渲染质量和速度，代表性结果图给出约 `128 FPS / 28.2 PSNR @ 540x960`。
>   - 其核心收益来自显著减少高斯数量，而不是单纯换更大的 deformation network。

---

## Part I / The "Skill" Signature

### 这篇论文到底在解决什么问题

EDGS 针对的是单目动态 3DGS 里一个很具体但很关键的问题:

- 现有 deformable Gaussian 方法为了拟合所有时间步，往往会保留太多高斯
- 静态区域本来不该随时间变化，却也被迫走 time-varying 建模
- 最终结果是渲染越来越慢，静态区域还容易抖

所以它真正要解的不是“能不能重建动态场景”，而是:

**能不能在不明显掉质的前提下，把动态高斯场景做得更稀疏、更快、并且对静区更稳定。**

### 核心能力定义

- **输入**: 标定后的 monocular dynamic video
- **输出**: 可随时间查询并实时渲染的动态 Gaussian scene
- **擅长**: 稀疏化动态表示、减少静区时间冗余、提升渲染速度
- **不擅长**: 极复杂拓扑变化、强语义可控编辑、单次前向预测

### 真正的挑战来源

- 动态场景里不是所有高斯都需要时间变化，但传统方法很难显式区分
- 4DGS 的速度瓶颈常常不是网络，而是高斯总数
- 单目场景本身就有观测不足问题，如果盲目稀疏化容易直接损伤细节

### 边界条件

- 它仍然是 per-scene optimization 路线，不是 feed-forward 4D 重建
- 方法假设静态区域占有相当比例，这样静动态拆分才真正有利
- 如果整场都是大幅时变形变，稀疏时间属性的收益会变弱

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

EDGS 的设计非常明确: **不要让每个 Gaussian 都承担完整时间建模责任，而是把动态自由度集中到更稀疏的 anchor 层。**

它把动态场景拆成三类对象:

- 时不变几何与外观
- 时变属性
- 运动流

这比“所有量都扔给 deformation MLP”更像一个有结构的系统分工。

### The "Aha!" Moment

真正的 aha 在于:

**动态 4DGS 里真正需要时间建模的不是每一个 dense Gaussian，而是少量控制动态的 anchors。**

作者因此做了两件事:

1. 用 sparse anchor-grid 承担场景主体表示
2. 用无监督策略把静态 anchors 过滤掉，只让动态 anchors 去查询 time-variant attributes

这样改变后，系统的信息流从“逐高斯逐时刻拟合”变成了“在控制层做时变，再向 dense Gaussians 传播”。  
带来的直接结果是:

- 高斯总量下降
- 静态区域不再被无意义地时间扰动
- 动态区域依然保留足够表达力

### 为什么这个设计有效

因为它抓住了一个常被忽略的事实:

**动态场景的时间复杂度通常远低于像素复杂度，也低于 dense Gaussian 的数量。**

把时变自由度提到 anchor 层，本质是在压缩时序自由度，而不是粗暴压缩空间分辨率。  
这也是它既能提速又不必明显掉质的原因。

### 对我当前方向的价值

对你现在的 4DGS 方向，这篇论文的价值不在于它是最强质量模型，而在于它提供了一个很实用的 operator:

- **time-invariant / time-variant attribute decoupling**
- **dynamic-anchor filtering**
- **sparse control, dense rendering**

这类思路很适合后续做更大场景、在线更新或编辑前的 backbone 压缩。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 它不是编辑方法，但如果编辑系统希望只改动动态主体而保持背景稳定，EDGS 这种静动态筛分会比“全场一起动”更适合作为底座。
- **与 3DGS_Editing**: 3DGS 编辑更强调局部几何或外观可控，而 EDGS 提供的是时间维度上的稀疏控制机制，可作为 3D 编辑扩展到 4D 时的控制层思路。
- **与 feed-forward Gaussians**: 它不是 feed-forward，但其“把复杂自由度压到稀疏控制节点”的做法很适合作为未来 feed-forward 4DGS 的蒸馏目标或结构先验。

### 战略权衡

- 优点: 更快、更紧凑、静态区更稳
- 代价: 动态识别和 anchor 设计一旦失误，可能损伤细节或让局部运动欠拟合

---

## Part III / Technical Deep Dive

### Pipeline

```text
monocular dynamic video
-> sparse anchor-grid initialization
-> split time-invariant and time-variant anchor attributes
-> unsupervised static-anchor filtering
-> kernel-based motion flow to dense Gaussians
-> dynamic Gaussian rendering
```

### 关键模块

#### 1. Sparse Anchor-Grid Representation

作者没有直接存一大堆 dense dynamic Gaussians，而是先建立一个更稀疏的 anchor 层，让场景的几何与外观有一个更紧凑的控制骨架。

#### 2. Sparse Time-Variant Attribute Modeling

只有和可变形物体相关的 anchors 才进入时间相关 MLP。  
这一步直接切掉了大量“其实不需要时间建模”的静态参数。

#### 3. Kernel-Based Motion Flow

dense Gaussians 的运动不是各自独立乱跑，而是由 anchor 层通过经典 kernel 形式传播出来。  
这让动态表达更像“控制驱动的展开”，而不是“海量粒子的独立拟合”。

### 关键实验信号

- 论文强调速度和质量是同时提升，而不是拿画质换速度
- 代表图直接把 **高斯数量、FPS、PSNR** 放在一起比较，说明作者真正优化的是系统表示效率
- 静态区域抖动被点名为旧方法问题，说明它不只做压缩，也在做时间稳定性改进

### 少量关键数字

- 代表性结果图给出约 `128 FPS / 28.2 PSNR @ 540x960`
- 论文实验中显式比较了高斯数量与 FPS 的关系，把“更少 Gaussians -> 更快渲染”作为核心证据

### 局限、风险、可迁移点

- **局限**: 当动态区域非常大、静区很少时，静动态解耦的收益会显著下降
- **风险**: 无监督静动态过滤如果误删动态 anchors，可能让快速细节或小物体运动被过度平滑
- **可迁移点**: 可直接迁移到在线 4DGS、动态场景压缩、编辑前场景清理，以及未来的 feed-forward 4D 表示设计

### 实现约束

- 单目动态重建设定
- 依赖 anchor-based 表示与 MLP 查询
- 关注实时渲染而非最强几何恢复

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/AAAI_2025/2025_Efficient_Gaussian_Splatting_for_Monocular_Dynamic_Scene_Rendering_via_Sparse_Time_Variant_Attribute_Modeling.pdf]]
