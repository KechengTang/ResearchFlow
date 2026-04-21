---
title: "DGNS: Deformable Gaussian Splatting and Dynamic Neural Surface for Monocular Dynamic 3D Reconstruction"
venue: ACMMM
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - neural-surface
  - hybrid-representation
  - geometry-reconstruction
  - monocular-dynamic-scene
  - depth-filtering
  - status/analyzed
core_operator: 将 deformable Gaussian splatting 与 dynamic neural surfaces 组合成 hybrid framework，用 DGS 提供高质量外观与深度监督，用 DNS 负责几何恢复，并通过双向耦合让 geometry 与 rendering 互相增益。
primary_logic: |
  先用 deformable Gaussian splatting 重建动态场景并输出深度图，
  再利用这些深度引导 dynamic neural surface 的射线采样和几何监督，
  同时由 neural surface 反向约束 Gaussian primitives 沿表面分布，
  最终兼顾 monocular dynamic novel-view synthesis 与 3D geometry reconstruction。
pdf_ref: paperPDFs/4DGS_Reconstruction/ACMMM_2025/2025_DGNS_Deformable_Gaussian_Splatting_and_Dynamic_Neural_Surface_for_Monocular_Dynamic_3D_Reconstruction.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:30
updated: 2026-04-18T16:30
---

# DGNS: Deformable Gaussian Splatting and Dynamic Neural Surface for Monocular Dynamic 3D Reconstruction

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2412.03910](https://arxiv.org/abs/2412.03910)
> - **Summary**: DGNS 试图解决一个长期割裂的问题: 动态 Gaussian 方法通常渲染好但几何弱，动态 surface 方法通常几何好但渲染差。它因此用一个 hybrid representation 把两者绑在一起，让 Gaussian 管 appearance，让 neural surface 管 geometry，再通过深度和表面分布做双向耦合。
> - **Key Performance**:
>   - 论文宣称在 3D reconstruction 上达到 SOTA，同时 novel-view synthesis 也保持 competitive。
>   - 关键收益来自 geometry-quality 和 rendering-quality 的同时兼顾，而不是单侧偏科。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

单目动态重建里经常会出现一种两难:

- Gaussian / radiance 方法擅长渲染
- surface / SDF 类方法擅长几何

但现实应用通常要两者都要。  
DGNS 的目标就是:

**在 monocular dynamic reconstruction 里，同时把 novel-view synthesis 和 3D geometry reconstruction 做好。**

### 核心能力定义

- **输入**: monocular dynamic video
- **输出**: 兼顾外观与几何的动态 3D 重建
- **强项**: geometry-rendering tradeoff 改善、深度监督、混合表示
- **弱项**: 系统复杂度高，不是纯高效单一表示

### 真正的挑战来源

- monocular 本来就缺多视角几何约束
- 只做 Gaussian，表面恢复往往不够硬
- 只做 neural surface，视觉渲染又容易差

### 边界条件

- 需要同时维护 Gaussian 和 neural surface 两套子系统
- 更偏 geometry-aware dynamic reconstruction，而不是纯渲染 benchmark
- 仍然是单目场景设定

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

DGNS 的设计哲学是:

**与其让一个表示同时兼顾 appearance 和 geometry，不如让两个擅长不同任务的表示互相监督。**

也就是说，它不是想让 Gaussian 变成 surface，也不是让 surface 变成渲染器，而是让两者各做擅长之事。

### The "Aha!" Moment

真正的 aha 是:

**Gaussian 产生的深度不仅可以用于渲染，还可以成为 dynamic neural surface 的几何监督；反过来 neural surface 也可以约束 Gaussian 更贴近真实表面。**

这使 DGNS 不只是简单拼接两种表示，而是形成了闭环互补。

### 为什么这个设计有效

Gaussian 的深度估计给了 surface 模块更好的采样与监督入口；  
surface 的几何约束又避免 Gaussian 漂离物体表面。  
于是 appearance branch 和 geometry branch 互相补短板，而不是彼此竞争。

### 对我当前方向的价值

这篇论文对你当前方向很重要，因为它明确指出:

**如果你后续重视 editable geometry 或 mesh extraction，单靠 4DGS 可能不够，hybrid representation 值得认真考虑。**

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 需要真实表面支撑的 4D 编辑会非常受益于 DGNS 这种 geometry-aware backbone。
- **与 3DGS_Editing**: 许多 3DGS 编辑方法都受限于不够稳定的几何，DGNS 提供的是补 geometry 这块短板的思路。
- **与 feed-forward Gaussians**: 它不是前向模型，但其 geometry-rendering 双分支互监督很适合未来作为 teacher framework。

### 战略权衡

- 优点: 外观和几何同时受益
- 代价: 两套表示耦合带来更高实现和训练复杂度

---

## Part III / Technical Deep Dive

### Pipeline

```text
monocular dynamic video
-> deformable Gaussian splatting branch for appearance and depth
-> depth-guided sampling and supervision for dynamic neural surface
-> neural surface regularizes Gaussian distribution around surfaces
-> improved geometry reconstruction and novel-view rendering
```

### 关键模块

#### 1. DGS Module

Gaussian 分支负责 appearance reconstruction，并输出深度图作为后续几何模块的重要中间量。

#### 2. DNS Module

dynamic neural surface 分支专门负责更可靠的 geometry reconstruction。

#### 3. Bidirectional Coupling

深度从 Gaussian 到 surface，表面约束从 surface 回到 Gaussian。  
这一步是 DGNS 真正区别于简单多模块拼接的地方。

#### 4. Depth Filtering

作者还引入 depth-filtering 来提升 depth supervision 的可信度，说明几何监督质量本身也是关键。

### 关键实验信号

- 论文主打“geometry SOTA + rendering competitive”的双指标组合
- 在论文叙事里，方法的价值正是打破二者偏科
- depth-filtering 被单列为贡献，说明深度监督质量确实重要

### 少量关键数字

- 论文强调 state-of-the-art 3D reconstruction 与 competitive novel-view synthesis

### 局限、风险、可迁移点

- **局限**: 系统比单一 4DGS 更重，训练调参与模块协同更复杂
- **风险**: 若深度监督不准，可能同时误导 Gaussian branch 和 surface branch
- **可迁移点**: hybrid Gaussian-surface representation、bidirectional geometry-appearance coupling、depth filtering 都很值得迁移

### 实现约束

- 单目动态视频
- 双分支耦合训练
- 目标是平衡 geometry 与 rendering

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/ACMMM_2025/2025_DGNS_Deformable_Gaussian_Splatting_and_Dynamic_Neural_Surface_for_Monocular_Dynamic_3D_Reconstruction.pdf]]
