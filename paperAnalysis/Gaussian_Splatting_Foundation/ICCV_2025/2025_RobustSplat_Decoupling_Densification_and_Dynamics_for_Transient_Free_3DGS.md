---
title: "RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS"
venue: ICCV
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - transient-object
  - densification
  - robust-reconstruction
  - mask-bootstrapping
  - static-scene-cleanup
  - status/analyzed
core_operator: 通过延迟 Gaussian growth 把 densification 从早期动态干扰中解耦出来，并使用 scale-cascaded transient mask bootstrapping 先粗后细识别 transient 区域，从而避免 3DGS 在有瞬态物体的场景中把干扰学进静态场景表示。
primary_logic: |
  先延迟 Gaussian splitting/cloning，让模型优先稳定静态结构，
  再用 low-to-high resolution 的 transient mask bootstrapping 获得可靠瞬态区域估计，
  随后在去除 transient 干扰的条件下再执行 densification 与优化，
  最终得到更干净、transient-free 的 3D Gaussian scene。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_RobustSplat_Decoupling_Densification_and_Dynamics_for_Transient_Free_3DGS.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T16:41
updated: 2026-04-18T16:41
---

# RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2506.02751](https://arxiv.org/abs/2506.02751) · [Project Page](https://fcyycf.github.io/RobustSplat/)
> - **Summary**: RobustSplat 关注的是现实静态场景重建里一个常见但很烦的问题: transient objects 会被 densification 学进去，最后在静态 3DGS 里留下伪影。它因此把 densification 和 transient dynamics 解耦，先让静态结构站稳，再逐步识别并剔除瞬态干扰。
> - **Key Performance**:
>   - 论文在多个 challenging datasets 上优于现有 transient-robust methods。
>   - 核心收益来自更干净的 transient-free reconstruction，而不是单纯的 point growth 技巧。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

很多 3DGS 假设多视图都来自同一静态场景。  
现实中却常常会出现:

- 行人路过
- 车短暂停留
- 其他瞬态干扰物

这些 transient objects 一旦在 densification 早期被模型吸收，就会长期留在结果里。  
RobustSplat 要解决的是:

**如何让 3DGS 在存在 transient objects 的多视图输入下仍重建出更干净的静态场景。**

### 核心能力定义

- **输入**: 含 transient objects 的多视图图像
- **输出**: transient-free static 3DGS
- **强项**: 鲁棒静态重建、干扰剔除、densification 过程修正
- **弱项**: 重点是 transient cleanup，不处理连续动态场景建模

### 真正的挑战来源

- densification 越早开始，越容易把 transient 干扰错误学成静态结构
- 若 transient mask 初始不准，高斯增长也会跟着跑偏
- 静态结构和 transient 干扰在图像里经常混杂

### 边界条件

- 目标是 transient-free static reconstruction
- 不适用于真正想保留动态体的 4D scene modeling
- 依赖 transient mask 逐级估计

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

RobustSplat 的哲学是:

**densification 不应该在场景还没分清静态和瞬态时就毫无节制地开始。**

因此作者做了两层解耦:

- 时间上先延迟 Gaussian growth
- 空间上先粗后细地估计 transient masks

### The "Aha!" Moment

真正的 aha 是:

**transient artifacts 很大程度上不是由渲染器引起，而是由过早 densification 把瞬态信息固化进了 Gaussian scene。**

一旦延迟 growth，让静态结构先成形，再做 mask-guided densification，很多瞬态伪影自然就不会长出来。

### 为什么这个设计有效

低分辨率特征更稳、更语义一致，所以适合先做粗 transient localization；  
高分辨率再细化边界，则能得到更精确的 mask。  
这种 scale-cascaded bootstrapping 比一开始就追求精细掩码更稳。

### 对我当前方向的价值

这篇对你当前方向很重要，因为它提醒一个基础事实:

**Gaussian densification 不是中立操作，它本身会决定模型学进什么。**

如果后面做编辑、清理、鲁棒重建，这类 densification-aware robustness 方法都非常关键。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 若 4D 编辑想先得到干净静态背景再叠动态效果，这篇的 transient cleanup 思路很实用。
- **与 3DGS_Editing**: 编辑前先清除 transient artifacts 往往能显著提升结果，RobustSplat 很适合作为 preprocessing。
- **与 feed-forward Gaussians**: delayed growth 和 mask bootstrapping 可转成前向 Gaussian 系统的 refinement / cleaning stage。

### 战略权衡

- 优点: 思路扎实，直接解决现实采集数据中的 transient noise
- 代价: 更偏 cleanup，不增加动态建模能力

---

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view images with transient objects
-> delay Gaussian growth
-> low-resolution transient mask estimation
-> high-resolution mask refinement
-> densification after static structure stabilizes
-> transient-free 3DGS reconstruction
```

### 关键模块

#### 1. Delayed Gaussian Growth

先稳住静态结构，再允许 Gaussian splitting/cloning。  
这是防止 transient 被过早固化的第一道防线。

#### 2. Scale-Cascaded Mask Bootstrapping

先低分辨率估计 transient 区域，再高分辨率细化。  
这样比一步到位更鲁棒。

#### 3. Densification-Dynamics Decoupling

作者明确把点增长和瞬态干扰处理拆开，这本身就是很值得借鉴的 optimization design。

### 关键实验信号

- 论文在多个 challenging datasets 上验证，说明 transient 问题确实是现实数据中的普遍痛点
- qualitative comparison 里 transient artifacts 的减少非常直观
- 其方法更像 pipeline-level robustness upgrade

### 少量关键数字

- 论文主打 consistent outperformance on transient-affected datasets，而不是单一 benchmark 数字

### 局限、风险、可迁移点

- **局限**: 只适合 transient-free static reconstruction 目标
- **风险**: mask bootstrapping 若失误，可能把真实静态结构误删
- **可迁移点**: delayed densification、transient mask bootstrapping、artifact-aware optimization 都很值得迁移

### 实现约束

- 静态场景中含 transient objects 的数据
- 依赖多阶段 mask 估计
- 更偏 robust cleanup

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_RobustSplat_Decoupling_Densification_and_Dynamics_for_Transient_Free_3DGS.pdf]]
