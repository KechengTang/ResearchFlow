---
title: "Fully Explicit Dynamic Gaussian Splatting"
venue: NeurIPS
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - explicit-representation
  - static-dynamic-separation
  - temporal-opacity
  - real-time
  - status/analyzed
core_operator: 将动态高斯的运动显式离散到稀疏时间戳，只存关键时刻的位置与旋转并用插值恢复连续运动，同时显式分离静态与动态高斯以降低动态建模成本。
primary_logic: |
  输入动态视频与点云先验，训练时先用 motion-based trigger 区分 static/dynamic Gaussians，
  再只在 sparse timestamps 上为 dynamic Gaussians 显式采样位姿并用插值与 temporal opacity 建模连续时间，
  同时配合 progressive training 和 point-backtracking 稳定长时序动态高斯优化。
pdf_ref: paperPDFs/4DGS_Reconstruction/NeurIPS_2024/2024_Fully_Explicit_Dynamic_Gaussian_Splatting.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T14:30
updated: 2026-04-18T14:30
---

# Fully Explicit Dynamic Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2410.15629) / [PDF](https://arxiv.org/pdf/2410.15629.pdf)
> - **Summary**: 这篇论文很值得放进你的 4DGS 支撑层，因为它把动态高斯从“每时刻都要算一遍”的隐式变形逻辑，改成了“关键时间点显式存储 + 插值恢复”的表示方式，还显式做了静动分离，这对后续编辑接口设计很有启发。
> - **Key Performance**:
>   - 论文报告在多场景上达到 SOTA 级别的渲染质量。
>   - 在 2080Ti 上可达约 62 FPS，说明显式时序表示也可以兼顾效率。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

Fully Explicit Dynamic Gaussian Splatting 针对的是动态 3DGS 里一个非常核心的工程问题：

**动态一旦复杂，建模和渲染成本就会随着时间维度快速上升。**

如果每个时间都显式维护一整套状态，代价很大；
如果完全依赖隐式 deformation，又容易牺牲显式表示的效率优势。

因此论文的目标是：
**在保留显式 3DGS 优势的同时，把动态时间维压缩到更可控的表示上。**

## Part II / High-Dimensional Insight

### 方法真正新在哪里

它真正新的地方是把动态高斯的时间表示做成了 **sparse explicit keyframes**：

- 不是每个时间都存全部状态
- 而是在稀疏时间点显式记录位置与旋转
- 中间时刻再用插值补足

### The "Aha!" Moment

真正的 aha 是：

**动态场景里的很多运动并不需要“每帧重新发明”，只需要一个足够稳定的关键时刻表示，再把中间过程插值出来。**

这件事一旦成立，就会产生两个很重要的副作用：

1. 时序参数量下降
2. 编辑接口更清楚，因为关键时间点天然更像控制节点

再加上静态/动态高斯分离，这篇论文实际上已经很接近为后续编辑系统准备“可控基础设施”。

### 权衡与局限

- 优势：显式、快、静动分离明显
- 局限：对非平滑运动、拓扑变化和强遮挡重出现象，单纯插值会有边界

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic scene data
-> separate static and dynamic Gaussians
-> explicitly sample dynamic positions / rotations at sparse timestamps
-> interpolate motion and temporal opacity
-> progressive training + point-backtracking
-> fast dynamic Gaussian rendering
```

### 关键技术点

#### 1. Static / Dynamic Separation

论文首先做静动态高斯区分，这一步不仅是为了建模效率，也让动态部分的时间参数化更聚焦。

#### 2. Sparse Temporal Sampling

动态高斯不在每个时间点都直接优化，而是在关键时间点显式采样状态，用插值恢复连续时间。

#### 3. Progressive Training and Point-Backtracking

作者还设计了 progressive training 和 point-backtracking 来提高长时序训练稳定性，避免错误高斯在时间上持续累积。

### 关键信号

- 62 FPS 是很强的工程信号，说明“显式时间表示”确实能换来速度收益。
- 同时论文报告在主流动态数据集上具有很强的画质竞争力。

### 对你的价值

这篇论文不一定是你 4DGS 编辑的主 baseline，但它很值得作为 **表示层参考**：

- 如果你想做可控编辑，静动分离很重要
- 如果你想做高效 4D 表示，关键时间点显式化很重要

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/NeurIPS_2024/2024_Fully_Explicit_Dynamic_Gaussian_Splatting.pdf]]
