---
title: "DGS-LRM: Real-Time Deformable 3D Gaussian Reconstruction From Monocular Videos"
venue: NeurIPS
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - feed-forward
  - monocular-video
  - transformer
  - scene-flow
  - motion-tracking
  - status/analyzed
core_operator: 用 large reconstruction model 前馈预测 deformable 3D Gaussian splats 与跨时刻 scene flow，把动态场景重建从 per-scene optimization 改成一次前向推理。
primary_logic: |
  输入带位姿的单目视频，先由 tokenizer 与 transformer 编码时序观测，
  再预测每个像素对应的 3D Gaussian 以及它到其他时刻的 scene flow，
  最后通过 warping 这些高斯实现动态新视角合成，并把物理一致的变形轨迹复用于长时跟踪。
pdf_ref: paperPDFs/4DGS_Reconstruction/NeurIPS_2025/2025_DGS_LRM_Real_Time_Deformable_3D_Gaussian_Reconstruction_From_Monocular_Videos.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T14:30
updated: 2026-04-18T14:30
---

# DGS-LRM: Real-Time Deformable 3D Gaussian Reconstruction From Monocular Videos

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2506.09997) / [PDF](https://arxiv.org/pdf/2506.09997.pdf)
> - **Summary**: DGS-LRM 是你当前“前馈式高斯”关注线里非常应该重点看的论文。它第一次把任意动态场景的 deformable 3DGS 重建做成真正的 feed-forward large reconstruction model，而不是每个场景都重新优化几个小时。
> - **Key Performance**:
>   - 论文报告其重建质量可接近 optimization-based 动态重建方法。
>   - 同时明显优于已有 predictive dynamic reconstruction baseline，并且还能直接支持 long-range 3D tracking。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

DGS-LRM 解决的是一个很直接也很难的问题：

**前馈重建模型在静态场景上已经跑起来了，但到了动态场景，大多数方法不是质量不够，就是表示不对，无法稳定表达运动。**

这件事难在三层：

- 训练动态前馈模型需要足够大的数据
- 表示必须既好学又能表达时间变化
- 网络要在一次前向里输出可渲染、可跟踪的动态 3D 结构

## Part II / High-Dimensional Insight

### 方法真正新在哪里

DGS-LRM 不是只换了一个更大的 transformer，而是把三件事同时补齐了：

1. 大规模合成动态数据集
2. 适合前馈预测的 deformable 3DGS 表示
3. 端到端的大模型重建架构

### The "Aha!" Moment

真正的 aha 在于：

**动态前馈重建做不起来，通常不是单纯网络不够大，而是“表示、监督和目标任务没有对齐”。**

DGS-LRM 的对齐方式是：

- 表示上：每像素 deformable 3D Gaussian
- 监督上：多视图视频 + 稠密 3D scene flow
- 任务上：既要 novel view synthesis，也要长时 3D tracking

这样输出的 3DGS 不只是“看起来像”，而是带着可以跨时间传播的物理/几何对应关系。

对你而言，这篇论文的重要性很高，因为你明确关注 feed-forward 高斯，而它正是动态前馈路线的代表作。

### 权衡与局限

- 优势：前馈、泛化、实时、还能兼顾跟踪
- 局限：依赖 posed monocular video 与大规模训练；极端运动与分布外场景仍可能受限

## Part III / Technical Deep Dive

### Pipeline

```text
posed monocular video
-> input tokenizer
-> large transformer
-> predict deformable 3D Gaussians + cross-time scene flow
-> warp Gaussians over time
-> dynamic novel view synthesis and long-range 3D tracking
```

### 关键技术点

#### 1. Deformable 3DGS Representation

论文没有直接预测静态 3DGS，而是预测随时间可变的高斯和对应 scene flow，这对动态场景是更自然的输出空间。

#### 2. Dataset and Supervision

作者专门扩展了大规模合成动态数据，并提供 ground-truth multi-view videos 与 dense 3D scene flow supervision，这一步是前馈路线能站住脚的关键。

#### 3. Reconstruction + Tracking Unification

一个很值得记住的点是，DGS-LRM 不是把 tracking 作为附加实验，而是让表示本身就能自然支持 tracking。

### 关键信号

- 文中明确指出它在真实场景上优于已有 predictive baseline。
- 与 optimization-based 方法接近的质量，是这篇论文真正的能力跳跃信号。

### 对你的价值

对你当前的研究路线，这篇论文值得进高优先级工作集，因为它同时覆盖了：

- 4D/动态高斯
- feed-forward
- scene flow / tracking / dynamic consistency

如果你后面考虑“语义驱动的前馈 4DGS 编辑”或“先预测后编辑”的路线，这篇论文几乎是必读基线。

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/NeurIPS_2025/2025_DGS_LRM_Real_Time_Deformable_3D_Gaussian_Reconstruction_From_Monocular_Videos.pdf]]
