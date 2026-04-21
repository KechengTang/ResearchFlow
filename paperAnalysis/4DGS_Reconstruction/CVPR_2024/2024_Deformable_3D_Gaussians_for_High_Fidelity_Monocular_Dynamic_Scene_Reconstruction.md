---
title: "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction"
venue: CVPR
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - deformable-gaussians
  - deformation-mlp
  - monocular-video
  - dynamic-scene
  - real-time
  - status/analyzed
core_operator: 在 canonical 3D Gaussian 场景上学习时间条件 deformation field，把动态重建拆成“静态高斯模板 + 时序变形”，兼顾高保真重建与实时渲染。
primary_logic: |
  输入单目动态视频及 SfM 位姿，先在 canonical space 优化一组 3D Gaussians，
  再用 deformation network 根据时间与空间位置预测当前高斯的形变状态，
  并用 annealing smoothing 训练机制减轻姿态误差对时间插值和真实数据稳定性的影响。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_Deformable_3D_Gaussians_for_High_Fidelity_Monocular_Dynamic_Scene_Reconstruction.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T14:30
updated: 2026-04-18T14:30
---

# Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Deformable_3D_Gaussians_for_High-Fidelity_Monocular_Dynamic_Scene_Reconstruction_CVPR_2024_paper.html) / [arXiv](https://arxiv.org/abs/2309.13101) / [PDF](https://arxiv.org/pdf/2309.13101.pdf) / [GitHub](https://github.com/ingra14m/Deformable-3D-Gaussians)
> - **Summary**: 这篇论文是动态高斯路线里非常关键的早期 backbone。它把“canonical 3DGS + deformation field”这条后来被大量工作沿用的范式真正跑通了，尤其适合你把它当作 4DGS 编辑支撑基线来读。
> - **Key Performance**:
>   - 论文在单目动态场景上报告了明显优于既有方法的画质与速度。
>   - 渲染速度可超过 100 FPS，说明 deformable 3DGS 路线兼顾了质量与实时性。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

它要解决的是：

**动态场景重建既要细节好，又要速度快，而传统隐式动态 NeRF 两边都不够理想。**

隐式方法通常会遇到：

- 细节保真有限
- 真实视频里的 pose 误差会破坏时间平滑
- 很难做到实时渲染

论文的解决方案是把动态场景建模改写成显式高斯 + 变形场。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

它真正重要的地方不只是“把 3DGS 用到动态场景”，而是提出了后来非常有影响的一套分解：

- canonical space 存主体结构
- deformation network 负责时间变化

### The "Aha!" Moment

真正的 aha 是：

**动态场景里很多复杂变化，其实可以通过“先学一个稳定的静态模板，再学如何随时间扭动它”来吸收。**

这套分解的好处很大：

1. 显式高斯保留高保真与实时渲染
2. 动态变化被压缩到 deformation field
3. 时间插值也有了更自然的表达路径

这也是为什么后续很多 4DGS 编辑工作，特别是 Instruct-4DGS 这种 static-dynamic separation 路线，会自然继承它的建模精神。

### 权衡与局限

- 优势：高保真、快、canonical/deformation 分解清晰
- 局限：依赖 pose 质量；极复杂拓扑变化和大遮挡场景仍有挑战

## Part III / Technical Deep Dive

### Pipeline

```text
monocular dynamic video + SfM poses
-> optimize canonical 3D Gaussians
-> learn deformation network over time
-> render deformed Gaussians with differentiable rasterization
-> dynamic novel view synthesis and interpolation
```

### 关键技术点

#### 1. Canonical-Space 3DGS

先学一个稳定 canonical Gaussian scene，是整条路线成立的前提。

#### 2. Time-Conditioned Deformation

动态信息不直接写进每个时刻的独立表示，而是交给 deformation field 统一建模。

#### 3. Annealing Smoothing

论文专门提出 annealing smoothing 来缓解真实视频中相机位姿误差造成的时间不平滑问题，这对真实场景非常关键。

### 关键信号

- 论文在多个动态数据集上展示了明显优于既有方法的质量与速度。
- 100 FPS 级别的实时渲染说明这条路线不仅理论上好看，工程上也成立。

### 对你的价值

对你当前主线，这篇论文属于 **基础骨架级别** 的工作：

- 做 4DGS 编辑时，它是重要的上游表示参考
- 做前馈或语义驱动动态高斯时，它仍然定义了很多后续方法默认采用的 decomposition 方式

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_Deformable_3D_Gaussians_for_High_Fidelity_Monocular_Dynamic_Scene_Reconstruction.pdf]]
