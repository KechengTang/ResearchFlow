---
title: "MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting"
venue: NeurIPS
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - deformable-gaussians
  - optical-flow-guided
  - motion-aware
  - pose-estimation
  - monocular-video
  - status/analyzed
core_operator: 从光流中显式分离 camera flow 和 motion flow，并用 motion flow 直接约束高斯形变，再交替优化相机位姿与 3DGS，提升动态单目重建稳定性。
primary_logic: |
  输入单目动态视频，先估计光流并通过 optical flow decoupling 把相机运动和物体运动拆开，
  再用 motion flow 作为显式先验约束 deformable 3DGS 的形变，
  同时引入 camera pose refinement 交替优化高斯和相机位姿，缓解姿态误差对动态重建的破坏。
pdf_ref: paperPDFs/4DGS_Reconstruction/NeurIPS_2024/2024_MotionGS_Exploring_Explicit_Motion_Guidance_for_Deformable_3D_Gaussian_Splatting.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T14:52
updated: 2026-04-18T14:52
---

# MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2410.07707) / [PDF](https://arxiv.org/pdf/2410.07707.pdf) / [Project Page](https://ruijiezhu94.github.io/MotionGS_page/)
> - **Summary**: MotionGS 是很典型的“编辑前应该先补的动态骨干论文”。它指出很多 deformable 3DGS 方法之所以难训，不是因为高斯表示不够，而是因为缺少显式运动约束。于是它把 motion flow 直接变成 deformation 的监督信号。
> - **Key Performance**:
>   - 论文在单目动态场景上报告了明显优于已有方法的质感和指标表现。
>   - 同时保留了 deformable 3DGS 的实时渲染优势。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

很多动态 3DGS 方法虽然能建模时间变化，但常见问题是：

- 运动约束太弱
- 优化容易漂
- 相机位姿误差会进一步破坏动态建模

MotionGS 的目标就是给 deformable 3DGS 加一个更明确的“运动老师”。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

它不是泛泛而谈 motion prior，而是把光流拆成两部分：

- camera flow
- motion flow

然后只把真正和物体运动相关的那一部分拿来约束 Gaussian deformation。

### The "Aha!" Moment

真正的 aha 是：

**如果不先把相机运动和物体运动分开，很多所谓的动态形变监督都会变脏。**

MotionGS 通过 decoupled optical flow 让 deformation field 不再混淆“镜头在动”还是“物体在动”，这会明显提升形变学习质量。

对你后面做 4DGS 编辑很重要，因为编辑时也经常会遇到一个类似问题：
你想控制的是对象运动，而不是相机引起的表观变化。

### 权衡与局限

- 优势：运动监督更明确，对真实单目设置更有帮助
- 局限：依赖光流估计质量；复杂遮挡和非刚体极端运动仍可能困难

## Part III / Technical Deep Dive

### Pipeline

```text
monocular dynamic video
-> optical flow estimation
-> decouple camera flow and motion flow
-> use motion flow to constrain deformable 3DGS
-> alternate Gaussian optimization and pose refinement
-> improved dynamic scene reconstruction
```

### 关键技术点

#### 1. Optical Flow Decoupling

论文把光流拆成相机分量和对象分量，这是整篇方法的核心支点。

#### 2. Explicit Motion Guidance

motion flow 被用来直接约束 Gaussian deformation，而不是只作为旁路辅助。

#### 3. Camera Pose Refinement

作者还引入交替式的 pose refinement，避免错误位姿持续污染动态重建。

### 关键信号

- 论文把优势集中体现在 monocular dynamic scenes，这比只在合成数据上有效更有说服力。
- 实时渲染仍被保留下来，说明新增运动约束没有把系统拖得过重。

### 对你的价值

MotionGS 很适合放在 4DGS 编辑准备清单里，因为它回答了一个编辑系统也必须回答的问题：
**动态变化究竟由什么显式信号来约束。**

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/NeurIPS_2024/2024_MotionGS_Exploring_Explicit_Motion_Guidance_for_Deformable_3D_Gaussian_Splatting.pdf]]
