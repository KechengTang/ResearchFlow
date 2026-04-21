---
title: "4D3R: Motion-Aware Neural Reconstruction and Rendering of Dynamic Scenes from Monocular Videos"
venue: NeurIPS
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - pose-free
  - motion-aware
  - bundle-adjustment
  - control-points
  - linear-blend-skinning
  - status/analyzed
core_operator: 先用 3D foundation models 做 pose/geometry 初始化，再通过 motion-aware bundle adjustment 和 motion-aware Gaussian Splatting 细化动态重建，把 pose-free monocular dynamic rendering 做成显式高斯表示。
primary_logic: |
  输入未知相机位姿的单目动态视频，先由 3D foundation model 估计初始位姿与几何，
  再用 motion-aware bundle adjustment 结合动态分割细化位姿，
  最后用带 control points、deformation MLP 和 linear blend skinning 的 MA-GS 表示建模动态对象，实现高质量 pose-free 4D 重建。
pdf_ref: paperPDFs/4DGS_Reconstruction/NeurIPS_2025/2025_4D3R_Motion_Aware_Neural_Reconstruction_and_Rendering_of_Dynamic_Scenes_from_Monocular_Videos.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T15:06
updated: 2026-04-18T15:06
---

# 4D3R: Motion-Aware Neural Reconstruction and Rendering of Dynamic Scenes from Monocular Videos

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2511.05229) / [PDF](https://arxiv.org/pdf/2511.05229.pdf)
> - **Summary**: 4D3R 值得补进来，因为它把 pose-free dynamic rendering 和 motion-aware Gaussian Splatting 接了起来。相比只在已知位姿条件下做动态高斯，它更接近真实单目视频管线。
> - **Key Performance**:
>   - 论文报告相比 SOTA 可有约 1.8 dB PSNR 提升。
>   - 同时计算成本可降到先前动态表示的约五分之一。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

动态单目视频的新视角合成如果没有相机位姿，难度会显著上升。

问题主要有两层：

- pose-free 场景下位姿和几何本身就不稳
- 动态对象还会进一步干扰重建和位姿估计

4D3R 的目标就是把这两层联动起来解决。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

4D3R 的结构是两阶段的：

1. foundation model 初始化位姿和几何
2. motion-aware refinement + MA-GS 细化动态重建

### The "Aha!" Moment

真正的 aha 是：

**在动态单目场景里，位姿估计和动态重建不该分开做，因为动态对象本身就是位姿优化的噪声来源。**

所以 4D3R 一方面用 motion-aware BA 改位姿，另一方面用 motion-aware GS 改表示，两者是联动的。

这对你以后做真实视频上的 4DGS 编辑也很相关，因为很多编辑流水线默认输入位姿可靠，但现实里这一步本身就可能不稳。

### 权衡与局限

- 优势：更贴近真实单目视频，静动问题联动解决
- 局限：系统较复杂，依赖 foundation model 初始化和动态分割质量

## Part III / Technical Deep Dive

### Pipeline

```text
pose-free monocular dynamic video
-> foundation-model-based initial pose and geometry
-> motion-aware bundle adjustment with dynamic segmentation
-> motion-aware Gaussian Splatting (control points + deformation MLP + LBS)
-> dynamic scene reconstruction and rendering
```

### 关键技术点

#### 1. Motion-Aware Bundle Adjustment

作者把 transformer prior 与动态分割结合进 BA，使动态对象不再单纯作为外点污染位姿估计。

#### 2. Motion-Aware Gaussian Splatting

MA-GS 用 control points、deformation field 和 linear blend skinning 描述动态运动，在保证质量的同时降低成本。

### 对你的价值

4D3R 在你的库里适合作为真实输入条件下的动态高斯重建参考，因为它把：

- pose-free
- motion-aware refinement
- Gaussian-based dynamic representation

这三件事真正接了起来。

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/NeurIPS_2025/2025_4D3R_Motion_Aware_Neural_Reconstruction_and_Rendering_of_Dynamic_Scenes_from_Monocular_Videos.pdf]]
