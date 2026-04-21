---
title: "Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh"
venue: CVPR
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - scene-manipulation
  - mesh-guided-editing
  - triangle-binding
  - local-triangle-space
  - large-deformation
  - soft-body-simulation
  - status/analyzed
core_operator: 用三角网格作为可编辑代理，在局部三角形坐标系里绑定并优化 Gaussian 属性，再把网格变形自适应传播回 3DGS。
primary_logic: |
  输入预训练 3DGS 与从场景中提取的三角网格，
  先把每个 Gaussian 绑定到局部三角形空间并在该局部坐标系中优化位置、旋转与尺度，
  再在用户操控网格发生大形变、局部编辑或软体模拟时，通过自适应公式把局部属性稳定映射回全局 3DGS，
  从而在保持高保真渲染的同时获得可控 3DGS 操作能力。
pdf_ref: paperPDFs/3DGS_Editing/CVPR_2025/2025_Mani_GS_Gaussian_Splatting_Manipulation_with_Triangular_Mesh.pdf
category: 3DGS_Editing
created: 2026-04-18T13:05
updated: 2026-04-18T13:05
---

# Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Mani-GS_Gaussian_Splatting_Manipulation_with_Triangular_Mesh_CVPR_2025_paper.pdf)
> - **Summary**: Mani-GS 把“如何编辑 3DGS”转成“如何编辑一个可操作的三角网格代理”，关键不是简单绑死 Gaussian 到网格表面，而是让 Gaussian 在局部三角形坐标系里保持可自适应自由度，因此在网格不准、形变很大时仍能维持高质量渲染。
> - **Key Performance**:
>   - 在 NeRF Synthetic 上，相比 NeRF-Editing 与 SuGaR，多个场景的 PSNR/SSIM/LPIPS 都更优；例如 `Mic` 场景达到 `37.46 PSNR / 0.992 SSIM / 0.007 LPIPS`。
>   - 不只支持拉伸、弯折等大形变，还支持局部操控和 soft-body simulation，说明它的目标不只是“看起来能改”，而是形成统一的 3DGS manipulation 接口。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

Mani-GS 面向的是 **3DGS 的几何可操控性**。3DGS 渲染很强，但原生 Gaussian 点云缺少像网格那样直接、稳定、可交互的编辑接口。此前方法要么落回 NeRF 体系，训练和推理都重；要么像 SuGaR 那样依赖高质量网格，一旦网格不准，渲染和编辑都会受损。

### 核心能力定义

- **输入**: 预训练 3DGS 与从场景中提取的三角网格。
- **输出**: 可被大形变、局部形变、软体模拟驱动的 3DGS。
- **擅长**: 几何级 manipulation，而不是文本驱动外观编辑。
- **边界**: 仍然依赖一个可用的网格代理；如果目标是语义替换或扩散式外观改写，这篇不是直接答案。

## Part II / High-Dimensional Insight

### 方法设计的核心判断

这篇工作的关键判断是：**3DGS 需要的不是再造一套专门的 deform 算法，而是借用成熟网格编辑接口作为“操纵柄”，再设计一种不被网格误差绑死的 Gaussian 绑定方式。**

### The "Aha!" Moment

真正的 aha 在于 **local triangle space**。

作者不是把 Gaussian 强行压到三角形表面，也不是给一个固定 offset 让它跟着网格走，而是把每个 Gaussian 的位置、旋转、尺度都表示在三角形的局部坐标系里。这样：

- 网格编辑时，局部坐标中的属性保持不变；
- 全局空间中的 Gaussian 属性根据三角形的新形状自适应更新；
- 即使网格本身不够准，Gaussian 仍有足够自由度在三角形附近做补偿。

这让“网格可编辑性”和“3DGS 渲染保真度”第一次被比较自然地兼容起来。

### 为什么这个设计有效？

- 网格负责提供稳定、可交互的几何控制接口。
- 局部三角形空间负责把这种控制变成可泛化到变形后的 Gaussian 更新规则。
- 允许 Gaussian 不完全贴死网格，使得系统对 mesh extraction 误差更鲁棒。

### 代价与局限

- 仍然需要先提网格，再做两阶段训练。
- 主要解决 manipulation，不直接解决文本、语义或材质编辑。

## Part III / Technical Deep Dive

### Pipeline

```text
pretrained 3DGS
-> extract triangle mesh
-> bind Gaussians to local triangle spaces
-> optimize Gaussian attributes in local coordinates
-> user edits / deforms mesh
-> self-adaptive update maps local attributes to deformed global space
-> manipulated 3DGS rendering
```

### 关键机制

#### 1. Triangle shape-aware Gaussian binding

绑定时显式使用三角形形状，而不是只用一个最近点或固定偏移。这样 Gaussian 的姿态变化能感知三角形变形，而不是只做刚体跟随。

#### 2. Self-adaptive propagation under deformation

局部属性在编辑后保持稳定，全局属性按新三角形几何重新计算。这个设计让它能统一处理 stretching、bending 甚至 soft-body 场景，而不是每种 manipulation 单独设计新算法。

#### 3. Tolerance to inaccurate meshes

这篇比 SuGaR 更重要的一点，是它默认网格会有误差。由于 Gaussian 可以在三角形附近自由补偿，所以不会因为网格局部缺损就直接把渲染质量拉坏。

### 关键实验信号

- NeRF Synthetic 多个场景上明显优于 NeRF-Editing 与 SuGaR。
- 相同底层网格下，在 DTU 上也优于 SuGaR 与 NeuMesh，说明优势并不只来自“网格更好”。
- 论文专门展示 inaccurate mesh、large deformation、local manipulation、soft body 四种情形，说明这不是只对轻微编辑有效的技巧。

### 实现约束

- 两阶段训练：先重建并提网格，再做绑定与优化。
- 论文使用单张 `NVIDIA A100`，第一阶段约 `30K` iterations，第二阶段约 `20K` iterations。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/CVPR_2025/2025_Mani_GS_Gaussian_Splatting_Manipulation_with_Triangular_Mesh.pdf]]
