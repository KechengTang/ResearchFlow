---
title: "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting"
venue: ICLR
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - dynamic-scene-rendering
  - explicit-representation
  - real-time
  - dynamic-scene
  - status/analyzed
core_operator: 直接用可在时空中旋转的 4D Gaussian primitives 与 4D spherindrical harmonics 建模动态场景，把空间和时间统一成显式 4D 原语而不是额外 deformation 附件。
primary_logic: |
  输入动态场景的单目或多视图时序图像，直接优化一组 4D Gaussian primitives，
  用显式几何和 4D spherindrical harmonics 建模时变外观，
  再通过专门的 4D splatting renderer 在任意时间与视角生成动态场景的新视图。
pdf_ref: paperPDFs/4DGS_Reconstruction/ICLR_2024/2024_Real_time_Photorealistic_Dynamic_Scene_Representation_and_Rendering_with_4D_Gaussian_Splatting.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T14:30
updated: 2026-04-18T14:30
---

# Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [OpenReview](https://openreview.net/forum?id=WhgB5sispV) / [arXiv](https://arxiv.org/abs/2310.10642) / [PDF](https://arxiv.org/pdf/2310.10642.pdf) / [Project Page](https://fudan-zvg.github.io/4d-gaussian-splatting)
> - **Summary**: 这篇 ICLR 2024 论文是另一条很重要的 4DGS 主线。它不把动态看成“静态高斯 + 形变附件”，而是直接把场景写成真正的 4D Gaussian primitives，因此对理解后续 4DGS 编辑的表示空间非常关键。
> - **Key Performance**:
>   - 论文在多种单目和多视图动态场景基准上都报告了优于先前方法的画质与效率。
>   - 文中展示的渲染速度可到百帧级，明显快于大多数隐式动态 NeRF 系方法。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

它解决的是动态场景表示层面的一个根问题：

**如果时间只是额外挂上的条件变量，模型往往很难既建好复杂运动，又保住实时渲染。**

传统动态 NeRF 路线通常会遇到两类困难：

- 直接拟合 6D plenoptic function，结构过于复杂
- 明确给每个元素建 deformation，在复杂动态下成本很高

这篇论文的选择是：
**不要把时间视为后处理变量，而是从表示层直接把场景写成 4D。**

## Part II / High-Dimensional Insight

### 方法真正新在哪里

这篇工作的核心不是“把 3DGS 改到能处理视频”，而是从表示假设开始改变：

- primitive 本身是 4D 的
- 颜色不是静态 SH，而是时间演化的 4D appearance basis
- 渲染器也围绕 4D primitive 重新设计

### The "Aha!" Moment

真正的 aha 在于：

**动态场景不一定要表示成“静态几何 + 时间变形”，也可以直接表示成时空一体的 4D 原语集合。**

这带来两个结果：

1. 模型不必先找 canonical 3D，再学如何把它扭回每个时间
2. 对任意时刻的渲染都直接落在统一 4D 表示上

对你来说，这篇论文的重要性在于它定义了另一种 4DGS 编辑前提：

- 如果你沿 canonical/deformation 路线做编辑，Instruct-4DGS 一类论文很关键
- 如果你沿 4D primitive 路线做编辑，这篇论文就是必须知道的表示基线

### 权衡与局限

- 优势：表示统一、时空建模直接、渲染高效
- 局限：语义接口不强，缺少天然的静动分离与编辑作用位点

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic scene images
-> optimize 4D Gaussian primitives
-> model time-varying appearance with 4D spherindrical harmonics
-> specialized 4D splatting renderer
-> photorealistic dynamic novel view synthesis
```

### 关键技术点

#### 1. 4D Gaussian Primitive

论文直接让高斯原语在时空中具有显式形状与旋转，而不是只在 3D 空间里存在。

#### 2. 4D Appearance Modeling

作者提出 4D spherindrical harmonics 来处理 view-dependent 且随时间变化的外观，这比静态 SH 更适合动态场景。

#### 3. Dedicated 4D Renderer

因为表示发生了变化，渲染器也不再是普通 3DGS 的轻微修改，而是围绕 4D primitive 定制。

### 关键信号

- 论文在 real / synthetic、monocular / multi-view 多个设置下都给出了优于前人的画质与速度。
- 文中百帧级速度非常重要，它说明 4D 显式原语并不一定意味着动态建模一定会变慢。

### 对你的价值

这篇论文在你的 KB 里应该被视为 **4DGS 表示分叉点**：

- 一支走向 canonical-space deformation
- 一支走向 true 4D primitives

如果你后面做 4DGS 编辑，这种分叉会直接影响你选什么编辑接口、什么时间一致性机制、什么语义作用位点。

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/ICLR_2024/2024_Real_time_Photorealistic_Dynamic_Scene_Representation_and_Rendering_with_4D_Gaussian_Splatting.pdf]]
