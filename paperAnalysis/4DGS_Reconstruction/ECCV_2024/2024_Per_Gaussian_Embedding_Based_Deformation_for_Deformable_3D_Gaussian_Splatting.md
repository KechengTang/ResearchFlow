---
title: "Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting"
venue: ECCV
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - deformable-gaussians
  - temporal-identity-field
  - coarse-fine-deformation
  - local-smoothness
  - dynamic-scene
  - status/analyzed
core_operator: 不再把 deformation 建成纯坐标函数，而是给每个 Gaussian 学习自身 embedding，并和 temporal embeddings 一起预测 coarse-to-fine deformation，从而更精确地建模复杂动态。
primary_logic: |
  输入 canonical 3DGS 与时间索引，先为每个 Gaussian 学习 per-Gaussian latent embedding，
  再结合 coarse/fine temporal embeddings 预测慢运动与快运动的形变，
  同时用局部平滑正则约束邻近高斯 embedding 的一致性，提升动态区域细节与稳定性。
pdf_ref: paperPDFs/4DGS_Reconstruction/ECCV_2024/2024_Per_Gaussian_Embedding_Based_Deformation_for_Deformable_3D_Gaussian_Splatting.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T14:52
updated: 2026-04-18T14:52
---

# Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2404.03613) / [PDF](https://arxiv.org/pdf/2404.03613.pdf) / [Project Page](https://jeongminb.github.io/e-d3dgs/)
> - **Summary**: 这篇论文很值得保留，因为它直接质疑了早期 deformable 3DGS 里最常见的 deformation design。作者认为高斯场景本来就是由多个 Gaussian-centered fields 组成，单纯坐标函数式 deformation 不适合，于是转向 per-Gaussian embeddings。
> - **Key Performance**:
>   - 论文强调其方法更能处理复杂动态场景。
>   - 细节提升主要来自 per-Gaussian identity 表示、coarse/fine deformation 分解和局部平滑正则。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

很多 deformable 3DGS 在复杂动态场景上表现不佳，作者把问题归因到 deformation field 的设计：

- 它们通常把 deformation 写成坐标函数
- 但 3DGS 并不是一个单一连续场，而是多个高斯中心局部场的集合

所以问题不是“不会形变”，而是 **形变作用对象建模错了**。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

作者提出不再让 deformation 完全依赖空间坐标，而是显式给每个 Gaussian 一个自身 embedding。

再配合 temporal embeddings，形变就变成：

- 每个 Gaussian 的身份相关动态
- 与时间状态相关的场景动态

### The "Aha!" Moment

真正的 aha 是：

**如果场景本身是由一组离散 Gaussian 组成，那么 deformation 也应该首先对这组 Gaussian 本体负责，而不是只对空间坐标负责。**

这会带来两个直接好处：

1. 每个 Gaussian 能拥有自己的动态身份
2. 复杂运动不再被迫塞进单一坐标函数

再加上 coarse/fine temporal decomposition，它对慢速和快速动态都更友好。

对你后续做 4DGS 编辑非常有帮助，因为编辑时经常也需要回答：
**到底是对“位置”编辑，还是对“原语身份和状态”编辑。**

### 权衡与局限

- 优势：更细粒度、更贴近高斯表示本体
- 局限：参数与训练复杂度会上升；对 embedding 学习质量敏感

## Part III / Technical Deep Dive

### Pipeline

```text
canonical 3DGS + time index
-> learn per-Gaussian latent embeddings
-> combine with coarse and fine temporal embeddings
-> predict per-Gaussian deformation
-> local smoothness regularization on neighboring embeddings
-> improved dynamic Gaussian reconstruction
```

### 关键技术点

#### 1. Per-Gaussian Identity Embedding

每个 Gaussian 都有自己的 latent embedding，这使 deformation 能捕捉“哪个高斯在怎么动”，而不是只看“这个坐标点会怎么动”。

#### 2. Coarse / Fine Deformation

论文把形变分成 coarse 和 fine 两部分，分别建模慢速和快速动态。

#### 3. Local Smoothness Regularization

邻近高斯 embedding 的相似性被显式约束，从而减少动态区域细节破碎。

### 关键信号

- 论文把对复杂动态的改进归因得很清楚，不只是“指标更高”，而是 deformation design 更合理。
- 这使它非常像一篇方法论意义很强的 backbone paper。

### 对你的价值

如果你后续想做 4DGS 编辑中的 finer-grained control，这篇论文值得优先回看，因为它已经把“高斯身份级别的动态建模”这个问题说得很清楚了。

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/ECCV_2024/2024_Per_Gaussian_Embedding_Based_Deformation_for_Deformable_3D_Gaussian_Splatting.pdf]]
