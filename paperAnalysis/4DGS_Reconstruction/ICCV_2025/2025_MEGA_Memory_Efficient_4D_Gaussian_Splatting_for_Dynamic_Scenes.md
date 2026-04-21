---
title: "MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes"
venue: ICCV
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - compact-model
  - budget-control
  - dynamic-scene-rendering
  - deformation-mlp
  - real-time
  - status/analyzed
core_operator: 通过直接颜色分解、共享轻量颜色预测器和熵约束形变，把 4DGS 中最占内存的颜色与高斯数量同时压缩，显著降低存储和显存成本。
primary_logic: |
  输入动态场景数据后，先把 per-Gaussian 颜色拆成少量 direct color 参数和共享轻量 AC color predictor，
  再用 entropy-constrained Gaussian deformation 扩大单个高斯的作用范围并限制总高斯数，
  最终在保持可比重建质量和渲染速度的同时，得到显著更紧凑的 4DGS 表示。
pdf_ref: paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_MEGA_Memory_Efficient_4D_Gaussian_Splatting_for_Dynamic_Scenes.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T15:06
updated: 2026-04-18T15:06
---

# MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2410.13613) / [PDF](https://arxiv.org/pdf/2410.13613.pdf) / [GitHub](https://github.com/Xinjie-Q/MEGA)
> - **Summary**: MEGA 把 4DGS 的一个现实问题讲得很透: 高质量没问题，但动辄几百万高斯和一大堆颜色参数让它很难真正用起来。它的价值就在于把 4DGS 从“可做”往“可部署”推了一步。
> - **Key Performance**:
>   - 论文报告相对原始 4DGS 在存储上能做到百倍级压缩。
>   - 同时保持了可比的渲染速度和场景表示质量。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

4DGS 的典型问题之一是内存和存储太重：

- 高斯数量巨大
- 每个高斯还带着很长的颜色/SH 属性

这导致动态高斯很容易变成只能离线跑的表示。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

MEGA 把压缩重点放在两个地方：

- 颜色表示
- 高斯数量

这比只做后处理压缩更值得关注，因为它是在表示层直接改设计。

### The "Aha!" Moment

真正的 aha 是：

**4DGS 很重，不只是因为高斯多，而是因为每个高斯都背着太多不必独立存的属性。**

MEGA 的回答是：

- 颜色拆成少量 direct color + 共享 predictor
- 用 entropy-constrained deformation 迫使更少高斯覆盖更多动态变化

也就是说，它不是压缩现有 4DGS，而是在训练时就逼表示更节俭。

### 权衡与局限

- 优势：大幅降存储，仍保质量
- 局限：更强的压缩假设也可能限制极复杂动态细节

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic scene data
-> direct color + shared lightweight color predictor
-> entropy-constrained Gaussian deformation
-> fewer Gaussians with wider action range
-> memory-efficient 4DGS
```

### 关键技术点

#### 1. Streamlined Color Attribute

作者去掉大部分 SH 系数，改成更紧凑的 direct color + shared predictor 方案，这是主要的内存节省来源之一。

#### 2. Entropy-Constrained Deformation

通过 deformation 扩展单个 Gaussian 的作用范围，再用 opacity-based entropy loss 限制高斯数量。

### 对你的价值

如果你后面考虑可编辑 4DGS 的工程落地，这篇论文很值得保留，因为编辑系统最终也会碰到表示体积、显存和速度问题。

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_MEGA_Memory_Efficient_4D_Gaussian_Splatting_for_Dynamic_Scenes.pdf]]
