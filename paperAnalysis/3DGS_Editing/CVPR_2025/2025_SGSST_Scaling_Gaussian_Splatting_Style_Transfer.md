---
title: "SGSST: Scaling Gaussian Splatting Style Transfer"
venue: CVPR
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - stylization
  - optimization-based
  - ultra-high-resolution
  - multi-scale-style-loss
  - global-style-statistics
  - appearance-editing
  - status/analyzed
core_operator: 用一个无额外超参的 SOS 多尺度风格损失，直接在预训练 3DGS 上优化全局神经统计量，把风格迁移扩展到超高分辨率场景。
primary_logic: |
  输入预训练 3DGS 场景与参考风格图，
  不再依赖复杂的多项辅助损失，而是仅通过 SOS 多尺度损失同时匹配多个尺度上的全局风格统计，
  主要优化 3DGS 的颜色相关参数以保持几何稳定，
  从而把 3DGS 风格迁移扩展到超高分辨率并提升风格保真度。
pdf_ref: paperPDFs/3DGS_Editing/CVPR_2025/2025_SGSST_Scaling_Gaussian_Splatting_Style_Transfer.pdf
category: 3DGS_Editing
created: 2026-04-18T13:06
updated: 2026-04-18T13:06
---

# SGSST: Scaling Gaussian Splatting Style Transfer

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Galerne_SGSST_Scaling_Gaussian_Splatting_Style_Transfer_CVPR_2025_paper.pdf)
> - **Summary**: SGSST 的核心贡献不是“又一个 3DGS 风格迁移器”，而是把 3DGS 风格迁移真正推进到 ultra-high-resolution。它用一个单一的 SOS 多尺度风格损失替代复杂损失拼装，在保住几何稳定的同时大幅提升高分辨率风格保真度。
> - **Key Performance**:
>   - 支持 `5187 × 3361` 级别的 UHR 风格迁移，分辨率能力相对既有 3DGS stylization 方法提升约 `4×`。
>   - 在 40 个实验、680 张投票的感知研究中，`66.3%` 的票数选择 SGSST，为 ARF 的 `31.6%` 和 StyleGaussian 的 `2.1%` 之上。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

很多 3DGS 风格化方法在中低分辨率还可以，但一旦走向 **AR/VR 所需的超高分辨率场景**，显存、训练成本和风格细节都会迅速崩掉。SGSST 关心的是：**怎样把 3DGS style transfer 扩到 UHR，同时保持全局风格质量，而不是只在低分辨率下看着像。**

### 核心能力定义

- **输入**: 预训练 3DGS 场景与单张风格参考图。
- **输出**: 超高分辨率、外观风格化后的 3DGS。
- **擅长**: appearance stylization，不涉及几何或语义编辑。
- **边界**: 仍是 optimization-based 方法，运行时间明显长于 feed-forward 方法。

## Part II / High-Dimensional Insight

### 方法设计的核心判断

SGSST 的核心判断是：**高分辨率 3DGS 风格迁移的瓶颈不只是算力，而是损失函数是否能在多尺度上稳定传递全局风格统计。**

### The "Aha!" Moment

真正的 aha 是 **SOS: Simultaneously Optimized Scales**。

作者没有继续堆很多辅助项，而是把多尺度风格统计放进一个统一、无额外超参的单一目标里。这样做的结果是：

- 高分辨率下仍能稳定传递 style image 的全局统计；
- 不需要复杂 loss cocktail 才能避免 artefact；
- 只优化颜色相关参数，就能尽量保住原始 3DGS 几何。

### 为什么这个设计有效？

- 多尺度统计让系统不只学到局部笔触，还能保住大范围风格分布。
- 只动颜色而少动其他 3DGS 参数，几何漂移更小。
- 对比 StyleGaussian 这类更快的方法，SGSST 把重点放在“质量可扩展到 UHR”，而不是单纯快。

### 局限

- 训练时间较长，约为原始 3DGS 训练时间的 `2-8×`。
- 属于高质量离线风格迁移，不适合追求秒级交互的应用。

## Part III / Technical Deep Dive

### Pipeline

```text
pretrained 3DGS + style image
-> render multi-scale views
-> compute SOS style loss over global neural statistics
-> optimize mainly color-related Gaussian parameters
-> obtain UHR stylized 3DGS
```

### 关键机制

#### 1. SOS multi-scale style loss

SOS 把多尺度风格统计联合优化，目的是让模型在 UHR 场景下既抓住全局风格，又不被某一个尺度主导。

#### 2. Parameter-efficient optimization

作者强调主要优化 3DGS 的颜色部分，而不是全面调整所有 Gaussian 参数，这降低了几何失真与训练不稳定性。

#### 3. Quality-over-speed design

这篇工作明确站在高视觉质量一侧，服务对象更接近 AR/VR 内容制作，而不是实时风格切换。

### 关键实验信号

- 感知研究中 `66.3%` 投票选 SGSST，明显高于 ARF 与 StyleGaussian。
- 论文强调其是首个直接把 3DGS style transfer 扩到 UHR 的方法，并在 HR 场景下保持优于基线的视觉质量。

### 实现约束

- 优化式方法，整体耗时较长。
- 更适合作为高质量 stylization baseline，而不是在线编辑模块。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/CVPR_2025/2025_SGSST_Scaling_Gaussian_Splatting_Style_Transfer.pdf]]
