---
title: "Tune-Your-Style: Intensity-tunable 3D Style Transfer with Gaussian Splatting"
venue: ICCV
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - stylization
  - intensity-control
  - gaussian-neurons
  - style-tuner
  - two-stage-optimization
  - cross-view-style-alignment
  - status/analyzed
core_operator: 用 Gaussian neurons 显式建模 style intensity，并用 learnable style tuner 在 full-style 与 zero-style guidance 之间连续调节 3DGS 风格注入强度。
primary_logic: |
  输入 3DGS 场景与参考风格图，
  先用 Gaussian neurons 与 style tuner 显式参数化风格强度，
  再通过 cross-view style alignment 生成多视角一致的 stylized views，
  最后用两阶段优化在 full-style guidance 与 zero-style guidance 之间调节监督强度，
  从而得到可连续调节 content-style 平衡的 stylized 3DGS。
pdf_ref: paperPDFs/3DGS_Editing/ICCV_2025/2025_Tune_Your_Style_Intensity_tunable_3D_Style_Transfer_with_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-18T13:08
updated: 2026-04-18T13:08
---

# Tune-Your-Style: Intensity-tunable 3D Style Transfer with Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_Tune-Your-Style_Intensity-tunable_3D_Style_Transfer_with_Gaussian_Splatting_ICCV_2025_paper.pdf) · [Project Page](https://zhao-yian.github.io/TuneStyle)
> - **Summary**: Tune-Your-Style 关注的不是“能不能风格化”，而是“风格化强度能不能连续调”。它把 style intensity 从隐式效果变成显式可控变量，因此在 3DGS stylization 里第一次比较系统地解决了 content-style tradeoff 的用户定制问题。
> - **Key Performance**:
>   - 定量上达到 `0.033 / 0.035` 的短程一致性、`0.062 / 0.067` 的长程一致性，并取得 `0.2619` 的 `CLIP S` 与 `0.2881` 的 `CLIP Sdir`。
>   - 用户研究得分为 `3.97 ± 0.13`，优于 StyleGaussian、G-Style 与 InstantStyleGaussian。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

大多数 3DGS 风格化方法默认只输出一个固定强度版本，但现实里用户常常需要不同程度的 style-content 平衡。Tune-Your-Style 解决的是 **3D stylization 的强度可调问题**，而不仅是单点最优质量。

### 核心能力定义

- **输入**: 3DGS 场景与参考风格图。
- **输出**: 能连续调节 style intensity 的 stylized 3DGS。
- **擅长**: 需要保留较多内容细节、但又希望灵活拉高或压低风格强度的场景。
- **边界**: 仍然是 style transfer，不处理对象级语义编辑。

## Part II / High-Dimensional Insight

### 方法设计的核心判断

作者的关键判断是：**style intensity 不应该只是训练后碰运气形成的隐变量，而应被显式建模为可学习、可连续插值的控制量。**

### The "Aha!" Moment

真正的 aha 是 **Gaussian neurons + style tuner**。

- Gaussian neurons 负责把风格强度作为显式可学习变量注入 3DGS；
- style tuner 学一个从强度系数到参数偏移的映射；
- 两阶段优化里，第一阶段先学 full-style 端点，第二阶段再在 full-style 与 zero-style 之间插值学习。

因此，模型学到的不是一个固定答案，而是一条可控的 stylization trajectory。

### 为什么这个设计有效？

- 显式强度建模避免了传统 fixed-output paradigm 的僵硬性。
- cross-view style alignment 先生成多视角一致的 stylized views，减小后续 3D 优化冲突。
- zero-style guidance 让低强度端不会失控，保住原始内容结构。

### 局限

- 仍需两阶段优化，不是完全即时。
- 可控的是 style intensity，不是语义区域级编辑范围。

## Part III / Technical Deep Dive

### Pipeline

```text
3DGS + style image
-> generate cross-view aligned stylized views
-> initialize Gaussian neurons and style tuner
-> stage 1: optimize full-style endpoint
-> stage 2: learn intermediate intensities with full-style + zero-style guidance
-> intensity-tunable stylized 3DGS
```

### 关键机制

#### 1. Gaussian neurons for explicit intensity modeling

作者不是把风格强度塞进一个普通标量超参，而是把它变成作用在 Gaussian 表示上的可学习神经元偏移，这让控制信号真正进入 3DGS 表示层。

#### 2. Learnable style tuner

style tuner 负责把不同强度系数映射到参数偏移，因此强度变化是连续的，不是离散切换几组模板。

#### 3. Tunable stylization guidance

第一阶段学 full-style guidance，第二阶段冻结 full-style 端点，只学习其他强度位置，并用 zero-style guidance 共同约束，避免内容完全塌掉。

### 关键实验信号

- 短程一致性 `0.033 / 0.035`，长程一致性 `0.062 / 0.067`。
- `CLIP S = 0.2619`，`CLIP Sdir = 0.2881`。
- 用户研究 `3.97 ± 0.13`，明显高于 StyleGaussian 与其他基线。

### 实现约束

- 需要先构建 cross-view aligned stylized views 作为 guidance。
- 方法重心是强度控制，不是训练时间最短。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ICCV_2025/2025_Tune_Your_Style_Intensity_tunable_3D_Style_Transfer_with_Gaussian_Splatting.pdf]]
