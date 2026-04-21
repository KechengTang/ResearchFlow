---
title: "Segment Any 3D Gaussians"
venue: AAAI
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - promptable-segmentation
  - gaussian-affinity-feature
  - scale-gate
  - sam-distillation
  - multi-granularity-segmentation
  - open-vocabulary-segmentation
  - status/analyzed
core_operator: 给每个 Gaussian 附加显式 affinity feature，并用 lightweight scale gate 在不同 3D 物理尺度下调制这些特征，再通过 scale-aware contrastive learning 把 SAM 的 promptable segmentation 能力蒸馏进 3DGS。
primary_logic: |
  输入预训练 3DGS 与由 SAM 生成的多视图 masks；
  为每个 Gaussian 学一个 affinity feature，并用 scale gate 处理 multi-granularity ambiguity；
  通过 scale-aware contrastive learning、local feature smoothing 与 feature norm regularization 训练这些特征；
  推理时输入 2D 点提示与尺度，即可毫秒级返回对应 3D 分割结果，也可进一步做 scene decomposition 与 vote-based open-vocabulary segmentation。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/AAAI_2025/2025_Segment_Any_3D_Gaussians.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:06
updated: 2026-04-20T19:06
---

# Segment Any 3D Gaussians

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://jumpat.github.io/SAGA/) / [arXiv](https://arxiv.org/abs/2312.00860)
> - **Summary**: SAGA 的贡献不是简单做一个 3D 版 SAM，而是证明了 promptable segmentation 能力可以直接蒸馏进 3DGS 的显式 Gaussian 属性里，并通过 scale gate 保住多粒度分割能力与实时性。
> - **Key Performance**:
>   - 在 NVOS 上达到 `92.6 mIoU / 98.6 mAcc`，超过 SA3D-GS 的 `92.2 / 98.5`。
>   - 在 3D-OVS 上达到 `96.0 mIoU`，并在推理时只需 `2~5 ms`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

SAGA 解决的是：**如何让 3DGS 具备像 SAM 一样的 3D promptable segmentation 能力，而且还能保留多粒度与实时交互。**

作者认为，3D promptable segmentation 有两个核心难点：

- **怎么把 SAM 的分割能力装进 Gaussian 本身**，而不是额外再挂一个笨重 segmentation module。
- **怎么保住多粒度分割能力**。同一个 Gaussian 在不同物理尺度下，可能属于不同 segmentation target。

因此 SAGA 的能力定义是：

- **能做**：基于 2D 点提示的 3D promptable segmentation，多粒度 segmentation，scene decomposition。
- **可扩展**：补充的 vote-based open-vocabulary segmentation。
- **不主打**：完全脱离 SAM 的泛化分割；动态场景时序语义。

### 输入 / 输出接口

- **输入**：预训练 3DGS、SAM 产生的多视图 masks，以及带尺度的 2D prompt points。
- **输出**：对应目标的 3D Gaussian 子集，或整场景的聚类式分解结果。

### 边界条件

- 方法依赖 SAM 自动提取到目标；若目标从未在 SAM mask 中稳定出现，SAGA 就很难分出来。
- 小目标最容易受这个限制影响。
- 论文的开放词汇分割主要放在补充材料里，核心主线仍是 promptable segmentation。

## Part II / High-Dimensional Insight

### 方法真正新的地方

SAGA 把“3D 分割”重写成了“Gaussian 属性学习”问题：

- 不再单独训练一个隐式 feature field。
- 而是给每个 Gaussian 显式挂上 **affinity feature**，让 Gaussian 自己带有“可被分割”的属性。

接着，为了处理 multi-granularity ambiguity，作者再加一个极轻量的 **scale gate**：

- 小尺度时，特征强调细粒度局部结构。
- 大尺度时，特征关闭一部分通道，转向更粗粒度对象感知。

### The "Aha!" Moment

真正的 aha 是：

**3D promptable segmentation 的关键不只是“像不像目标”，而是“在什么物理尺度下，这个 Gaussian 该算作这个目标的一部分”。**

SAGA 不是像 GARField 那样在查询时反复扫不同尺度，而是把尺度条件直接内化进 Gaussian affinity feature 的门控里。

因此它把原来的推理难题变成了训练期蒸馏：

`SAM 的多粒度 mask 关系 -> scale-aware contrastive signal -> Gaussian affinity feature + scale gate -> 推理时一次查询即可得到对应尺度下的 3D 分割`

### 为什么这个设计有效

- **显式 affinity feature** 让 3DGS 自身成为可查询分割载体，不需要额外 3D segmentation field。
- **scale gate** 把多尺度歧义从“测试时重复搜索”变成“训练时可学习调制”，因此推理速度很快。
- **local feature smoothing** 与 **feature norm regularization** 分别处理 3D 空间离群点和 2D/3D feature 对齐问题。

### 策略权衡

- **优点**：速度快，结构简单，多粒度 segmentation 很自然。
- **代价**：本质上仍是从 SAM masks 蒸馏而来，所以对未被 SAM 覆盖的目标泛化较弱。

## Part III / Technical Deep Dive

### Pipeline

```text
pretrained 3DGS + multi-view SAM masks
-> attach per-Gaussian affinity feature
-> scale gate modulates feature channels by physical scale
-> scale-aware contrastive learning distills mask correspondence
-> local feature smoothing + feature norm regularization
-> promptable 3D segmentation / scene decomposition
-> optional vote-based open-vocabulary segmentation
```

### 关键技术信号

- **Gaussian affinity features**  
  作者把分割能力直接写进每个 Gaussian，而不是把 3DGS 当成查询外壳。

- **scale-aware contrastive learning**  
  监督不是简单的同类拉近、异类推远，而是显式地带尺度条件。这样同一物体在不同尺度上可呈现 part / whole 的不同身份。

- **feature smoothing 与 norm regularization**  
  前者负责清掉 3D 空间中的离群假阳性，后者则缓解“2D 渲染后能分，但 3D Gaussian 本身并不对齐”的问题。

### 少量关键数字

- NVOS 上：`92.6 mIoU / 98.6 mAcc`。
- SPIn-NeRF 上：`93.4 mIoU / 99.2 mAcc`。
- 3D-OVS 上：平均 `96.0 mIoU`。
- 推理时间仅 `2~5 ms`，相比 GARField 的 `30~70 ms` 更快。

### 实现约束

- affinity feature 维度设为 `32`，KNN smoothing 的 `K = 16`。
- 训练 `10,000` iter；每次随机采样 `8` 个尺度与 `1,000` 个像素做 contrastive 学习。
- SAM 使用 `ViT-H`；补充的开放词汇分割使用 `OpenCLIP ViT-B/16`；全部实验在单张 RTX 3090 上完成。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/AAAI_2025/2025_Segment_Any_3D_Gaussians.pdf]]
