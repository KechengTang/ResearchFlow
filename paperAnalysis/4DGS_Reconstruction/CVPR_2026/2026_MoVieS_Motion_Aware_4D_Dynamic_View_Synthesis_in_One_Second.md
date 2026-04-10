---
title: "MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second"
venue: CVPR
year: 2026
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - feed-forward
  - monocular-video
  - motion-aware
  - dynamic-scene
  - point-tracking
  - scene-flow
  - status/analyzed
core_operator: 用 dynamic splatter pixels 把静态 pixel-aligned Gaussian 和显式时间形变场解耦，并通过 depth head、splat head、time-conditioned motion head 联合学习 appearance、geometry 与 motion。
primary_logic: |
  输入带位姿视频与查询时间，
  先用共享图像编码器和带相机/时间 token 的特征主干聚合多帧信息，
  再分别由 depth head 预测几何、splat head 预测高斯外观属性、motion head 预测面向任意查询时刻的 3D 位移，
  最终将静态 splatter pixels 与时间形变场组合为 dynamic splatter pixels，
  同时支撑 novel view synthesis、3D point tracking 与零样本运动理解。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2026/2026_MoVieS_Motion_Aware_4D_Dynamic_View_Synthesis_in_One_Second.pdf
category: 4DGS_Reconstruction
created: 2026-04-10T19:43
updated: 2026-04-10T19:43
---

# MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://chenguolin.github.io/projects/MoVieS/) | [arXiv 2507.10065](https://arxiv.org/abs/2507.10065)
> - **Summary**: MoVieS 试图把 4D 场景里的 appearance、geometry、motion 三件事放到一个前向框架里一起学，不再把 tracking、depth、view synthesis 当成彼此割裂的任务，而是用 dynamic splatter pixels 统一建模。
> - **Key Performance**:
>   - 单场景推理约 `0.93s`，在 DyCheck 上达到 `mPSNR 18.46 / mSSIM 58.87 / mLPIPS 0.3094`。
>   - 在 3D point tracking 上，Aria Digital Twin 达到 `EPE 0.2153 / δ0.05 52.05% / δ0.10 71.63%`，显著优于 2D tracking + depth 的基线。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

MoVieS 想解决的是：

**能不能用一个前向模型，同时恢复动态场景的外观、几何和运动，而不是分别训练 depth、tracking 和 novel view synthesis 系统？**

这和很多 4D 方法不一样。很多工作只把 novel view synthesis 当目标，运动只是隐式存在；或者只做 tracking / depth，却不具备可渲染的 4D 表示。MoVieS 的目标更统一。

### 核心能力定义

- **输入**：带位姿单目视频和查询时间
- **输出**：可渲染的动态 3DGS，以及任意像素跨时刻的 3D 运动估计
- **支持任务**：novel view synthesis、depth estimation、3D point tracking、零样本 scene flow、moving object segmentation
- **不主打**：无位姿视频、强物理约束的长期运动模拟

### 真正的挑战来源

- 动态场景中的 motion 与 geometry 容易纠缠，单靠 view synthesis 难以学稳
- 现有 tracking / depth 数据和 4D 渲染数据分布不一致，训练目标异构
- 若运动只靠隐式学习，模型会退化成“近似静态重建”

### 边界条件

- 论文默认输入视频有相机参数
- 训练依赖多种异构数据源和课程训练，工程复杂度不低

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

MoVieS 的核心思想是：

**把一个动态高斯拆成“静态几何/外观底座 + 时间相关形变场”，并且显式监督这个形变场。**

也就是说，作者并不满足于像某些方法那样对每个时间戳单独预测高斯，而是希望：

- 同一个 pixel-aligned Gaussian 在不同时间有可追踪的身份
- novel view synthesis 和 motion estimation 共享同一套表示

### The "Aha!" Moment

真正的 aha 是：

**只有把 motion 作为一等公民显式预测，并让它直接作用于可渲染高斯，4D 表示才可能同时服务于视图合成与运动理解。**

于是作者定义了 `dynamic splatter pixel`：

- 静态部分由 `x, q, s, α, c` 描述
- 动态部分由时间相关的 `Δx(t), Δa(t)` 描述

这样一来，view synthesis 不再只是外观重建任务，而成为 motion 学习的稠密代理任务；反过来，motion supervision 也帮助高斯在时序上保持身份一致。

### 为什么这个设计有效？

- 深度头继承 VGGT 的几何先验，降低了从零学 3D 的负担
- splat head 专注外观，motion head 专注时间形变，避免单头过载
- point-wise L1 和 distribution loss 同时约束运动，前者给绝对位移，后者保局部结构

### 战略权衡

- **优点**：前向、统一、可扩展，还自然支持 zero-shot 下游任务
- **局限**：训练非常依赖大规模异构数据与课程式优化；同时仍要求位姿

## Part III / Technical Deep Dive

### Pipeline

```text
posed monocular video + query time
-> image encoder + Plucker embedding + camera token + timestamp token
-> attention backbone aggregates inter-frame context
-> depth head predicts geometry
-> splat head predicts Gaussian appearance attributes
-> motion head predicts 3D deformation to the query timestamp
-> compose dynamic splatter pixels
-> render novel views / infer 3D tracks / derive scene flow and motion masks
```

### 关键模块

#### 1. Dynamic Splatter Pixels

这是整篇最核心的表示。它把动态场景写成：

- 静态 canonical splatter pixels
- 针对任意查询时刻的 deformation field

因此既保留了 pixel-aligned 3DGS 的可渲染性，又把时序 identity 通过位移场显式写出来。

#### 2. 三头解耦

作者没有像一些 feed-forward 3DGS 方法那样把所有属性塞进一个头里，而是拆成：

- `depth head`
- `splat head`
- `motion head`

其中 motion head 通过 AdaLN 注入查询时间，专门学习“当前 Gaussian 往目标时刻怎么动”。

#### 3. 多源异构训练

MoVieS 同时使用静态场景、动态场景、tracking 数据集训练。关键在于它并不要求每个数据集都具备全部标注，而是让各自提供互补 supervision：

- 深度
- 渲染
- 点跟踪

### 关键实验信号

- 在静态 `RealEstate10K` 上，`Ours` 达到 `PSNR 26.98 / SSIM 81.75 / LPIPS 0.111`，说明动态建模没有明显牺牲静态性能
- 在 DyCheck 上优于 `MoSca / Shape-of-Motion / Splatter-a-Video` 等强基线，且只需约 `0.93s` 每场景
- 在 NVIDIA Dynamic Scene 上，MoVieS 达到 `PSNR 19.16`，虽不一定在所有指标绝对最优，但速度远高于优化式方法
- 在 3D point tracking 上，三套数据都显著优于 `BootsTAPIR + depth`、`CoTracker3 + depth` 和 `SpatialTracker`

### 对当前 idea 的启发

MoVieS 对你当前方向有两个特别重要的启发：

- 它说明 **motion supervision 和 view synthesis 可以互相增益**，这对以后做 4D 编辑时的 temporal adapter 很有参考价值
- `dynamic splatter pixel = static representation + explicit deformation` 这套解耦，也很适合迁移到“canonical edit + temporal propagation”的编辑框架中

如果后面你想要一个比 BTimer 更显式、更适合承载时序编辑的前向骨干，MoVieS 是非常值得继续深挖的一篇。

### 实现约束

- 基于 `VGGT` 初始化并全量微调
- 使用 `32 x H20`，约 `5` 天训练
- 作者明确提到训练不稳定，需要 curriculum 与一系列工程优化

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/CVPR_2026/2026_MoVieS_Motion_Aware_4D_Dynamic_View_Synthesis_in_One_Second.pdf]]
