---
title: "PanoGS: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding"
venue: CVPR
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - panoptic-segmentation
  - open-vocabulary-grounding
  - scene-understanding
  - tri-plane
  - gaussian-clustering
  - status/analyzed
core_operator: 用金字塔 tri-plane 学连续 3D 语言特征场，再把语言特征与几何一起写进图聚类流程，把海量 Gaussian 先压成 super-primitives，再做实例一致的开放词汇 panoptic 分割。
primary_logic: |
  输入多视图 RGB-D 与重建好的 3DGS 场景，先用 pyramid tri-plane + 3D decoder 拟合连续语言特征场，再通过 language-guided graph cuts 合并 Gaussians 为 super-primitives，并结合 SAM 掩码估计边亲和力做渐进式图聚类，最终输出开放词汇语义与实例一致的 3D panoptic 结果。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_PanoGS_Gaussian_based_Panoptic_Segmentation_for_3D_Open_Vocabulary_Scene_Understanding.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:23
updated: 2026-04-20T19:23
---

# PanoGS: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhai_PanoGS_Gaussian-based_Panoptic_Segmentation_for_3D_Open_Vocabulary_Scene_Understanding_CVPR_2025_paper.pdf) / [Project Page](https://zju3dv.github.io/panogs/)
> - **Summary**: PanoGS 的重点不是再做一个 3D 语言热图，而是把开放词汇语义和实例分组同时纳入 3DGS 图结构里，让系统能输出真正可查询、可区分实例的 3D panoptic 结果。
> - **Key Performance**:
>   - 在 ScanNetV2 上，纯开放词汇语义达到 `50.72 mIoU / 70.20 mAcc`，明显高于 OpenScene(Ens.) 的 `47.63 / 69.74`。
>   - 在 Replica 上，达到 `54.98 mIoU / 67.35 mAcc`；配合监督 mask 做 panoptic 时，thing/stuff PRQ 达到 `49.26 / 48.24`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题?
PanoGS 要补的空白很明确: 现有 3DGS 语义方法大多只能回答“哪里像这个词”，却很难回答“这是哪一个实例”。  
这会导致两类问题同时出现:

- 开放词汇查询通常只返回一张相似度热图，多个同类物体容易粘在一起。
- 离散地给每个 Gaussian 存特征，会被 alpha blending 和局部噪声破坏，语义不够平滑，也不利于大场景扩展。

所以这篇论文真正追求的能力边界是:

- **能做**: 3D 开放词汇语义分割、实例一致的 3D panoptic 分割、文本查询对应的实例级结果。
- **不主打**: 动态场景、时间一致语义、交互式编辑。

### 输入 / 输出接口

- **输入**: 多视图 RGB-D、重建好的 3DGS 场景、2D 视觉语言特征、SAM 生成的实例 mask。
- **输出**: 每个查询词对应的 3D 语义结果，以及按实例分开的 3D panoptic 分割。

### 边界条件

- 方法主要面向大规模室内静态场景，实验集中在 ScanNetV2 和 Replica。
- 实例一致性仍然依赖 2D mask 质量；如果 2D 实例边界本身混乱，图聚类上限也会受限。

## Part II / High-Dimensional Insight

### 方法真正新的地方

PanoGS 的核心转向有两步:

1. 不再把语言特征简单当作“每个 Gaussian 一个离散向量”，而是改成 **连续可回归的 tri-plane 特征场**。
2. 不再直接在数百万个 Gaussian 上做实例分组，而是先用 **language-guided graph cuts** 把它们压成 super-primitives，再在更稳的图上做聚类。

这使它把“开放词汇语义”与“实例分割”从串联关系改成了统一图推理关系。

### The "Aha!" Moment

真正的 aha 在于:

**3D panoptic open-vocabulary 的难点不是先有语义、后补实例，而是要先把 3D 场景变成一个既保留语义平滑性、又能承载实例边界的图结构。**

PanoGS 的做法是:

- 用 pyramid tri-plane 把语言特征从离散 Gaussian 参数，改写成连续可插值的 3D 特征场。
- 用 language-guided graph cuts 先合并局部一致的 Gaussian，降低图规模并提升语义稳定性。
- 再结合 SAM 提供的 2D 线索估计 super-primitive 边亲和力，做渐进式 graph clustering，得到实例级结果。

这条因果链比较完整:

`连续语言场更平滑 -> super-primitives 更稳定 -> 图聚类不再被海量 noisy Gaussian 拖垮 -> 开放词汇查询能落到实例级 panoptic 结果`

### 为什么这个设计有效?

- tri-plane 解决的是 **高维语言特征难以显式存储** 的问题。
- super-primitives 解决的是 **图太大、边太噪** 的问题。
- SAM 引导的边亲和力解决的是 **实例边界仅靠语言特征不够稳** 的问题。

### 策略权衡

- **优点**: 同时兼顾开放词汇语义与实例一致性；适合大室内场景；图结构清晰，后续也便于扩展到 query-driven 场景理解。
- **代价**: 流程比纯语言热图方法更重；需要 RGB-D、2D 视觉语言特征和实例 mask 多种先验一起配合。

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view RGB-D + 3DGS
-> extract 2D vision-language features
-> pyramid tri-plane + 3D decoder regress continuous 3D language field
-> language-guided graph cuts merge Gaussians into super-primitives
-> compute SAM-guided edge affinity between super-primitives
-> progressive graph clustering
-> open-vocabulary semantic + instance-level 3D panoptic output
```

### 关键信号

- 论文明确指出，旧方法的问题不是“没有语言特征”，而是只能输出文本相似度热图，无法稳定区分同类不同实例。
- 它把 panoptic 分割重写成图聚类问题，而不是继续在每个 Gaussian 上逐点打标签。
- `Ours + Sup. mask` 在 panoptic 指标上提升很大，说明实例边界这一步确实主要依赖图结构和 mask 引导，而不是单靠语言场本身。

### 少量关键数字

- ScanNetV2: `50.72 mIoU / 70.20 mAcc`，高于 OpenScene(Ens.) 的 `47.63 / 69.74`。
- ScanNetV2 panoptic: `49.26 PRQ(T) / 48.24 PRQ(S)`。
- Replica: `54.98 mIoU / 67.35 mAcc`，panoptic `43.04 PRQ(T) / 30.60 PRQ(S)`。

### 实现约束

- 2D 视觉语言特征使用 LSeg；文本特征使用 OpenCLIP。
- 图规模控制依赖 super-primitives，这一步是能否在室内大场景上跑通的关键工程设计。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_PanoGS_Gaussian_based_Panoptic_Segmentation_for_3D_Open_Vocabulary_Scene_Understanding.pdf]]
