---
title: "SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining"
venue: ICCV
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - scene-understanding
  - self-supervision
  - vision-language-pretraining
  - open-world-segmentation
  - status/analyzed
core_operator: 以 SceneSplat-7K 为数据底座，在原生 3DGS 上做 vision-language pretraining 与 GaussianSSL 自监督预训练，让模型能单次前向为数百万个 Gaussians 预测开放词汇语言特征。
primary_logic: |
  输入大规模 3DGS 室内场景与配套 2D 语义线索，先在 SceneSplat-7K 上做 vision-language pretraining 学习 Gaussian 级语言特征，再用 Gaussian masking 与 self-distillation 的 GaussianSSL 扩展到无标注场景，最终得到可前向推理的 3DGS 场景理解模型，用于零样本与监督语义分割。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_SceneSplat_Gaussian_Splatting_based_Scene_Understanding_with_Vision_Language_Pretraining.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:23
updated: 2026-04-20T19:23
---

# SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICCV 2025 Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_SceneSplat_Gaussian_Splatting-based_Scene_Understanding_with_Vision-Language_Pretraining_ICCV_2025_paper.pdf)
> - **Summary**: SceneSplat 真正想做的不是单篇论文级别的 3DGS 语义蒸馏，而是给 3DGS 搭一个“可预训练、可前向推理”的场景理解底座，并顺手补齐了此前缺失的大规模 3DGS 室内数据集。
> - **Key Performance**:
>   - 单源 ScanNet 训练时，在 ScanNet200 上达到 `18.9 f-mIoU / 31.7 f-mAcc`，比 Mosaic3D 的 `13.0 / 24.5` 提升明显。
>   - 三源训练时，在 ScanNet++ 上达到 `28.4 f-mIoU / 50.0 f-mAcc`，比并发工作高 `10.4% f-mIoU`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题?
SceneSplat 针对的是一个更高层的问题:

**为什么 3DGS 已经成为主流 3D 表示，但还没有一个像 2D foundation model 那样、能够在原生 3DGS 上做场景理解预训练的模型?**

作者认为缺口有两层:

- 没有足够大、足够统一的 3DGS 语义预训练数据。
- 现有 3D 开放词汇方法多数仍依赖 2D 或文本模态在训练或推理阶段持续介入，不是真正的 native 3DGS 理解模型。

因此论文的能力定义是:

- **能做**: 在原生 3DGS 上进行零样本与监督语义分割，单次前向为数百万个 Gaussians 预测语言特征。
- **额外价值**: 提供 SceneSplat-7K 数据底座，支持后续 3DGS 预训练研究。
- **不主打**: 动态场景、编辑系统、单场景精调。

### 输入 / 输出接口

- **输入**: SceneSplat-7K 中的大量 3DGS 场景，以及对应的 2D 语义线索与语言标签。
- **输出**: feed-forward 的 Gaussian 语言特征，以及零样本/监督语义分割预测。

### 边界条件

- 数据和任务都聚焦室内场景。
- 预训练流程很重，价值更多体现在“平台化能力”，而非单场景轻量部署。

## Part II / High-Dimensional Insight

### 方法真正新的地方

SceneSplat 的真正创新不止是一个模型，而是一套配套基础设施:

1. **SceneSplat-7K**: 首个大规模 3DGS 室内场景理解数据集。
2. **Vision-Language Pretraining**: 在 Gaussian 级别学习语言特征，而不是只做 2D 特征蒸馏。
3. **GaussianSSL**: 用 Gaussian masking + self-distillation 把无标注 3DGS 场景也纳入预训练。

### The "Aha!" Moment

真正的 aha 在于:

**如果 3DGS 想拥有类似 2D foundation model 的泛化能力，就不能一直停留在 per-scene distillation，而要转向 dataset-scale 的 feed-forward 预训练。**

这意味着研究重点要从“如何把一张 2D feature map 蒸到单个场景里”转向:

- 如何让模型直接吃 Gaussian token。
- 如何在没有完整语言标注的情况下，继续扩大 3DGS 预训练规模。

SceneSplat 的回答就是:

`大规模 3DGS 数据集 -> 语言预训练学开放词汇特征 -> GaussianSSL 吸收无标注场景 -> feed-forward 3DGS scene understanding`

### 为什么这个设计有效?

- SceneSplat-7K 让 3DGS 终于有了统一的大规模训练底座。
- 语言预训练提供 zero-shot 能力，自监督部分则缓解了 3D 语言标注稀缺问题。
- 直接在 Gaussians 上建模，而不是转成点云或投回 2D，使模型更贴近 3DGS 的原生表示。

### 策略权衡

- **优点**: 具备平台价值；能支持零样本和监督两类场景理解任务；适合继续扩展到更大 3DGS 预训练。
- **代价**: 数据构建与训练成本高；目前仍以室内静态场景为主。

## Part III / Technical Deep Dive

### Pipeline

```text
SceneSplat-7K 3DGS scenes
-> generate 2D semantic / language supervision
-> vision-language pretraining on Gaussian tokens
-> Gaussian masking + self-distillation (GaussianSSL)
-> feed-forward Gaussian language features
-> zero-shot or supervised 3D semantic segmentation
```

### 关键信号

- SceneSplat 不再做 per-scene feature distillation，而是把 3DGS 场景理解上升到 dataset-scale pretraining。
- GaussianSSL 的作用不是提高单一 benchmark 指标，而是把大量无标注 3DGS 数据纳入同一训练体系。
- 论文还比较了 point parameters 与 3DGS parameters，证明原生 Gaussian 参数作为输入比点云参数更适合这类预训练。

### 少量关键数字

- SceneSplat-7K 含 `7916` 个场景、`4.72M` RGB 帧，平均重建质量 `29.64 PSNR`。
- 单源 ScanNet 训练时，ScanNet200 上 `18.9 f-mIoU / 31.7 f-mAcc`，高于 Mosaic3D 的 `13.0 / 24.5`。
- 三源训练时，ScanNet200 / Matterport3D / ScanNet++ 分别达到 `21.4 / 38.7`、`13.8 / 31.8`、`28.4 / 50.0`，其中 ScanNet++ 相比并发工作提升 `10.4% f-mIoU`。
- 监督分割上，优于 PTv3：ScanNet20 `77.2 / 84.6`，ScanNet200 `35.9 / 46.1`。

### 实现约束

- 数据集来自 ScanNet、ScanNet++、Replica、Hypersim、3RScan、ARKitScenes、Matterport3D 等多个来源。
- 论文明确提到数据构建本身耗费约 `150 GPU days`，说明这是基础模型路线而非轻量单场景方案。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_SceneSplat_Gaussian_Splatting_based_Scene_Understanding_with_Vision_Language_Pretraining.pdf]]
