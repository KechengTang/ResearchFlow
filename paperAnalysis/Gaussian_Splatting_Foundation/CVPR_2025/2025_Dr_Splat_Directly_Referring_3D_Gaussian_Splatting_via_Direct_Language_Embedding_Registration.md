---
title: "Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration"
venue: CVPR
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - language-embedded-gaussians
  - open-vocabulary-grounding
  - compact-semantic-representation
  - product-quantization
  - status/analyzed
core_operator: 直接把 CLIP 语言嵌入注册到主导像素射线的 Gaussians 上，并用预训练 Product Quantization 压缩嵌入，绕开 render-then-optimize 的中间特征图流程。
primary_logic: |
  输入训练图像、SAM masks 与 3DGS 场景，先沿每条像素射线把 CLIP 嵌入分配给 top-k 主导 Gaussians 并做多视图聚合，再用在大规模图像数据上训练好的 PQ 编码每个 Gaussian 的语言表示，最终直接在 3D Gaussian 空间完成文本查询、定位、选择与分割。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_Dr_Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_Embedding_Registration.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:23
updated: 2026-04-20T19:23
---

# Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf) / [Project Page](https://drsplat.github.io/)
> - **Summary**: Dr. Splat 的关键不是再学一个 3D language field，而是把语言特征直接登记到 Gaussian 本体上，再用全局 PQ 压缩保存，从而把 3D 查询从“先渲染再搜索”改成“直接在 Gaussian 空间检索”。
> - **Key Performance**:
>   - 在 LeRF-OVS 的 3D object selection 上，`Top-20` 版本达到 `43.26 mIoU / 64.32 mAcc@0.25`，优于 OpenGaussian 的 `43.06 / 59.61`。
>   - 在 ScanNet 的 10 类 open-vocabulary 3D semantic segmentation 上，`Top-40` 版本达到 `50.2 mIoU / 73.5 mAcc`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题?
Dr. Splat 解决的是一个很具体但很关键的问题:  
**如果语言特征始终停留在渲染后的 2D 特征图里，那么 3D 查询就永远要先 render，再 search，效率和完整性都很差。**

此前语言增强 3DGS 方法的问题主要有两类:

- 语言特征经常是先渲染到 2D，再与文本做比较，3D 查询要绕一大圈。
- 压缩方式往往是 per-scene 学出来的，场景外泛化和工程复用都差。

所以论文的能力目标是:

- **能做**: 文本驱动的 3D object selection、3D localization、open-vocabulary semantic segmentation。
- **不主打**: 动态场景、交互式编辑、复杂时序语义。

### 输入 / 输出接口

- **输入**: 已训练好的 3DGS、训练图像、SAM 生成的掩码、CLIP 图像与文本嵌入。
- **输出**: 每个 Gaussian 上的压缩语言表示，以及直接在 3D 空间完成的查询结果。

### 边界条件

- 方法默认单场景静态 3DGS。
- 语言特征分配依赖“主导 Gaussian”假设，遮挡复杂场景下仍可能有错误归属。

## Part II / High-Dimensional Insight

### 方法真正新的地方

Dr. Splat 的路线和 LangSplat / OpenGaussian 都不一样。  
它不是去优化一个新的 3D 语义场，而是直接问:

**能不能把语言嵌入像颜色一样，直接注册到 Gaussian 上?**

为此它做了两件事:

1. 用每条像素射线上的 top-k 主导 Gaussians 吸收该像素的 CLIP 语言嵌入。
2. 用在大规模图像数据上预训练好的 PQ，对 Gaussian 语言嵌入做全局压缩，而不是 scene-specific codebook。

### The "Aha!" Moment

真正的 aha 是:

**3D 开放词汇理解的瓶颈不一定在“如何学一个更好的语义场”，而在“是否敢把语言特征直接作为显式 3D 资产保存下来”。**

一旦你直接把语言嵌入挂到 Gaussian 上:

- 查询阶段就不用反复渲染 2D feature map。
- 3D 选择和定位直接变成对 Gaussian 集合的检索问题。
- PQ 压缩可以把存储成本压下来，而不用每个场景再训一遍压缩器。

这条链路很明确:

`像素级 CLIP 嵌入 -> 主导 Gaussian 注册 -> 多视图聚合 -> PQ 压缩保存 -> 直接 3D 查询`

### 为什么这个设计有效?

- top-k 主导 Gaussian 分配避免把语言特征平均抹到所有可见高斯上。
- 全局 PQ 让压缩过程脱离场景定制，工程复用性更强。
- 显式语言 Gaussian 也使得新评测协议可以真正度量“3D 物体是否被选中”，而不是只看 2D 渲染结果。

### 策略权衡

- **优点**: 查询链路短；不用 per-scene 优化 codebook；更贴近真实 3D 检索。
- **代价**: 直接注册语言嵌入的质量受像素射线归属质量影响；PQ 压缩也会引入一定量化误差。

## Part III / Technical Deep Dive

### Pipeline

```text
3DGS + training images
-> SAM masks + CLIP patch embeddings
-> assign embeddings to top-k dominant Gaussians on each pixel ray
-> aggregate multi-view language features per Gaussian
-> encode features with pretrained Product Quantization
-> direct 3D text query / localization / object selection / segmentation
```

### 关键信号

- 论文最核心的反常识点是“绕开 rendered feature maps”，直接在 Gaussian 上做语言登记。
- PQ 在这里不是配角，而是让显式语言高斯真正可存、可查、可扩展的关键。
- 作者还专门提出 Gaussian IoU 评测协议，说明它想把任务从 2D 热图评估推向真正 3D 体素/体积层面的评估。

### 少量关键数字

- LeRF-OVS object selection: `43.26 mIoU / 64.32 mAcc@0.25`，优于 OpenGaussian 的 `43.06 / 59.61`。
- ScanNet 10 类语义分割: `50.2 mIoU / 73.5 mAcc`。
- 3D localization 上，`Top-40` 达到 `25.4 mIoU`，在更高 IoU 阈值下也优于大多数基线。

### 实现约束

- 特征抽取依赖 CLIP 与 SAM。
- PQ codebook 不是 per-scene 学习，而是在大规模图像数据上预训练，这是方法泛化与效率的前提。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_Dr_Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_Embedding_Registration.pdf]]
