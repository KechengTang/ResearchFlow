---
title: "LangSplat: 3D Language Gaussian Splatting"
venue: CVPR
year: 2024
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - 3d-language-field
  - clip-features
  - sam-hierarchy
  - scene-wise-autoencoder
  - open-vocabulary-query
  - semantic-segmentation
  - status/analyzed
core_operator: 用 SAM 给 2D 图像先切出 whole/part/subpart 三层语义，再把对应 CLIP 特征压进 scene-wise latent space，最后把低维语言特征直接挂到每个 Gaussian 上，构成可实时查询的 3D language field。
primary_logic: |
  输入多视图图像，先训练 RGB 3DGS 场景；
  再用 SAM 生成层级 mask，并对每个 mask 提取 CLIP 特征；
  通过 scene-wise autoencoder 把高维 CLIP 特征压缩到低维 latent；
  让每个 Gaussian 学习三组层级语言特征，并在查询时解码回 CLIP 空间做开放词汇定位与分割。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/CVPR_2024/2024_LangSplat_3D_Language_Gaussian_Splatting.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:06
updated: 2026-04-20T19:06
---

# LangSplat: 3D Language Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.html) / [Project Page](https://langsplat.github.io/)
> - **Summary**: LangSplat 的核心不是单纯把 CLIP 特征塞进 3DGS，而是先用 SAM 把语义层级和边界理清，再用 scene-wise autoencoder 把显式 Gaussian 语言场压到可训练、可实时渲染的低维空间里。
> - **Key Performance**:
>   - 在 LERF 上，3D object localization 整体准确率达到 `84.3%`，高于 LERF 的 `73.6%`；3D semantic segmentation 整体 IoU 达到 `51.4%`，比 LERF 高 `14.0` 个点。
>   - 在 3D-OVS 上，整体 `mIoU = 93.4%`；同时在 RTX 3090 上相对 LERF 达到 `119x` 到 `199x` 的查询加速。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

这篇论文要解决的是：**如何让 3DGS 不只会渲染颜色，还能支持精确、快速、开放词汇的 3D 语言查询。**

此前的 3D language field 大多基于 NeRF。它们的问题不是“没有语义”，而是两头都吃亏：

- NeRF 渲染慢，开放词汇查询代价高。
- 直接蒸馏 CLIP 特征时，边界容易发虚，物体与物体之间不够干净。
- 一个点提示往往对应 whole / part / subpart 多种尺度语义，现有方法缺少稳定的层级接口。

LangSplat 想得到的能力边界很明确：

- **能做**：开放词汇 3D object localization、3D semantic segmentation、任意文本查询。
- **不主打**：动态场景、时序语义状态、点级几何理解。

### 输入 / 输出接口

- **输入**：多视图图像，以及先训练好的 RGB 3DGS 场景。
- **输出**：一个可渲染、可查询的 3D language field；给定任意文本，可输出相关 3D 点或 3D mask。

### 边界条件

- 方法依赖 CLIP 语义空间和 SAM 分割质量。
- 语义层级来自 SAM 的 whole / part / subpart 定义，因此它更适合“物体级到部件级”的查询，而不是更抽象的关系推理。
- 论文目标是静态 3D 场景，不涉及 4D 或动态语义。

## Part II / High-Dimensional Insight

### 方法真正新的地方

LangSplat 的关键不是“把 3DGS 换成更快的 NeRF”，而是同时改了两个瓶颈：

1. **语义监督的形状不对**  
   CLIP 给的是全局图文对齐特征，但 3D 查询需要的是边界清晰、像素对齐、并且带尺度语义的特征。

2. **显式 3DGS 直接学 512 维 CLIP 特征太贵**  
   若把高维语言特征直接显式挂到数百万个 Gaussian 上，显存和缓存代价会爆炸。

为此，作者做了两步：

- 先用 **SAM 的层级 mask** 把 CLIP 特征变成 three-level 的 pixel-aligned language supervision。
- 再用 **scene-wise autoencoder** 把场景内稀疏分布的 CLIP 特征压到低维 latent，再由 Gaussian 学低维语言表示。

### The "Aha!" Moment

真正的 aha 在于：

**3D 语言场最难的不是“有没有语言特征”，而是“语言特征是否边界清楚、尺度明确、还能以显式 3D 形式高效保存”。**

LangSplat 的回答是：

- 用 SAM 先把“同一个点击到底指 whole、part 还是 subpart”固定成离散层级。
- 用 scene-specific autoencoder 承认“一个具体场景里的 CLIP 特征分布比通用 CLIP 空间稀疏得多”，因此可以做激进压缩。
- 用显式 Gaussian 存三组语言 latent，让查询阶段只需先渲染、再解码，而不必像 NeRF 那样在高成本体渲染上兜圈子。

这使得方法的提升不是单点技巧，而是一个完整因果链：

`层级 mask 监督更干净 -> 语言边界更清楚 -> 低维 scene latent 让显式建模可行 -> 3DGS 保留实时渲染优势 -> 开放词汇查询又快又准`

### 为什么这个设计有效

- SAM 解决了 point ambiguity，把本来模糊的“点击点语义”变成了预定义层级。
- autoencoder 解决了显式 3DGS 承载高维语言特征的内存瓶颈。
- 3DGS 的 tile-based rasterization 让语言场查询从 NeRF 式慢查询变成接近实时的显式渲染。

### 策略权衡

- **优点**：边界更准、查询更快、开放词汇接口更自然。
- **代价**：依赖 CLIP + SAM 两个 2D foundation model，且语言场仍是 scene-specific，不是跨场景通用编码器。

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view images
-> train RGB 3DGS scene
-> SAM whole/part/subpart masks
-> CLIP features on masked regions
-> scene-wise autoencoder compresses CLIP features
-> attach low-dim language latents to each Gaussian
-> render language latents and decode back to CLIP space
-> open-vocabulary localization / segmentation
```

### 关键技术信号

- 论文明确指出，LERF 一类方法的问题不只是慢，更是 **3D language field 边界发糊**，导致 query activation 分散、mask 贴不住物体边界。
- 作者没有用额外 DINO 正则来补边界，而是直接用 SAM 的层级分割结果把监督信号重写成更结构化的语义目标。
- 把 512 维 CLIP 特征压到 `3` 维 scene latent 这一点很激进，但实验说明只要压缩发生在 scene-specific 空间里，语义仍能保住。

### 少量关键数字

- LERF 数据集上，整体 localization accuracy 从 `73.6%` 提升到 `84.3%`。
- 3D-OVS 上，整体 `mIoU = 93.4%`。
- 在 ramen 场景的 ablation 中，最终版本查询耗时 `0.26 s/query`，相对 LERF 的 `30.93 s/query` 提速约 `119x`。

### 实现约束

- 2D 语义特征提取使用 `OpenCLIP ViT-B/16`，SAM 使用 `ViT-H`。
- 每个场景先训练 `30,000` iter 的 RGB 3DGS，再训练 `30,000` iter 的语言特征。
- 场景通常约有 `2,500,000` 个 Gaussians；在 RTX 3090 上，论文报告高分辨率场景训练约 `25` 分钟、占用约 `4GB` 内存。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/CVPR_2024/2024_LangSplat_3D_Language_Gaussian_Splatting.pdf]]
