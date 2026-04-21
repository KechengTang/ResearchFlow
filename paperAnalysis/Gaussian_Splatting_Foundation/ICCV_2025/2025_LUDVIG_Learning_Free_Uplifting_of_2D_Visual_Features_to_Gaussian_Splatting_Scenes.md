---
title: "LUDVIG: Learning-Free Uplifting of 2D Visual Features to Gaussian Splatting Scenes"
venue: ICCV
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - learning-free
  - graph-diffusion
  - open-vocabulary-grounding
  - semantic-segmentation
  - status/analyzed
core_operator: 不再训练 3D 特征场，而是直接利用 3DGS 渲染权重把任意 2D foundation features 聚合到 Gaussian 上，并通过 DINOv2 几何相似图上的 graph diffusion 把粗糙响应变成可用的 3D 分割。
primary_logic: |
  输入多视图图像与训练好的 3DGS，先用渲染权重把 DINOv2、SAM、CLIP 等 2D 特征直接 uplift 到每个 Gaussian，再在由几何邻接和 DINOv2 相似度构成的图上做 diffusion 精炼，最终支持多视图分割、开放词汇定位、开放词汇分割与语义分割。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_LUDVIG_Learning_Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_Scenes.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:23
updated: 2026-04-20T19:23
---

# LUDVIG: Learning-Free Uplifting of 2D Visual Features to Gaussian Splatting Scenes

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICCV 2025 Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Marrie_LUDVIG_Learning-Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_ICCV_2025_paper.pdf)
> - **Summary**: LUDVIG 的关键判断是，很多 3D 语义方法把“2D 特征搬到 3D”这件事做得过重了。它直接复用 3DGS 的渲染权重做 feature uplifting，再用图扩散补结构，就能用更少代价拿到接近甚至超过优化式方法的效果。
> - **Key Performance**:
>   - 在 LERF object segmentation 上，达到 `64.3` overall IoU，而 LangSplat 为 `51.4`，用时从 `105` 分钟降到 `10` 分钟。
>   - 在 ScanNet semantic segmentation 上，达到 `33.9 mIoU / 51.4 mAcc`，明显高于 OpenGaussian 的 `24.7 / 41.5`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题?
LUDVIG 针对的是一个常被默认接受的前提:  
**把 2D foundation features 提升到 3D，似乎就必须额外训练一个 3D feature field。**

作者认为这一步往往太重了:

- 特征学习通常要一到两小时，场景级优化成本高。
- 不同特征类型经常要不同训练流程，不够通用。
- 很多工作只是为了把 2D 线索搬到 3D，却重复做了一遍昂贵优化。

因此 LUDVIG 想要的能力边界是:

- **能做**: 把 DINOv2、SAM、CLIP 等任意 2D 特征直接 uplift 到 3DGS；支持多视图分割、开放词汇定位和语义分割。
- **不主打**: 动态 4D 语义建模、端到端生成式语义场。

### 输入 / 输出接口

- **输入**: 训练好的 3DGS、多视图图像、任意 2D foundation model 产出的特征或掩码。
- **输出**: 每个 Gaussian 上的 uplifted feature，以及后续 3D 分割/定位结果。

### 边界条件

- 性能依赖底层 3DGS 和相机位姿质量。
- graph diffusion 明显依赖 DINOv2 这种具备结构相似性的特征；若基础特征本身太弱，扩散也难救。

## Part II / High-Dimensional Insight

### 方法真正新的地方

LUDVIG 的第一原则很直接:

**既然 3DGS 渲染本身已经定义了“哪个像素由哪些 Gaussian 贡献”，那它就天然提供了一个从 2D 回流到 3D 的加权汇聚算子。**

因此它不学习 uplift，而是直接算:

- 用渲染权重把每个像素的 2D feature 聚合回对该像素有贡献的 Gaussians。
- 再用图扩散把这些初始 3D features 沿着几何与 DINOv2 相似图传播，得到更完整的目标区域。

### The "Aha!" Moment

真正的 aha 在于:

**2D -> 3D uplifting 的本质不是“再训练一个 3D 编码器”，而是“把 rasterizer 已知的可见性和贡献权重反过来用一遍”。**

这让 LUDVIG 把问题拆成两段:

- 第一段只做无学习的 feature aggregation。
- 第二段只做 geometry-aware graph diffusion。

于是原本耦合在一起的“学习 3D 特征”和“补齐 3D 结构”，被拆成两个更便宜也更通用的操作。

### 为什么这个设计有效?

- 渲染权重已经编码了 visibility 和 contribution，直接复用比重新学更省。
- DINOv2 图扩散把局部粗糙响应变成结构完整的 3D 目标区域，尤其适合从 scribble、CLIP relevancy 这类弱输入出发。
- 同一套 uplift 流程可以兼容 SAM masks、DINOv2 features、CLIP features，通用性明显强于 task-specific 训练。

### 策略权衡

- **优点**: 快，通用，基本无额外训练；在多个任务上都能复用。
- **代价**: 本质仍依赖已有 2D 特征质量；不是一个从零学习 3D 语义结构的模型。

## Part III / Technical Deep Dive

### Pipeline

```text
trained 3DGS + multi-view images
-> extract 2D features or masks from DINOv2 / SAM / CLIP
-> uplift 2D features to Gaussians using rasterization weights
-> build Gaussian graph from geometry + DINOv2 similarity
-> graph diffusion refines coarse responses
-> multi-view segmentation / open-vocabulary localization / semantic segmentation
```

### 关键信号

- 论文最重要的贡献不是单一任务指标，而是证明“learning-free uplifting + graph diffusion”可以替代大量 scene-specific feature learning。
- DINOv2 graph diffusion 在这里承担的是结构恢复器，不是分类器。
- 作者还展示了 DINOv2 单独做分割时已经能接近 SAM-based 方法，这说明 uplift 后的 3D 结构信息本身很值钱。

### 少量关键数字

- SPIn-NeRF 多视图分割中，`Uplifting + Diffusion` 达到 `93.8 IoU`，明显高于仅几何参考的 `73.1`。
- LERF object localization: `86.3` overall，高于 LangSplat 的 `84.3`。
- LERF object segmentation: `64.3` overall，时间 `10` 分钟；LangSplat 为 `51.4`，时间 `105` 分钟。
- ScanNet semantic segmentation: `33.9 mIoU / 51.4 mAcc`，高于 OpenGaussian 的 `24.7 / 41.5`。

### 实现约束

- 论文实验使用 DINOv2、SAM/SAM2 和 OpenCLIP。
- ScanNet 语义分割实验中，LUDVIG 替换的是 OpenGaussian 的量化特征训练阶段，把约 `40` 分钟缩到约 `3` 分钟。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_LUDVIG_Learning_Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_Scenes.pdf]]
