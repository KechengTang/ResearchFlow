---
title: "TRASE: Tracking-free 4D Segmentation and Editing"
venue: 3DV
year: 2026
tags:
  - Gaussian_Splatting_Foundation
  - 4dgs
  - 4d-segmentation
  - dynamic-scene-editing
  - gaussian-clustering
  - tracking-free
  - soft-label-segmentation
  - status/analyzed
core_operator: 在 4DGS 上学习一个由 SAM masks 弱监督驱动的 32 维分割特征场，并用 soft-mined contrastive objective 替代显式 tracking/ID 传播，最后通过 3D 聚类得到跨视角时序一致的动态对象分割。
primary_logic: |
  输入动态场景的 4D 重建与多视图 SAM masks，先通过 differentiable feature rendering 在 Gaussian 上学习 32 维分割特征，并以软采样的像素对对比学习稳定跨时间跨视角一致性，再用无监督聚类把 Gaussians 组织成对象级 segment，最终支持点击/文本驱动的 4D 分割与编辑。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/3DV_2026/2026_TRASE_Tracking_free_4D_Segmentation_and_Editing.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:23
updated: 2026-04-20T19:23
---

# TRASE: Tracking-free 4D Segmentation and Editing

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [OpenReview PDF](https://openreview.net/pdf?id=p5PGrx7HbX) / [Project Page](https://yunjinli.github.io/project-sadg/)
> - **Summary**: TRASE 把 4D 分割从“先追踪对象 ID，再把标签传给 4DGS”改成“直接学习一个 tracking-free 的 4D 特征场”，从而在动态场景里更稳地获得跨视角时序一致分割，并且天然适合做对象级编辑。
> - **Key Performance**:
>   - 在五个动态 benchmark 上都给出最优结果，mIoU 分别达到 `0.8768 / 0.8663 / 0.9022 / 0.9234 / 0.9308`。
>   - 相比 DGD，训练与总体时间超过 `3x` 更快，存储开销约 `4.5x` 更小。

## Part I / The "Skill" Signature

### 它到底在解决什么问题?
TRASE 要处理的是动态场景里一个更难的问题:

**如果没有稳定 tracking labels，能不能仍然学出跨时间、跨视角一致的 4D 对象分割?**

现有路线常见痛点是:

- 依赖 video tracker 或 consistent object IDs，一旦 tracker 崩了，多视图一致性也跟着崩。
- 特征维度高、训练慢，不适合真正的交互式 4D 编辑。
- 分割往往只是附属输出，不适合作为对象级编辑接口。

TRASE 的目标因此非常明确:

- **能做**: tracking-free 的 4D 对象分割、对象移除、组合、风格迁移等编辑。
- **不主打**: 高级语义理解或语言场建模本身。

### 输入 / 输出接口

- **输入**: 动态场景 4D reconstruction、多视图图像、SAM masks。
- **输出**: 每个 Gaussian 的 32 维分割特征、聚类后的对象 segment，以及点击/文本驱动的编辑结果。

### 边界条件

- 方法建立在已有 4D reconstruction 之上，本身不是重建方法。
- 仍依赖 SAM masks 作为弱监督，只是不再依赖 tracking IDs。

## Part II / High-Dimensional Insight

### 方法真正新的地方

TRASE 的核心转向是:

**把“对象跟踪”换成“像素对关系学习”。**

它不去维护显式 object IDs，而是从每一帧的二值 masks 出发，构造 pixel-mask correspondence，再用 contrastive learning 直接训练 Gaussian feature field。

更关键的是，它没有用硬正负样本规则，而是用了 **soft-mined pixel-pair objective**:

- 只要存在足够可信的正关系，就把该像素对纳入正样本。
- 只要存在明显冲突的负关系，就纳入负样本。

### The "Aha!" Moment

真正的 aha 在于:

**动态 4D 分割不一定要先知道“这是同一个 ID”，而可以先学会“哪些像素关系在 4D 空间里应该靠近或远离”。**

这使 TRASE 避开了 tracking 路线最脆弱的部分:

- 不需要跨视角关联 object IDs。
- 不需要在每帧上维护一致标签链。
- 最后直接在 3D Gaussian feature space 里聚类，就能得到对象级 segment。

因果链很干净:

`SAM 弱监督 -> 像素对关系 -> 32 维 Gaussian 分割特征 -> 3D 聚类 -> tracking-free 4D 对象分割`

### 为什么这个设计有效?

- 对比学习对象是 pixel-pair relation，而不是 tracker ID，所以对标注噪声更鲁棒。
- feature 维度只有 `32`，比 DGD 这种高维特征更适合高效渲染与交互。
- 聚类发生在 3D Gaussian 空间里，自然带来跨视角一致性。

### 策略权衡

- **优点**: 不依赖 tracking labels；适合实时交互编辑；特征紧凑。
- **代价**: 仍受底层 4D reconstruction 与 SAM mask 质量影响；对象分割最终由聚类形成，仍有超参数敏感性。

## Part III / Technical Deep Dive

### Pipeline

```text
4D reconstruction backbone
-> render Gaussian features to image space
-> build pixel-mask correspondence from SAM masks
-> soft-mined contrastive learning on pixel pairs
-> learn 32D Gaussian segmentation features
-> cluster Gaussians into object segments
-> click/text prompt selection + object removal / composition / style transfer
```

### 关键信号

- 论文把“tracking-free”做成了真正的方法核心，而不是口号，因为监督完全绕开了显式 ID 传播。
- soft sample mining 是解决 floaters 和 out-of-FoV 错误分割的关键，而不是普通的训练技巧。
- 作者还单独比较了时间和存储，说明 TRASE 明显把交互式编辑视为目标场景。

### 少量关键数字

- 五个 benchmark 上，TRASE 的 mIoU 分别为 `0.8768 / 0.8663 / 0.9022 / 0.9234 / 0.9308`，对应 mAcc 为 `0.9831 / 0.9845 / 0.9945 / 0.9945 / 0.9917`。
- 相比 DGD，整体时间超过 `3x` 更快，存储约 `4.5x` 更省。
- 单帧 mask 监督版本与 multi-frame 版本接近，说明 tracking-free 学习本身已经足够稳定。

### 实现约束

- 特征维度为 `32`，这是其交互式效率的重要原因。
- 论文把对象编辑直接建立在 learned segmentation space 上，而不是另加单独编辑网络。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/3DV_2026/2026_TRASE_Tracking_free_4D_Segmentation_and_Editing.pdf]]
