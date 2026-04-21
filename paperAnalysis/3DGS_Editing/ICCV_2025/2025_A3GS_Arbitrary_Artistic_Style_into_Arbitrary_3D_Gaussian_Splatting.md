---
title: "A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting"
venue: ICCV
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - stylization
  - feed-forward
  - zero-shot
  - gcn-autoencoder
  - adain-style-injection
  - large-scale-scene
  - status/analyzed
core_operator: 用 3D-GCN autoencoder 把 3DGS 映射到可操作 latent space，再用 AdaIN 注入任意参考风格，实现任意风格到任意 3DGS 的零样本前向风格迁移。
primary_logic: |
  输入任意 3DGS 场景与任意参考风格图，
  先用基于 3D-GCN 的 autoencoder 聚合空间相邻 Gaussian 特征并编码到 latent space，
  再在 latent 中用 AdaIN 注入风格图统计，
  最后通过解码器还原 stylized 3DGS，并借助两阶段训练实现零样本、秒级的任意风格迁移。
pdf_ref: paperPDFs/3DGS_Editing/ICCV_2025/2025_A3GS_Arbitrary_Artistic_Style_into_Arbitrary_3D_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-18T13:08
updated: 2026-04-18T13:08
---

# A3GS: Arbitrary Artistic Style into Arbitrary 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Fang_A3GS_Arbitrary_Artistic_Style_into_Arbitrary_3D_Gaussian_Splatting_ICCV_2025_paper.pdf)
> - **Summary**: A3GS 想解决的是 3DGS stylization 里“任意风格、任意场景、快速切换”这三个目标长期无法同时满足的问题。它把 3DGS stylization 从 per-style / per-scene optimization 转成 feed-forward latent editing，因此第一次把零样本、可扩展的大场景风格迁移推到比较实用的速度。
> - **Key Performance**:
>   - 典型场景只需约 `10 秒` 即可完成风格迁移，相比 StyleSplat 的约 `300 秒` 以及 StyleGaussian 需要额外训练，切换风格成本大幅下降。
>   - 用户研究中在 `Style Consistency / Content Preservation / Visual Naturalness` 上分别达到 `4.1 / 4.5 / 4.5`，同时在多视角一致性上取得最优或次优结果。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

现有 3DGS 风格迁移主要有两类问题：

- optimization-based 方法质量高，但每换一个风格都要重新优化，切换太慢；
- 轻量方法通常难以扩到大场景，或者跨视角一致性不足。

A3GS 要解决的是 **任意风格图到任意 3DGS 场景的零样本 stylization**，并且要能扩到大规模 Gaussian 场景。

### 核心能力定义

- **输入**: 任意 3DGS 场景 + 任意参考风格图。
- **输出**: 保持空间结构与内容布局的 stylized 3DGS。
- **擅长**: 大规模场景的快速风格切换。
- **边界**: 它是 appearance stylization，不处理几何编辑或语义对象替换。

## Part II / High-Dimensional Insight

### 方法设计的核心判断

A3GS 的核心判断是：**如果想让 3DGS stylization 真正变成“任意风格、任意场景”的前向过程，就必须先把离散且规模巨大的 Gaussian 集合压进一个可泛化的结构化 latent space。**

### The "Aha!" Moment

真正的 aha 在于 **3D-GCN autoencoder + AdaIN latent injection** 这组组合。

- 3D-GCN 负责在 Gaussian 邻域间做局部特征聚合，解决“大量离散 Gaussian 缺少结构”的问题；
- AdaIN 则把任意风格图的统计量注入 3DGS latent；
- 两阶段训练保证模型既学会重建场景，又学会在 latent 中做可泛化的 style transfer。

这意味着风格切换不再需要重新优化整场景，而是变成一次前向 latent 变换。

### 为什么这个设计有效？

- 直接对原始 Gaussian 做独立前向映射，通常会丢掉局部空间关系；GCN 恰好补上这一点。
- 把 style injection 放到 latent space，比直接在像素或单个 Gaussian 参数层面注入更稳。
- 分 batch 处理局部 Gaussian cluster，使它对超大场景更可扩展。

### 局限

- 质量仍然受训练数据分布影响。
- 主要解决 zero-shot style transfer，不是可交互、局部可控编辑。

## Part III / Technical Deep Dive

### Pipeline

```text
arbitrary 3DGS + arbitrary style image
-> 3D-GCN encoder aggregates local Gaussian structure
-> map scene into latent space
-> inject style statistics with AdaIN
-> decoder reconstructs stylized Gaussians
-> output stylized 3DGS in a single forward pass
```

### 关键机制

#### 1. 3D-GCN autoencoder

这是整篇的结构基础。作者发现简单 MLP 或普通图卷积都不够，必须显式利用局部几何邻域关系，才能同时保住内容结构与风格可迁移性。

#### 2. AdaIN in 3DGS latent space

把任意风格图的统计量注入场景 latent，而不是对每个视图单独做 2D stylization，再回蒸到 3D。

#### 3. Two-stage training and batched scalability

两阶段训练先学重建再学风格化；对于超大场景，方法支持按 local Gaussian clusters 分批处理，论文中单批最多可处理约 `15M` Gaussian primitives。

### 关键实验信号

- 典型场景风格迁移约 `10 秒`。
- 多视角一致性上，短程指标达到 `0.033 LPIPS / 0.029 RMSE`，长程为 `0.061 / 0.057`。
- 用户研究三项主观指标都优于 StyleRF、StyleSplat 和 StyleGaussian。

### 实现约束

- 需要专门构建 3DGS stylization 训练集。
- 方法主要面向快速全局 style transfer，而不是局部区域编辑。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ICCV_2025/2025_A3GS_Arbitrary_Artistic_Style_into_Arbitrary_3D_Gaussian_Splatting.pdf]]
