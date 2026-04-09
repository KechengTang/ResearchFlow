---
title: "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis"
venue: CVPR
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - spacetime-gaussians
  - temporal-opacity
  - parametric-motion
  - splatted-features
  - guided-sampling
  - compact-model
  - high-resolution-rendering
  - status/analyzed
core_operator: 在 3D 高斯基础上加入 temporal opacity 与参数化运动/旋转得到 Spacetime Gaussians，再用 splatted neural features 替代 SH 建模时变外观，并通过误差与粗深度引导的新高斯采样提高难区域收敛质量。
primary_logic: |
  先将每个 3D Gaussian 扩展为带 temporal opacity、运动与旋转参数的时空高斯，
  再把神经特征而非球谐系数 splat 到图像平面并经 MLP 生成颜色，
  同时在训练中根据重建误差与粗深度沿高误差像素射线补采样新高斯，以提升动态场景重建质量与速度。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_Spacetime_Gaussian_Feature_Splatting_for_Real_Time_Dynamic_View_Synthesis.pdf
category: 4DGS_Reconstruction
created: 2026-04-09T17:08
updated: 2026-04-09T17:08
---

# Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.html) · [Project Page](https://oppo-us-research.github.io/SpacetimeGaussians-website/) · [arXiv 2312.16812](https://arxiv.org/abs/2312.16812) · [Code](https://github.com/oppo-us-research/SpacetimeGaussians)
> - **Summary**: Spacetime Gaussian Feature Splatting 走的是和 4D-GS 不太一样的一条线。它不是“基础高斯 + 外部形变场”范式，而是直接把时间相关属性内化进高斯本身，包括 temporal opacity、参数化运动与旋转；再用 neural feature splatting 代替传统 SH 表示 view/time-dependent appearance。这样做的目标是把动态、高分辨率和紧凑存储一起兼顾。
> - **Key Performance**:
>   - 项目页报告 lite 版本在 RTX 4090 上可实现 8K 分辨率 `60 FPS` 量级渲染。
>   - 论文强调在 Neural 3D Video 等数据集上同时取得高质量、速度和紧凑模型大小。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

这篇论文针对的是动态新视角合成里另一个重要矛盾：

- 想要高质量时变外观
- 想要高分辨率实时渲染
- 又不想模型太大

很多方法只能在其中两者之间做权衡。  
Spacetime Gaussian Feature Splatting 的目标是尽量三者兼得。

### 它的方法直觉

它的方法直觉和 4D-GS 很不一样：

1. **时间属性可以直接内嵌到 Gaussian primitive 本身**
2. **外观不一定要靠 SH，可以用更灵活的神经特征 + MLP**
3. **难收敛区域应该被显式补采样，而不是等优化自己修好**

这让它更像一种“增强版动态 Gaussian primitive”路线。

### 一句话能力画像

- **核心 primitive**：Spacetime Gaussians
- **动态属性**：temporal opacity + parametric motion/rotation
- **外观建模**：splatted features + MLP
- **收敛增强**：error/depth-guided Gaussian sampling

### 对你的研究最重要的启发

这篇论文最值得你记住的是：

**动态高斯不一定非得由外部 deformation field 驱动，也可以把一部分时间行为直接做成高斯自身参数。**

这对于以后做：

- 特征型 4DGS
- 可编辑 4DGS
- 带语言或控制信号的高斯表示

都很有启发，因为它提供了一个“把更多能力塞进高斯 primitive 自身”的方向。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

它的新意主要体现在三个地方，而项目页也正好把这三点列得很清楚：

- Spacetime Gaussians
- splatted feature rendering
- guided Gaussian sampling

这三者分别对应：

- 动态结构表示
- 复杂时变外观建模
- 训练时难区域收敛问题

### The "Aha!" Moment

真正的 aha moment 是：

**如果高斯本身就携带时间相关的透明度、运动和旋转参数，那么动态场景中的“出现/消失/短暂内容”也能被更自然地表达。**

这比只靠外部形变去解释所有动态现象更直接。  
也正因此，论文特别强调它能表示：

- static content
- dynamic content
- transient content

### 为什么这个思路值得长期记住

4DGS 社区后面很多工作都在争论一件事：  
动态到底应该放进“场”，还是放进“粒子本身”。

Spacetime Gaussian Feature Splatting 是后者的一个非常典型的代表。  
所以它对你后面做方法对比时很重要。

### 它的边界

- 把更多时间属性塞进高斯本身，可能带来更复杂的 primitive 参数设计
- feature + MLP 外观建模虽然灵活，但系统复杂度也会提高
- guided sampling 需要额外训练策略来管理新增高斯

---

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic scene
-> initialize spacetime Gaussians
-> splat neural features to image plane
-> MLP decodes features to color
-> monitor reconstruction error + coarse depth
-> sample new Gaussians along hard rays
-> real-time high-resolution dynamic view synthesis
```

### 关键模块

#### 1. Spacetime Gaussians

项目页明说，每个 Gaussian 在 3DGS 基础上额外拥有：

- temporal opacity
- parametric motion
- parametric rotation

这使其可以同时建模静态、动态和短暂成分。

#### 2. Splatted Feature Rendering

这是另一个核心创新。  
作者用 neural features 替代 SH，然后把这些特征 splat 到 2D 图像平面，再通过 MLP 输出颜色。

好处是：

- 更适合表达 view-dependent appearance
- 也更适合表达 time-dependent appearance
- 同时仍能保持较紧凑存储

#### 3. Guided Sampling

项目页明确说明：

- 对训练误差大的像素
- 结合 coarse depth
- 沿其射线补采样新的 Gaussians

这是一种很实用的训练增强策略，因为很多动态场景的难点就是初始化高斯太 sparse，导致某些局部始终学不好。

### 关键实验信号

- 论文强调在多个 established datasets 上 SOTA quality + speed + compactness。
- 项目页给出了 temporal consistency 可视化，说明方法能减少 temporal noise。
- 它特别强调高分辨率实时能力，这是其与很多动态 NeRF 路线的显著区别。

### 对你可迁移的 operator

- **time-aware Gaussian primitives**
- **feature splatting instead of SH for dynamic appearance**
- **error/depth-guided adaptive Gaussian densification**

如果你未来想做 feature-rich 或可编辑的 4DGS，这篇论文的 primitive 设计思想很值得借鉴。

---

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_Spacetime_Gaussian_Feature_Splatting_for_Real_Time_Dynamic_View_Synthesis.pdf]]
