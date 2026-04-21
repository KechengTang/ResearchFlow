---
title: "Dynamic 2D Gaussians: Geometrically Accurate Radiance Fields for Dynamic Objects"
venue: ACMMM
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 2dgs
  - dynamic-object-reconstruction
  - mesh-reconstruction
  - sparse-control-points
  - surface-consistency
  - status/analyzed
core_operator: 以 2D Gaussians 作为几何主表示，用 sparse-controlled points 驱动其动态形变，并结合基于渲染 RGB mask 的深度过滤去除 floaters，从而兼顾动态新视角渲染与高质量 mesh 提取。
primary_logic: |
  先用 2D Gaussians 建模物体表面并以 sparse-controlled points 表达其动态形变，
  再从高质量渲染图提取物体 mask 并对渲染深度做遮罩清理以去除 floaters，
  最后通过更干净的一致深度与表面表示导出平滑且细节更完整的动态 mesh 序列。
pdf_ref: paperPDFs/4DGS_Reconstruction/ACMMM_2025/2025_Dynamic_2D_Gaussians_Geometrically_Accurate_Radiance_Fields_for_Dynamic_Objects.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:07
updated: 2026-04-18T16:07
---

# Dynamic 2D Gaussians: Geometrically Accurate Radiance Fields for Dynamic Objects

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2409.14072](https://arxiv.org/abs/2409.14072) · [Code](https://github.com/hustvl/Dynamic-2DGS)
> - **Summary**: D-2DGS 关心的不是把动态物体渲染得更像，而是把动态物体真正做成可导出、可用、几何干净的表面。它把 2DGS 的表面一致性优势带到动态物体重建里，并额外处理 floaters 和不稳定深度，目标是得到高质量动态 mesh，而不是只做视频级 novel view synthesis。
> - **Key Performance**:
>   - 论文强调在 D-NeRF 与 DG-Mesh 风格基准上能从 sparse inputs 重建更平滑、细节更完整的动态 mesh。
>   - 综合资源表中，D-2DGS 给出约 `32 mins` 训练、`71 FPS` 渲染、`20 MB` 存储，明显偏向“质量优先的几何型表示”。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

很多 4D 表示已经能把动态对象渲染得很好，但要导出高质量 mesh 时会露馅:

- 3D Gaussian 对表面几何并不天然友好
- 动态重建里常出现 floaters、深度污染和表面不稳定
- sparse views 下要同时兼顾运动建模和几何精度尤其困难

D-2DGS 的目标非常明确:

**不仅要渲染动态对象，还要把动态对象恢复成几何上更可信、更平滑的 mesh 序列。**

### 核心能力定义

- **输入**: 稀疏多视角动态物体图像
- **输出**: 高质量 novel views 与可导出的 dynamic mesh sequence
- **强项**: 几何准确性、表面一致性、mesh 质量
- **弱项**: 大场景动态重建、纯速度导向的 4DGS 实时系统

### 真正的挑战来源

- 动态对象的非刚体运动本身就很难稳定建模
- 传统 3DGS 更偏渲染表示，不保证表面多视图一致
- sparse input 下的 floaters 会直接污染 TSDF 或后续 mesh 提取

### 边界条件

- 这篇更像“动态物体几何重建”而不是一般 4D scene reconstruction
- 表示主体是 2D Gaussian surface，不是通用的 3D/4D scene Gaussians
- 如果任务重点是大场景实时交互，这条路线不是最优解

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

D-2DGS 的核心哲学是:

**如果目标是高质量动态 mesh，就不能只沿用为渲染优化的 3D Gaussian 表示，而要从一开始选择更贴近表面的 2D Gaussian 几何载体。**

因此它不是在“3DGS 上补一个 mesh export”，而是把表面一致性放进表示层本身。

### The "Aha!" Moment

真正的 aha 有两个层次:

1. **用 2D Gaussians 取代 3D Gaussians 作为动态表面载体**
2. **把 mesh 质量问题显式转成 floaters 清理和深度净化问题**

作者意识到，仅有更好的动态形变还不够。  
如果深度里混着背景颜色相近的漂浮高斯，后续 TSDF 或 mesh extraction 一样会失败。  
所以它引入从高质量渲染 RGB 图中提取 object mask，再去遮罩渲染深度的流程，本质上是在给 dynamic mesh extraction 加一道几何清洗阀门。

### 为什么这个设计有效

因为它同时作用于两个最关键的信息瓶颈:

- **表示瓶颈**: 2D Gaussian 更接近 surface primitive
- **提取瓶颈**: mask-guided depth filtering 降低 floaters 对 mesh 的污染

前者决定“能不能表达出像表面的东西”，后者决定“最终导出的 mesh 能不能真的干净”。

### 对我当前方向的价值

这篇论文对你最大的价值，是它提供了一个很清晰的判断:

**当任务核心是可编辑几何或可导出表面，而不是纯新视角渲染时，表示形式本身必须换。**

这对你做 Gaussian editing/reconstruction 的边界划分很重要:

- 只做渲染与外观控制，3D/4DGS 够用
- 要做可靠 geometry carrier，2DGS 或 mesh-aware Gaussian 更值得关注

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 如果将来想做“对动态物体表面真实改形”而不是只改 appearance，这篇提供了比普通 4DGS 更好的几何底座。
- **与 3DGS_Editing**: 很多 3DGS 编辑论文受限于几何不够硬，D-2DGS 这种表面型表示能缓解 mask 漏选、编辑外溢和局部形变不稳的问题。
- **与 feed-forward Gaussians**: 它完全不是 feed-forward，但它定义了一个高质量 geometry teacher；未来 feed-forward 动态几何模型可以考虑蒸馏 2DGS 风格的表面结构。

### 战略权衡

- 优点: mesh 友好、表面一致性强、动态物体细节恢复更可靠
- 代价: 渲染速度不是同类最快，适用范围也更偏 object-level

---

## Part III / Technical Deep Dive

### Pipeline

```text
sparse dynamic object images
-> 2D Gaussian surface representation
-> sparse-controlled points for deformation
-> render RGB and depth
-> mask-guided depth cleanup to remove floaters
-> dynamic mesh extraction
```

### 关键模块

#### 1. 2D Gaussians As Surface Primitives

2DGS 的关键优势是更天然地贴近表面，这让它比普通 3DGS 更适合做几何恢复，而不是只做体渲染式近似。

#### 2. Sparse-Controlled Points

作者没有让每个 2D Gaussian 独立学复杂运动，而是用 sparse-controlled points 来驱动其形变。  
这相当于给动态表面加了一个更稳定的控制骨架。

#### 3. Masked Depth Cleaning

这是整篇里很关键、也很工程有效的一步。  
通过从高质量渲染图中提取 object mask，再去清理渲染深度，作者把 floaters 问题显式压下去，直接服务于后续 mesh extraction。

### 关键实验信号

- 论文的主要证据不是单看 NVS 指标，而是看 mesh 相关的质量
- 作者明确区分了 D-NeRF 类数据集上的渲染评估与 DG-Mesh 风格数据集上的几何评估
- 综合表说明它愿意牺牲一部分纯 FPS，换来更可靠的几何和更可用的 mesh

### 少量关键数字

- 综合资源对比表给出 `32 mins` 训练、`71 FPS` 渲染、`3690 MB` 显存、`20 MB` 存储
- 与高 FPS 的动态 3DGS 方法相比，它更偏向“几何准确性优先”的路线

### 局限、风险、可迁移点

- **局限**: 更偏动态物体而不是开放场景；若场景规模很大或背景复杂，流程会变重
- **风险**: mask 提取质量会直接影响深度清洗效果，错误 mask 会污染最终 mesh
- **可迁移点**: 2D surface primitives、sparse-controlled deformation、mask-guided depth cleanup 都很适合迁移到可编辑动态几何和 4D asset creation

### 实现约束

- 偏 sparse-view dynamic object reconstruction
- 依赖后续 mesh extraction 流程
- 不以极致实时为首要目标

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/ACMMM_2025/2025_Dynamic_2D_Gaussians_Geometrically_Accurate_Radiance_Fields_for_Dynamic_Objects.pdf]]
