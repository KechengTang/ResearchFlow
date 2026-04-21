---
title: "Localized Gaussian Splatting Editing with Contextual Awareness"
venue: WACV
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - localized-editing
  - object-insertion
  - object-replacement
  - illumination-aware-editing
  - anchor-view-proposal
  - di-sds
  - status/analyzed
core_operator: 先用 Anchor View Proposal 找到最能代表局部光照的视角做 2D inpainting 条件图，再通过 coarse 3D lifting + DI-SDS texture enhancement 把新对象以正确光照和遮挡关系嵌入原始 3DGS 场景。
primary_logic: |
  输入源 3DGS、目标局部 3D bbox 与文本指令，
  先在目标区域周围采样视角并通过 A VP 选出最能体现环境高光与阴影的 anchor view，
  再对该视角做 depth-conditioned 2D inpainting，并把前景分割出来用于 coarse image-to-3D generation，
  最后以 DI-SDS 在带背景上下文与深度条件的情况下细化几何和纹理，使新对象与场景光照、遮挡和外观自然融合。
pdf_ref: paperPDFs/3DGS_Editing/WACV_2025/2025_Localized_Gaussian_Splatting_Editing_with_Contextual_Awareness.pdf
category: 3DGS_Editing
created: 2026-04-18T17:00
updated: 2026-04-18T17:00
---

# Localized Gaussian Splatting Editing with Contextual Awareness

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [WACV 2025 PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Xiao_Localized_Gaussian_Splatting_Editing_with_Contextual_Awareness_WACV_2025_paper.pdf) · [Project Page](https://corneliushsiao.github.io/GSLE.html)
> - **Summary**: 这篇工作真正补的是“局部物体替换/插入时怎么和原场景的光照、阴影、遮挡自然对齐”。它不是单纯把一个生成好的 3D 物体塞进场景里，而是把 2D inpainting 的上下文理解能力和 3D lifting 结合起来，让目标对象从一开始就吸收 scene-wide illumination 线索。
> - **Key Performance**:
>   - 用户偏好中 `70.51%` 选择本方法，明显高于 `GaussianEditor 29.49%`，`Vica-NeRF 0%`。
>   - 消融显示，没有 coarse step 或没有 Texture Enhancement 时，要么多视角不一致，要么纹理细节不足；完整 pipeline 才能同时保留高光/阴影与多视角一致性。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

这篇论文不是在解决一般的文本编辑，而是在解决一个更具体也更难的问题：

**当你想在现有 3D 场景里插入或替换一个对象时，新对象如何自动匹配原场景的局部与全局光照。**

很多 text-guided 3D 生成方法能生成“看起来不错”的单物体，但一旦把它放进真实场景，最明显的问题就是：

- 光照方向不对
- 阴影/高光不匹配
- 前景与背景融合生硬
- 被遮挡关系处理不好

### 核心能力定义

- **输入**：源 3DGS 场景、待编辑的 3D bounding box、文本生成指令。
- **输出**：在目标区域内替换或插入的新对象，并与周围环境在光照和外观上尽量协调。
- **擅长**：object insertion、object replacement、显著几何变化下的局部编辑。
- **能力重点**：context-aware object synthesis，而不是简单局部重着色。

### 真正的挑战来源

- 大多数 3D 生成 prior 只理解对象本身，不理解对象放进某个已有场景后的 illumination context。
- 单个视角的条件图如果没选好，会把错误的 lighting cue bake 进整个 3D lifting 过程。
- 仅用 3D-aware prior 能保证多视角一致，但补不出足够细的纹理与场景融合细节；仅用 2D inpainting 又缺少 3D 一致性。

### 边界条件

- 论文默认待生成对象是 opaque。
- 需要预先给出目标编辑区域的 3D bounding box。
- 更偏向对象插入/替换，不是通用对象级操控系统。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

这篇工作最核心的设计哲学是：

**先找到最能代表局部环境光照的那个视角，再让整个 3D 生成过程围绕这条 illumination cue 展开。**

它没有走 physically based relighting 或复杂 inverse rendering 路线，而是利用大规模 2D inpainting 模型本身对背景上下文和光照融合的强能力，把它转成 3D 编辑的条件引导。

### The "Aha!" Moment

真正的 aha 是：

**局部 3D 编辑里最重要的不是“先生成多视角”，而是先选对 anchor view。**

作者观察到，多视角扩散模型如果输入的单视角图像已经 baked in 正确的 illumination cue，那么后续 lifting 出来的 3D 物体就更容易整体对光。所以他们提出 `Anchor View Proposal (AVP)`，在目标 bbox 周围采样一圈视角，专门挑那个最能体现强光照对比、阴影或高光的视角做条件图。

### 为什么这个设计有效？

- `AVP` 给 coarse 3D lifting 提供了“光照方向的启发式锚点”，避免从一个光照信息平淡的视角出发。
- coarse step 用 3D-aware prior，先确保物体基本几何和多视角一致性。
- fine step 用 `DI-SDS`，把深度、bbox mask、masked image 和背景上下文一起送进 depth-guided inpainting diffusion，使细节纹理和背景融合更自然。
- 这两个阶段一粗一细，分别负责“像个对的 3D 物体”和“像是长在这个场景里的 3D 物体”。

### 战略权衡

- **优点**：对 object replacement / insertion 这种最容易露馅的任务非常有针对性。
- **代价**：流程比一般局部编辑重，且依赖 bbox 定位和一个好的 anchor view。
- **定位**：这是一个 illumination-aware local insertion / replacement 管线，而不是通用文本编辑框架。

## Part III / Technical Deep Dive

### Pipeline

```text
source 3DGS + target 3D bbox + text prompt
-> render azimuth views around bbox
-> Anchor View Proposal selects the most illumination-informative view
-> depth-conditioned 2D inpainting on anchor view
-> segment foreground object from inpainted anchor view
-> coarse image-to-3D generation with 3D-aware diffusion prior
-> DI-SDS texture enhancement with scene depth and masked background context
-> merge generated object Gaussians with original scene Gaussians
```

### 关键模块

#### 1. Anchor View Proposal

AVP 本质上是在寻找“哪一个视角最能暴露目标区域的光照结构”。作者把渲染图转到 HSV 空间，用亮度分布的左右/上下差异作为启发式信号。虽然它不是严格物理估计，但足够稳地找到高光、阴影最明显的视角。

#### 2. Coarse image-to-3D generation

coarse step 主要负责把 inpainted anchor view 里的新对象 lift 成多视角一致的 3D Gaussian object。作者没有用 Point-E 初始化，而是在球体内初始化高斯，并降低初始 opacity / color，减少 floaters 和条件错配。

#### 3. DI-SDS

这部分是最关键的 operator。普通 SDS 只看渲染结果和文本，不看背景上下文；而 `DI-SDS` 把 depth-guided ControlNet 的中间特征、bbox mask 和 masked image 都并入扩散条件中，让优化同时感知：

- 目标几何深度
- 背景与前景的上下文关系
- 文本描述的目标外观

因此细化出来的纹理不只是“看起来更精致”，而是更符合当前场景光照。

### 关键实验信号

- 用户研究里，作者的方法在 realism + prompt consistency 上显著领先基线。
- 对比 `GaussianEditor` 与 `Vica-NeRF` 时，论文强调后两者缺少 explicit illumination handling，因此会出现和场景光照不匹配的替换结果。
- 消融显示：只做 coarse step 时几何一致但细节不够；跳过 coarse step 直接 enhancement 会导致多视角不稳；完整 coarse-to-fine 才能两者兼顾。

### 少量关键数字

- 用户偏好：`70.51%`
- `GaussianEditor`: `29.49%`
- `Vica-NeRF`: `0%`

### 实现约束

- 实验使用 `LERF`、`MipNeRF360` 和两个自采集带定向光照的数据集。
- coarse step 采用 `Stable Zero123`，Texture Enhancement 使用 `ControlNet Depth + Stable Diffusion Inpainting`。
- 扩散模型参数在训练中保持冻结。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/WACV_2025/2025_Localized_Gaussian_Splatting_Editing_with_Contextual_Awareness.pdf]]
