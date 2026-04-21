---
title: "3D Geometry-Aware Deformable Gaussian Splatting for Dynamic View Synthesis"
venue: CVPR
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - geometry-aware-deformation
  - sparse-convolution
  - monocular-dynamic-scene
  - local-geometry
  - status/analyzed
core_operator: 将动态场景表示为随时间形变的 3D Gaussians，并通过对 Gaussian 分布体素化后提取的 sparse-conv geometry features 来约束 deformation learning，使动态形变显式感知 3D 局部结构。
primary_logic: |
  先用 3D Gaussians 建立场景的 canonical 表示，
  再将高斯分布体素化并通过 sparse convolution 提取局部几何特征，
  之后把 geometry-aware features 注入 deformation network 预测高斯的时序平移与旋转，
  最终获得几何一致性更强的动态视角合成与 3D 动态重建结果。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_3D_Geometry_Aware_Deformable_Gaussian_Splatting_for_Dynamic_View_Synthesis.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:07
updated: 2026-04-18T16:07
---

# 3D Geometry-Aware Deformable Gaussian Splatting for Dynamic View Synthesis

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://npucvr.github.io/GaGS/) · [arXiv 2404.06270](https://arxiv.org/abs/2404.06270)
> - **Summary**: 这篇论文抓住的是 monocular dynamic reconstruction 里最根本的缺口: 现有 deformation learning 往往只看位置编码或隐式特征，缺少对真实 3D 邻域结构的感知，因此形变容易不连贯、不几何一致。它通过把 Gaussian 几何显式体素化并用 sparse conv 提取局部几何特征，让 deformation network 真正“看见”3D 结构。
> - **Key Performance**:
>   - 论文在 D-NeRF 和真实场景上都报告新的 SOTA 结果，重点提升动态视角合成与 3D 动态重建的一致性。
>   - 文中给出的实现代价大约为平均 `2h` 训练、固定视角约 `12 FPS` 渲染、模型规模约 `48 MB`，明显偏向质量导向。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

这篇论文要解决的是动态场景形变学习里一个非常核心的问题:

**deformation model 会动，但不一定按几何规律动。**

在 monocular dynamic view synthesis 中，这个问题尤其严重，因为:

- 观测本来就少
- 运动和形状存在歧义
- 如果 deformation 只靠坐标编码或隐式场，很容易学出局部不连贯的运动

所以作者聚焦的是:

**如何让 Gaussian 的动态形变显式利用 3D 局部几何结构，从而得到更几何一致的动态重建。**

### 核心能力定义

- **输入**: monocular dynamic video
- **输出**: 任意视角、任意时刻的动态场景渲染
- **强项**: 几何一致形变、局部平滑、动态重建质量
- **弱项**: 极致实时性、语义级控制、单次前向泛化

### 真正的挑战来源

- 真实世界的形变不是孤立点运动，而是与邻域结构深度耦合
- 早期动态 NeRF/3DGS 的 deformation 往往没有充分编码局部 3D geometry
- monocular 设定下没有多视角一致性兜底，几何错误更容易被学进去

### 边界条件

- 仍属于 canonical field + deformation model 路线
- 依赖体素化和 sparse conv 提取几何特征，因此会引入额外结构开销
- 更适合连续形变，不专门面向大拓扑变化

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

这篇论文最重要的设计哲学是:

**动态形变不是纯时间函数，而是受局部几何结构约束的 3D 过程。**

所以作者没有继续在坐标编码上做文章，而是直接从 Gaussian 本身的空间分布中抽几何特征，再把这些特征送入 deformation learning。

### The "Aha!" Moment

真正的 aha 是:

**如果 deformation model 根本不知道一个点周围的 3D 结构长什么样，它就不可能稳定学出几何一致的形变。**

作者为此做了一个很有代表性的 operator:

- 把一组 Gaussian distributions 体素化
- 在体素空间上用 sparse convolution 提取 geometry-aware features
- 再让 deformation network 基于这些特征预测动态运动

这相当于把“高斯的局部结构”变成了 deformation 的显式输入，而不是隐含假设。

### 为什么这个设计有效

因为真实物体的运动通常具有邻域耦合性:

- 邻近点的运动往往相关
- 局部几何边界决定了形变传播方式
- 单点位置编码无法提供这种结构上下文

geometry-aware features 的作用，就是把 deformation 的条件从“当前位置在哪”提升为“当前位置处于什么局部几何环境中”。  
这会直接改善局部平滑性和结构一致性。

### 对我当前方向的价值

对你现在的方向，这篇论文很值得记住，因为它不是单纯提出一个新 backbone，而是给出一个可迁移的普适判断:

**4DGS 的动态质量，往往取决于 deformation 是否真正使用了 geometry prior。**

这对后续做 4D 编辑、语义 propagation、局部控制都很关键，因为一旦 deformation 本身不稳，上层编辑就会很难稳定。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 动态编辑通常要求修改能沿局部运动正确传播，这篇的 geometry-aware deformation 正好提供了更可靠的传播底座。
- **与 3DGS_Editing**: 很多 3D 编辑方法也依赖几何局部性来稳定局部修改；这篇虽然做的是重建，但“结构感知形变”这个 operator 可直接借给编辑系统。
- **与 feed-forward Gaussians**: 它不是 feed-forward，但说明未来 feed-forward 4D 模型若只预测位移而不显式注入几何上下文，很可能仍会欠稳。

### 战略权衡

- 优点: 形变更几何一致，动态重建更稳
- 代价: 稀疏体素与 sparse conv 带来额外复杂度，实时性不如纯轻量 deformation 路线

---

## Part III / Technical Deep Dive

### Pipeline

```text
monocular video
-> canonical 3D Gaussian scene
-> voxelization of Gaussian distributions
-> sparse convolution geometry feature extraction
-> geometry-aware deformation prediction
-> time-varying Gaussian translation/rotation
-> dynamic view synthesis
```

### 关键模块

#### 1. Canonical Gaussian Representation

作者延续 canonical scene + deformation 的主流框架，先学静态高斯底座，再通过时间条件做动态展开。

#### 2. Geometry Feature Extraction by Sparse Convolution

最关键的一步是把 Gaussian 分布体素化，再用 sparse conv 抽取局部 3D 结构特征。  
这让 deformation 不再只是坐标回归，而是带有结构感知的预测。

#### 3. Geometry-Aware Deformation Learning

高斯在时间上的移动和旋转由 geometry-aware features 引导。  
因此模型学到的不是“哪里该动”，还包括“在这样的几何局部中应该怎样动得更合理”。

### 关键实验信号

- 论文强调 synthetic 和 real datasets 上都优于以往方法，说明收益不只来自数据集偏置
- 其主张并不是提高一点像素指标，而是让动态场景更几何一致、重建更可靠
- 相比只靠 positional encoding 或 grid interpolation 的方法，它强调结构特征输入是关键因果改动

### 少量关键数字

- 文中给出平均约 `2h` 训练
- 固定视角渲染约 `12 FPS`
- 模型规模约 `34 MB` 点云 + `14 MB` 网络

### 局限、风险、可迁移点

- **局限**: 实时性一般，且主要面向连续非刚体形变
- **风险**: 体素化分辨率和 sparse conv 表达能力如果设置不当，可能让细粒度局部结构仍被平滑掉
- **可迁移点**: Gaussian voxelization、geometry-aware motion features、structure-conditioned deformation 都很适合迁移到 4D 编辑和鲁棒动态建模

### 实现约束

- monocular dynamic view synthesis 设定
- 需要体素化与 sparse convolution backbone
- 质量优先而非极致实时

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_3D_Geometry_Aware_Deformable_Gaussian_Splatting_for_Dynamic_View_Synthesis.pdf]]
