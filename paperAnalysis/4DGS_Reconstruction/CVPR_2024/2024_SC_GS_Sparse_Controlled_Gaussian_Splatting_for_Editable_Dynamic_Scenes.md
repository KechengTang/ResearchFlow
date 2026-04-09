---
title: "SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes"
venue: CVPR
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - sparse-control-points
  - deformation-mlp
  - linear-blend-skinning
  - arap-regularization
  - motion-editing
  - editable-dynamic-scene
  - status/analyzed
core_operator: 将动态场景的运动压缩到少量 sparse control points 上，并用 deformation MLP 预测控制点的时变 6DoF 变换，再通过 KNN + LBS 把稠密高斯运动插值出来，从而同时获得高质量重建与可编辑运动表示。
primary_logic: |
  先在 canonical space 中学习稀疏控制点及其半径，
  再用 deformation MLP 根据控制点位置与时间预测每个控制点的 6DoF 运动，
  随后通过 KNN 与 RBF 权重把控制点变换插值到每个 Gaussian 上，
  最后在 ARAP 约束和自适应控制点增删下完成动态渲染与运动编辑。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_SC_GS_Sparse_Controlled_Gaussian_Splatting_for_Editable_Dynamic_Scenes.pdf
category: 4DGS_Reconstruction
created: 2026-04-09T18:42
updated: 2026-04-09T18:42
---

# SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://yihua7.github.io/SC-GS-web/) · [arXiv 2312.14937](https://arxiv.org/abs/2312.14937)
> - **Summary**: SC-GS 的核心想法是把“动态场景里最难学的时序运动”从海量高斯上剥离出来，只交给少量可解释的 sparse control points 表达；高斯负责外观，控制点负责运动，于是动态重建天然获得了可编辑性。
> - **Key Performance**:
>   - 在 D-NeRF 上，论文报告平均 `43.31 PSNR / 0.9976 SSIM / 0.0063 LPIPS`。
>   - 在 NeRF-DS 上，平均结果为 `24.1 PSNR / 0.891 MS-SSIM / 0.140 LPIPS`，并额外支持 motion editing。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？

SC-GS 针对的是动态新视角合成里的一个关键矛盾：

- 想要高质量动态渲染，就需要细致建模每个位置的运动
- 但如果直接给每个 Gaussian 学独立运动，表示既重又不稳，也几乎不可编辑

所以它要解决的是：  
**如何在保持 3DGS 高保真渲染的同时，把动态场景的运动表示变成一个紧凑、可解释、可编辑的空间。**

### 核心能力定义

- **输入**：来自 monocular dynamic video 的图像序列
- **输出**：可渲染的动态高斯场景，以及一个显式可操作的运动控制图
- **特别能力**：不仅做 reconstruction，还支持用户控制 motion editing

### 真正的挑战来源

- 动态场景运动是高维、时变、局部复杂的
- 每个 Gaussian 单独建模运动会导致参数爆炸与过拟合
- 纯隐式 deformation field 不容易给后续编辑留接口

### 边界条件

- 方法假设局部运动具有一定低秩性和局部刚性
- 对强 specular 场景没有专门建模
- 当相机位姿误差较大时，NeRF-DS 上性能会受影响

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

SC-GS 的设计非常像“运动骨架 + 外观蒙皮”的 3DGS 版本：

- 高斯负责外观与渲染
- 少量控制点负责运动基
- 每个 Gaussian 的时变轨迹通过对邻近控制点插值获得

这意味着作者把动态 4D 表示问题，从“每个 Gaussian 都学自己的完整运动”，改写成了“**用稀疏可解释基控制稠密高斯**”。

### The "Aha!" Moment

真正的 aha 是：

**动态场景里真正稀缺且值得压缩的，不是外观，而是运动自由度。**

作者观察到真实场景运动常常是：

- 稀疏的
- 空间连续的
- 局部刚性的

所以没有必要让数十万高斯都独立学 motion，只要让大约 `512` 个控制点去描述底层运动空间，再把这个运动插值到高斯上即可。

### 为什么这个设计有效

这样做同时带来三个收益：

- **参数更紧凑**：运动从 dense per-Gaussian 变成 sparse basis
- **优化更稳**：ARAP 约束控制点局部刚性，减少无意义漂移
- **更可编辑**：既然运动已经显式压缩到 control graph，用户自然可以直接编辑控制点轨迹

### 战略权衡

- 优点：表示层更结构化，兼顾重建和编辑
- 代价：控制点假设对强非刚性、极细碎拓扑变化未必最优

## Part III / Technical Deep Dive

### Pipeline

```text
monocular dynamic video
-> canonical Gaussians + sparse control points
-> deformation MLP predicts time-varying 6DoF on control points
-> KNN + RBF weights + LBS interpolate control motion to Gaussians
-> rendering loss + ARAP regularization + adaptive control point update
-> dynamic scene reconstruction and motion editing
```

### 关键模块

#### 1. Sparse Control Points

作者在 canonical space 里学习一组控制点，每个控制点有：

- 3D 位置
- 控制半径
- 时变 6DoF 变换

这组点不是为了直接渲染，而是为了构成一个稀疏运动基空间。

#### 2. Deformation MLP

与其为每个时间步直接存控制点变换，不如用一个 MLP 根据 `(point, time)` 查询控制点在该时刻的旋转和平移。  
这让运动成为一个可查询的连续场，而非离散查表。

#### 3. Gaussian Motion via LBS

每个 Gaussian 只查自己的 `K=4` 个近邻控制点，通过 RBF 权重插值这些控制点的变换，再用 LBS 得到本时刻高斯的位置与旋转。  
这一步本质上就是把“控制图驱动形变”搬到了 4DGS。

#### 4. ARAP Regularization

作者对控制点轨迹构图，并施加 ARAP loss，鼓励局部运动尽可能刚性。  
这不是传统几何编辑里的附属项，而是整个 motion space 可泛化、可编辑的关键稳定器。

#### 5. Motion Editing

训练好后，作者直接在控制点图上做图形学式的 ARAP deformation，再把编辑后的控制点变换回灌到 Gaussians。  
因此 SC-GS 的编辑能力来自**表示层本身可操作**，而不是后接一个额外编辑器。

### 关键实验信号

- 在 D-NeRF 上显著优于 D-NeRF、TiNeuVox、K-Planes、4D-GS 等基线
- 在 NeRF-DS 上虽然没有专门处理 specular，但平均表现仍领先
- 论文展示了 motion out of training set 的编辑结果，这对“可编辑动态场景”很关键

### 少量关键数字

- D-NeRF 平均：`43.31 PSNR / 0.9976 SSIM / 0.0063 LPIPS`
- 去掉 control points 的基线只有：`38.512 / 0.9922 / 0.0162`
- 去掉 ARAP loss 后也会退化到：`42.617 / 0.9963 / 0.0067`

### 对你可迁移的 operator

- **motion-as-sparse-basis instead of per-Gaussian deformation**
- **control-point-driven 4DGS**
- **ARAP regularized motion space**
- **editable motion graph from reconstruction-time representation**

如果你以后想做 4DGS 编辑，SC-GS 特别值得记住，因为它告诉你：  
**“可编辑性”最好在动态表示建模时就埋进去，而不是重建完再临时外接。**

### 实现约束

- 控制点数量远小于高斯数量
- 邻域插值默认使用 KNN 与 RBF 权重
- motion editing 的自然前提是控制点图已经学到合理的局部相关性

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_SC_GS_Sparse_Controlled_Gaussian_Splatting_for_Editable_Dynamic_Scenes.pdf]]
