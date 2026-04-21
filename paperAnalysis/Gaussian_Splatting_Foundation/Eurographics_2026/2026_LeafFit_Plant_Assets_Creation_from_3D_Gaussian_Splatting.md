---
title: "LeafFit: Plant Assets Creation from 3D Gaussian Splatting"
venue: Eurographics
year: 2026
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - instance-segmentation
  - mesh-extraction
  - template-instancing
  - moving-least-squares
  - real-time-rendering
  - status/analyzed
core_operator: 先基于 Gaussian center 上的 geodesic distance 自动切分叶片实例并选出模板叶，再用可微 Moving Least Squares 把单个模板叶拟合到所有其他叶片，最后只为模板提取薄片 mesh 与纹理，在运行时通过顶点着色器按实例控制点实时变形成整株植物资产。
primary_logic: |
  输入单株植物的 3DGS，
  先由用户指定 root Gaussian，再用 geodesic distance、apex grouping 和 petiole detection 自动分出每片叶子，并保留手工 drag/brush 修正作为 fallback，
  随后选择一个模板叶，借助控制点和可微 MLS 优化把模板叶对齐到其它叶片，
  再只对模板叶做 BPA 薄片 mesh 重建与纹理提取，并把每片叶子的控制点参数存成轻量实例数据，
  运行时通过 vertex shader 评估 MLS 形变，实时生成可编辑、低存储、游戏引擎友好的植物 mesh 资产。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/Eurographics_2026/2026_LeafFit_Plant_Assets_Creation_from_3D_Gaussian_Splatting.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T19:20
updated: 2026-04-18T19:20
---

# LeafFit: Plant Assets Creation from 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://netbeifeng.github.io/LeafFit/) · [GitHub](https://github.com/netbeifeng/leaf_fit) · [arXiv 2602.11577](https://arxiv.org/abs/2602.11577)
> - **Summary**: LeafFit 不是在追求更好的植物 3DGS 重建，而是在解决“高保真 3DGS 如何真正变成游戏和内容生产里可用的植物资产”这个落地问题。它抓住了植物叶片高度重复、但又略有形变的结构先验，把沉重的高斯场压缩成“一个模板叶 + 每片叶子的轻量变形参数”。
> - **Key Performance**:
>   - 分割上达到 `Acc 98.95 / mIoU 98.20 / mF1 99.08 / PQ 98.20`，显著优于 Point Transformer 和 Euclidean Density heuristic。
>   - 资产效率上，整株植物 mesh 仅 `1.13 MB`、`2432` 顶点、`11980.11 FPS`；单叶模板仅 `0.334 MB`、`2048` 顶点，同时优于基于 2DGS / GOF 的隐式转网格基线。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

LeafFit 关注的不是常规重建论文里的 novel-view synthesis，而是一个很工程、但也很有价值的问题：

**如何把高保真的植物 3DGS 转成真正可编辑、可实例化、可在游戏引擎里高效运行的 mesh 资产。**

原始 3DGS 对植物确实很合适，因为它能保住薄叶和复杂轮廓。  
但它也有两个明显缺点：

- 原始高斯表示内存占用高，而且没有显式拓扑，难以融入标准内容生产流程。
- 现有 implicit-surface 转 mesh 的方法会把薄叶变厚，且网格过密，不适合做游戏资产。

### 核心能力定义

- **输入**：单株植物的 3D Gaussian splatting 表示。
- **输出**：模板叶 mesh、纹理，以及每片叶子的轻量控制参数，最终可实例化为整株植物资产。
- **擅长**：叶片重复度高、薄片结构明显的植物资产压缩与编辑。
- **不擅长**：严重遮挡、复杂拓扑重叠以及纹理差异巨大的叶片群。

### 真正的挑战来源

- 同一株植物的叶片“相似但不完全相同”，简单 retrieval 不够，逐叶单独建模又浪费。
- 植物叶片是薄片双面结构，implicit surface 很容易产生厚化和双层伪影。
- 高斯点没有拓扑，必须先解决 per-leaf segmentation，再谈模板化和实例化。

### 边界条件

- 方法默认输入是已经清理过背景的单株植物 3DGS。
- 某些场景仍需要手工修正分割。
- 叶片纹理最终是模板共享的，因此独立叶片的细微纹理差异不会被完全保留。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

LeafFit 的思路非常清楚：

**不要把植物当成一大团高斯去转网格，而要先把它因式分解成“许多相似的叶片实例 + 一个共享模板”。**

这其实是把植物建模从“点云/高斯重建问题”改写成了“模板复用与形变参数化问题”。

### The "Aha!" Moment

真正的 aha 是：

**叶片之间最大的冗余不是纹理，而是形状结构；只要能把每片叶子都配准到同一个模板，就能把 3DGS 的冗余压缩成少量控制点。**

于是作者做了三件事：

1. 在 Gaussian centers 上计算 geodesic distance，从 root 出发找到每片叶子的 apex 和 petiole。
2. 选一个模板叶，只对它做 mesh 提取和纹理烘焙。
3. 用可微 MLS 把模板叶拟合到其它叶子，运行时再通过 vertex shader 在线变形。

换句话说，LeafFit 最核心的转变，是把“每片叶子都存一份几何”改成“每片叶子只存一组控制参数”。

### 为什么这个设计有效？

- geodesic segmentation 利用植物叶片从根到尖端的天然拓扑先验，训练成本低，也比泛化性不足的学习式分割更稳。
- 选择模板叶后，复杂的 mesh extraction 和 UV/texture transfer 只做一次，极大降低资产复杂度。
- MLS deformation 比刚性配准更适合叶片这类轻微弯曲、伸缩、扭转的薄片结构。
- 运行时把形变放到 vertex shader 里做，意味着存储成本极低，同时保留实时编辑能力。

### 战略权衡

- **优点**：很适合把 3DGS 植物捕捉结果落到游戏资产和 DCC 工具链。
- **局限**：强依赖叶片可分割性，对严重重叠叶群和复杂异形叶片仍会失败。

## Part III / Technical Deep Dive

### Pipeline

```text
plant 3DGS
-> pick root Gaussian
-> geodesic distance + apex grouping + petiole detection
-> segment leaf instances (optional manual drag/brush refinement)
-> choose one template leaf
-> farthest-point control sampling
-> differentiable MLS fits template to all leaves
-> BPA reconstructs thin template mesh + texture
-> store per-leaf control parameters only
-> runtime vertex shader deforms template into full plant asset
```

### 关键模块

#### 1. Geodesic leaf segmentation

作者利用 root 到 apex 的 geodesic distance，配合 apex grouping 和 petiole detection 来切分叶片。  
这套方法的关键点不是“分割更复杂”，而是利用了植物天然的叶脉拓扑结构，因此在没有训练数据时仍很稳。

#### 2. Template-driven MLS registration

作者对每片叶子采样控制点，再用 differentiable MLS 去优化模板叶到目标叶的配准。  
这个设计使得最终保存的不是重网格，而是模板和控制参数，从根上降低资产大小。

#### 3. Thin-surface mesh extraction

与隐式场方法不同，LeafFit 对模板叶采用 BPA 重建薄片表面，并显式处理正反面纹理。  
这比 Marching Cubes 路线更适合叶片这种极薄结构，也避免了常见的厚化问题。

### 关键实验信号

- 分割上，LeafFit 在 `Acc / mIoU / mF1 / PQ` 四项都领先，并且仍是 training-free 方法。
- 网格效率上，整株植物仅 `1.13 MB`、`2432` 顶点、`11980.11 FPS`，远优于 implicit baseline 转出的重网格。
- 形变上，`Corr / CD / HD` 都优于 `PCA / NR-ICP / BCPD`，其中 full model 的 `Corr 0.0823 / CD 0.0022 / HD 0.4669` 最好。
- 消融表明 `K=32` 控制点是很好的折中，去掉优化则误差显著增加，说明 MLS fitting 不是可有可无。

### 方法边界与风险

- 水平重叠和垂直重叠叶片会破坏 geodesic 路径，导致分割失败。
- 当前重点放在典型椭圆叶上，更复杂的叶形还需要更强先验。
- 所有叶片共享模板纹理，因此个体纹理差异被压缩掉了。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/Eurographics_2026/2026_LeafFit_Plant_Assets_Creation_from_3D_Gaussian_Splatting.pdf]]
