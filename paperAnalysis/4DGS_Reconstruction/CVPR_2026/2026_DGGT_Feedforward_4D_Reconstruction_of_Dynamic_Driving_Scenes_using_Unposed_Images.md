---
title: "DGGT: Feedforward 4D Reconstruction of Dynamic Driving Scenes using Unposed Images"
venue: CVPR
year: 2026
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - feed-forward
  - pose-free
  - driving-scene
  - dynamic-decomposition
  - motion-interpolation
  - diffusion-refinement
  - status/analyzed
core_operator: 把动态驾驶场景重建拆成“无位姿前向高斯重建 + 静动态分解 + 稀疏时刻间 3D 运动插值 + 图像空间扩散细化”，让相机、几何与运动在一次前向里同时落地。
primary_logic: |
  输入一组无位姿驾驶图像后，先用共享骨干同时预测每帧相机参数、像素对齐 Gaussian map、动态掩码和 lifespan，
  再把跨帧稳定部分聚合成静态高斯、把当前时刻运动部分作为动态高斯，并用 motion head 对动态对象做 3D 位移跟踪与插值，
  最后用 diffusion refinement 修补稀疏输入下的 ghosting 与空洞，得到可渲染、可编辑的动态驾驶场景。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2026/2026_DGGT_Feedforward_4D_Reconstruction_of_Dynamic_Driving_Scenes_using_Unposed_Images.pdf
category: 4DGS_Reconstruction
created: 2026-04-17T10:45
updated: 2026-04-17T10:45
---

# DGGT: Feedforward 4D Reconstruction of Dynamic Driving Scenes using Unposed Images

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://xiaomi-research.github.io/dggt/) | [arXiv 2512.03004](https://arxiv.org/abs/2512.03004)
> - **Summary**: DGGT 的关键不是简单把静态 feed-forward 3DGS 扩到动态场景，而是把驾驶场景里的“相机未知 + 物体在动 + 视角稀疏”三个难点合并建模，再用静动态分解和 3D 运动插值把动态对象从高斯层面稳定地接起来。
> - **Key Performance**:
>   - 在 Waymo 上，三视图设置达到 `27.41 PSNR / 0.846 SSIM / 3.47 D-RMSE`，重建耗时约 `0.39s`，同时优于 STORM 的质量指标。
>   - 零样本从 Waymo 泛化到 `nuScenes 25.31 PSNR`、`Argoverse2 26.34 PSNR`；3D 运动估计达到 `0.183m EPE3D / 85.42 Acc5 / 90.42 Acc10`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

这篇论文面向的是一个很实用的目标：
**能不能直接从无位姿驾驶图像里，在不到一秒的前向推理里重建可渲染、可跟踪、可编辑的动态 4D 场景？**

难点不只在于做 novel view synthesis，而在于三件事同时成立：

- 输入没有现成相机位姿，不能把 pose 当先验喂给模型。
- 场景里有大量运动目标，不能直接把跨帧高斯无脑叠加。
- 驾驶视频往往很长，方法必须能在输入视图数变多时仍然可扩展。

### 核心能力定义

- **输入**：多帧无位姿驾驶图像。
- **输出**：每帧相机参数、像素对齐 Gaussian map、动态掩码、3D 运动，以及可插值渲染的动态场景。
- **额外能力**：高斯级别的实例编辑，例如移除、平移或插入车辆/行人。
- **不擅长**：动态掩码失准或严重遮挡时，跨帧跟踪和动态插值会不稳。

### 边界条件

DGGT 的设计高度贴合驾驶数据分布。它最擅长的是大规模稀疏视角、带明显前景运动的道路场景；如果动态分解错了，后面的运动插值和编辑都会被系统性带偏。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

DGGT 的核心不是“每个时刻都单独预测一套 4D 场景”，而是先把问题拆开：

1. 先从每帧图像前向预测相机和高斯。
2. 用动态头把高斯分成静态部分和动态部分。
3. 静态部分跨帧聚合，动态部分只保留当前时刻，再用 motion head 在稀疏时间点之间补运动。

这相当于把动态驾驶重建从“直接学整条 4D 时空体”改写成“静态底座聚合 + 动态目标显式对齐”。

### The "Aha!" Moment

真正的 aha 在于：

**动态驾驶场景里，最不该做的事情是把所有帧的高斯直接并在一起；最该做的是先显式分离静态背景和运动目标，再让两者遵循不同的时间规则。**

具体来说：

- 静态部分来自所有帧的聚合，因为它们跨时间共享结构。
- 动态部分只取当前时刻，再靠 motion head 预测 3D 位移去连接相邻时刻。
- lifespan 参数进一步控制高斯在时间上的可见范围，避免“本该短暂出现的外观”被错误地长时间复用。

### 为什么这个设计有效？

- 它把动态物体对背景重建的污染压低了。
- 它把时间一致性问题从“重建所有帧”转成“对齐动态高斯轨迹”。
- 它让无位姿设置也能成立，因为 pose 本身也是网络输出，而不是硬前提。

### 战略权衡

- **优点**：一次前向即可同时给出相机、几何、动态掩码和 3D 运动，且视图数增加时退化不明显。
- **局限**：对动态掩码质量和遮挡区跟踪较敏感；扩散细化修的是渲染伪影，不是从根本上修正错误运动。

## Part III / Technical Deep Dive

### Pipeline

```text
unposed driving images
-> ViT backbone + DINO feature fusion
-> camera head / Gaussian head / lifespan head / dynamic head / motion head / sky head
-> per-frame pixel-aligned Gaussian maps + dynamic masks + camera poses
-> aggregate all static Gaussians across frames
-> keep current-frame dynamic Gaussians and align them with 3D motion interpolation
-> compose full scene at target time
-> diffusion refinement removes ghosting and disocclusion artifacts
```

### 关键机制

#### 1. Lifespan-aware Gaussian map

每个像素不只预测普通高斯属性，还预测 lifespan。它不是附属小技巧，而是时间可见性的开关：哪些高斯应该长期稳定存在，哪些只在局部时间片里起作用，模型会显式学出来。

#### 2. Dynamic decomposition as reconstruction interface

dynamic head 先输出动态概率图，再把每帧 Gaussian map 拆成静态和动态两部分。之后的场景合成不是“所有帧都叠进去”，而是“所有静态 + 当前动态 + sky Gaussians”。

#### 3. Motion head for 3D alignment

motion head 不是只预测速度标量，而是对查询像素预测 3D 位移，并通过邻域到邻域注意力去细化轨迹。这让它能更好处理驾驶场景里非线性运动和遮挡恢复。

#### 4. Diffusion refinement

插值后的 3DGS 渲染仍会有空洞、重影和遮挡边界伪影。DGGT 用单步 diffusion refinement 在图像空间做后处理，主要提升的是视觉完整性和编辑后的可用性。

### 关键实验信号

- Waymo 三视图插值任务上，DGGT 达到 `27.41 PSNR / 0.846 SSIM / 3.47 D-RMSE`，优于 STORM 的 `26.38 / 0.794 / 5.48`。
- 跨数据集零样本泛化时，Waymo 训练模型在 `nuScenes` 上达到 `25.31 PSNR`、在 `Argoverse2` 上达到 `26.34 PSNR`，说明 pose-free 设计没有把模型锁死在单一域。
- 3D 运动估计上，DGGT 达到 `0.183m EPE3D / 85.42 Acc5 / 90.42 Acc10 / 0.328 rad`，明显强于 STORM 的 `0.276 / 81.12 / 85.61 / 0.658`。
- 输入视图从 `4 / 8 / 16` 变化时，性能波动较小，说明它对长序列和稀疏采样都比较稳。

### 实现约束

论文实验围绕 Waymo、nuScenes、Argoverse2 这类驾驶数据展开；动态监督和 sky 建模都服务于道路场景，迁移到非驾驶域时需要重新判断这些先验是否还成立。

## Local Reading / 本地 PDF 引用

![[paperPDFs/4DGS_Reconstruction/CVPR_2026/2026_DGGT_Feedforward_4D_Reconstruction_of_Dynamic_Driving_Scenes_using_Unposed_Images.pdf]]
