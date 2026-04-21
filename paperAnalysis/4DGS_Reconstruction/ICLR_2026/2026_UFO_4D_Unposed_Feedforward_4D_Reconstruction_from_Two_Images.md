---
title: "UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images"
venue: ICLR
year: 2026
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - feed-forward
  - unposed-images
  - two-view
  - self-supervision
  - scene-flow
  - pose-estimation
  - 4d-interpolation
  - status/analyzed
core_operator: 用一套从两张无位姿图像直接预测的 Dynamic 3D Gaussians 同时承载 geometry、motion 与 pose，再借助可微 4D rasterization 把图像、点图和 scene flow 监督耦合到同一组高斯上。
primary_logic: |
  输入两张无位姿图像和相机内参后，先用共享 ViT 编码器和多头读出中心、属性、速度与相对位姿，
  再把两张图各像素对应的动态高斯统一到第一视角 canonical space，并在连续时间上用 4D rasterization 渲染图像、点图和运动，
  最后联合光度自监督与稀疏几何/运动标签训练，让同一组高斯同时服务于重建、位姿估计和 4D 时空插值。
pdf_ref: paperPDFs/4DGS_Reconstruction/ICLR_2026/2026_UFO_4D_Unposed_Feedforward_4D_Reconstruction_from_Two_Images.pdf
category: 4DGS_Reconstruction
created: 2026-04-17T10:45
updated: 2026-04-17T10:45
---

# UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://ufo-4d.github.io/) | [arXiv 2602.24290](https://arxiv.org/abs/2602.24290)
> - **Summary**: UFO-4D 把“从两张无位姿图像同时恢复几何、运动和相机”的问题统一压进一套 Dynamic 3D Gaussian 表示里，不再为 depth、scene flow、pose 各自训练割裂的表征，而是让所有监督都落在同一组时空高斯上。
> - **Key Performance**:
>   - 几何估计在 `Stereo4D` 上达到 `0.659 EPE / 0.106 AbsRel / 88.38 δ<1.25`，在 `Bonn` 上达到 `0.162 EPE / 0.064 AbsRel / 96.04 δ<1.25`。
>   - 运动估计在 `Stereo4D` 上达到 `0.049 / 0.050` 的前后向 `EPE3D`，在 `KITTI` 上达到 `0.137 EPE3D`；位姿估计在 `Stereo4D` 上达到 `0.0101 ATE`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

UFO-4D 针对的是一个很难但很核心的问题：
**能不能只看两张无位姿图像，就前向恢复一个显式的 4D 场景表示，并从中一致地读出几何、运动和相机位姿？**

传统做法要么依赖测试时优化，要么把深度、光流、scene flow、pose 拆成多个独立任务。这样的问题是：

- 计算慢，很难落地到 feed-forward。
- 各任务之间信息不能共享，监督彼此割裂。
- 在 4D 数据稀缺的情况下，很难稳定训练。

### 核心能力定义

- **输入**：两张无位姿图像和相机内参。
- **输出**：第一视角 canonical space 下的 dynamic 3D Gaussians，以及两帧相对位姿。
- **可直接支持的任务**：depth、point map、scene flow、optical flow、novel-view/time interpolation。
- **边界**：核心设定是双帧输入，连续时间插值默认线性运动，因此它更擅长局部 4D 重建，而不是长时程动态建模。

### 真正的挑战来源

- 双帧、无位姿、动态场景，本身就是强欠约束问题。
- 真实 4D 数据标注稀疏且带噪，单靠监督信号很难学稳。
- 如果 geometry 和 motion 不共享底层表示，彼此很难相互校正。

### 边界条件

UFO-4D 假设相机内参可用，并以第一张图像作为 canonical 坐标系。它的主要优势在两帧设定和统一监督，而不是显式处理长序列记忆。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

UFO-4D 的想法很直接也很重要：

1. 用一组 dynamic 3D Gaussians 统一承载 appearance、geometry 和 motion。
2. 让图像重建、点图、scene flow 全都通过同一个可微 rasterizer 回传到这组高斯。
3. 用 rendering 得到密集自监督，再用 sparse label 补足几何和运动方向。

这让“多任务联合”不再只是多头输出，而是共享同一个显式 4D 物理载体。

### The "Aha!" Moment

真正的 aha 在于：

**如果 geometry、motion 和 pose 最终都来自同一组动态高斯，那么监督其中一个模态，本质上也在约束另外两个模态。**

这带来两个直接好处：

- 图像重建损失提供了密集自监督，不再完全依赖稀疏或噪声标注。
- 因为 geometry 和 motion 共用同一组 primitives，错误很难只藏在单一模态里，任务之间会互相正则化。

### 为什么这个设计有效？

- 双帧场景里最稀缺的是密集可靠监督，rendering 恰好补了这一块。
- 显式 Dynamic 3D Gaussians 比点图更适合直接做 novel-view 和 novel-time 渲染。
- 直接前向输出 relative pose，避免了额外的 PnP 或测试时回归步骤。

### 战略权衡

- **优点**：统一、显式、两帧即可工作，既能做 4D reconstruction，又能做 4D interpolation。
- **局限**：主要建模双帧局部时域；中间时刻插值基于线性运动近似，复杂长程动态仍有上限。

## Part III / Technical Deep Dive

### Pipeline

```text
two unposed images + camera intrinsics
-> shared ViT encoder/decoder with intrinsic token and pose token
-> center / attributes / velocity / pose heads
-> predict one dynamic Gaussian per pixel from both images
-> transform all Gaussians into first-camera canonical space
-> differentiable 4D rasterization renders image / point / motion at arbitrary time and view
-> combine self-supervised photometric loss with sparse geometry and motion supervision
```

### 关键机制

#### 1. Dynamic 3D Gaussian per pixel

每个像素都预测一颗动态高斯，包含中心、速度、旋转、尺度、球谐颜色和 opacity。第二帧高斯会通过 `μ + v` 和反向速度统一到第一帧时刻，这样两张图的高斯自然落到同一个 canonical space。

#### 2. Differentiable 4D rasterization

UFO-4D 不只渲染 RGB，还渲染 point map 和 scene flow。连续时间 `t + Δt` 下，高斯中心沿速度线性平移，于是同一个 rasterizer 可以同时提供外观、几何和运动监督。

#### 3. Semi-supervision by coupling signals

论文最重要的训练 insight 是：rendering 提供密集 photometric 自监督，稀疏点图/运动标签提供方向性约束，而它们都约束到同一组高斯上，因此能在 4D 数据稀缺时仍然把模型拉稳。

### 关键实验信号

- **几何估计**：`Stereo4D` 上达到 `0.659 EPE / 0.106 AbsRel / 88.38 δ<1.25`；`Bonn` 上达到 `0.162 / 0.064 / 96.04`；`KITTI` 上达到 `2.568 / 0.090 / 88.92`。
- **运动估计**：`Stereo4D` 前向/后向 `EPE3D` 分别为 `0.049 / 0.050`，对应 `83.06 / 82.67` 的 `δ0.05`；`KITTI` 上 `EPE3D 0.137`，相对 prior 有超过 3 倍误差下降。
- **位姿估计**：`Stereo4D / Bonn / Sintel` 上分别达到 `0.0101 / 0.0020 / 0.0122` 的 ATE，说明双帧无位姿设定下依然能给出很稳的相对相机。
- **额外能力**：除了重建，论文还展示了 image、depth 和 motion 在任意新视角、新时间的 4D 插值能力，这正是显式动态高斯表示带来的附加价值。

### 实现约束

训练使用 `Stereo4D + PointOdyssey + Virtual KITTI 2` 的混合数据，120k steps，约 3 天、4 张 A100。它的强项是双帧统一 4D 感知；如果后续要做更长时间范围的 4DGS 系统，还需要再补长期记忆或多帧聚合机制。

## Local Reading / 本地 PDF 引用

![[paperPDFs/4DGS_Reconstruction/ICLR_2026/2026_UFO_4D_Unposed_Feedforward_4D_Reconstruction_from_Two_Images.pdf]]
