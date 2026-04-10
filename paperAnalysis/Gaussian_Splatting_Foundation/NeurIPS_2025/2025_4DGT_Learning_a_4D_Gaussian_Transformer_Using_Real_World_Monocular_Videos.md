---
title: "4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos"
venue: NeurIPS
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 4dgs
  - feed-forward
  - 4d-reconstruction
  - dynamic-scene
  - transformer
  - density-control
  - motion-understanding
  - status/analyzed
core_operator: 用 transformer 直接从 posed monocular video 预测带 lifespan、velocity 与 angular velocity 的 4D Gaussian，并通过两阶段 density control 与 multi-level spatiotemporal attention 把时空采样做大而不把计算炸掉。
primary_logic: |
  输入带位姿的单目视频，
  先把图像 patch、Plücker ray、时间戳和 DINOv2 特征一起编码进 transformer，
  再直接解码为包含空间参数与时间参数的 dynamic Gaussians，
  训练时先低分辨率粗训，再按 opacity activation histogram 剪枝并在空间与时间上 densify，
  最后借助 multi-level 时空注意力高效预测长时序一致的 4DGS。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/NeurIPS_2025/2025_4DGT_Learning_a_4D_Gaussian_Transformer_Using_Real_World_Monocular_Videos.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-10T18:24
updated: 2026-04-10T18:24
---

# 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://4dgt.github.io) | [arXiv 2506.08015](https://arxiv.org/abs/2506.08015)
> - **Summary**: 4DGT 把动态场景重建从“逐场景优化数小时”推进到“带位姿单目视频直接前向预测 4DGS”，核心在于把动态高斯表示、时空密度控制和高效注意力一起设计成可扩展的 feed-forward 4D backbone。
> - **Key Performance**:
>   - 训练策略带来约 `80%` Gaussian 减少、`16x` 更高训练采样率和约 `5x` 渲染提速。
>   - 在 ADT 上纯前向结果达到 `PSNR 28.31 / LPIPS 0.243 / RMSE 0.93`，推理约 `25 ms/frame`；只微调 `10s` 后可达 `PSNR 31.98`，仍比 SoM 类优化法快约两个到三个数量级。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

4DGT 解决的是一个非常核心、也非常难的基础问题：

**如何只用真实世界的带位姿单目视频，训练出一个可以秒级输出动态 4DGS 的 feed-forward 重建模型？**

这件事难，不只是因为 4D 表示比 3D 更复杂，更因为单目视频天生存在严重的时空歧义：

- 同一像素变化可能来自相机运动，也可能来自物体运动
- 真实世界几乎没有足够规模的多视角动态真值可用于监督
- 若直接把长视频做成 pixel-aligned 4D Gaussian，token 数和 Gaussian 数都会迅速爆炸

### 核心能力定义

- **输入**：带相机位姿的单目视频
- **输出**：可实时渲染的动态 4D Gaussian scene
- **额外能力**：在没有显式监督的情况下，学出动态区域区分、光流与运动分割等涌现性质
- **不解决**：无位姿输入、开放域任意设备零校准重建

### 真正的挑战来源

- 单目动态视频缺少稳定的 4D 监督
- 4DGS 的时空采样一旦做密，训练和渲染都会被冗余高斯拖垮
- vanilla self-attention 对长时序 token 的二次复杂度太高

### 边界条件

- 论文默认可获得可靠位姿，来自 SLAM、ARKit 或离线标定
- 泛化主要建立在“与训练设备/标定质量相近”的真实视频分布上

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

4DGT 的关键不是把 3D LRM 简单加一个时间轴，而是把 **动态性** 明确写进 Gaussian 表示本身：

- 静态内容通过长 lifespan、近零速度来表达
- 动态内容通过短 lifespan、速度与角速度来表达

于是静态与动态不再需要两套完全不同的建模逻辑，而是落在同一个 4D Gaussian 参数化下。

### The "Aha!" Moment

最关键的 insight 是：

**动态场景的可扩展 feed-forward 重建，不能靠“从一开始就高分辨率密采样 + 全量注意力”硬撑，而要先粗建 scaffold，再根据激活模式选择性保留真正有用的 4D Gaussians。**

作者因此引入两步联动：

- 第一阶段在粗分辨率上训练 pixel-aligned 4DGS，拿到一个可用的时空脚手架
- 第二阶段根据 patch 内 opacity activation histogram 做 pruning，只保留高激活通道，再同时在空间和时间上 densify

这一步的因果关系很直接：

- 若不剪枝，更多时空采样只会把冗余高斯和显存一起放大
- 剪枝后再 densify，新增预算就能用在真正活跃、真正承载动态的区域上
- 再配合 multi-level spatiotemporal attention，长窗口建模才在计算上变得可行

### 为什么这个设计有效？

- `lifespan + velocity + angular velocity` 让一个高斯自己决定“活多久、怎么动”，比按帧独立预测更像真正的 4D 表示
- `Lv / Lω / Ll` 这些正则把网络从“所有点都短命乱动”的坏局部最优中拉出来
- 深度与法线 expert guidance 提供了单目训练时最缺的几何锚点
- multi-level attention 用空间分辨率换时间范围，把时空混合建模成本控制在可训练范围内

### 战略权衡

- **优点**：速度极快、可以处理长视频、适合作为 feed-forward 4D backbone
- **局限**：依赖位姿与设备分布；极端新视角与未见设备上的质量仍会下降

## Part III / Technical Deep Dive

### Pipeline

```text
posed monocular video
-> patchify RGB + Plucker rays + timestamps + DINOv2 features
-> 4D Gaussian Transformer fusion
-> decode 2DGS geometry + temporal center + lifespan + velocity + angular velocity
-> stage-1 coarse training on sparse space-time samples
-> histogram-based pruning + space-time densification
-> multi-level spatiotemporal attention for efficient long-window fusion
-> real-time renderable 4DGS
```

### 关键模块

#### 1. 动态 Gaussian 表示

每个 Gaussian 除了空间位置、尺度、朝向、透明度外，还额外携带：

- temporal center
- life-span
- velocity
- angular velocity

这使得同一套参数同时覆盖静态背景与短暂出现的动态物体。

#### 2. 两阶段时空密度控制

第二阶段使用固定的 `Rs = 2`、`Rt = 4`、`S = 10`、`p = 14`，在剪掉冗余高斯后继续提高时空采样密度。结果是最终 Gaussian 总量只剩原始密采样方案的约 `20%`，但时空覆盖反而更强。

#### 3. 单目训练的几何约束

作者只使用真实世界单目视频训练，但通过 `DepthAnythingV2 / StableNormal / UniDepth` 一类 expert 提供深度和法线伪监督，再叠加 lifespan 与 motion regularization，降低时空歧义。

### 关键实验信号

- 与 L4GM、MonST3R、单目 expert 和 SoM 相比，4DGT 在真实视频上显著更稳，且运行时间从优化法的 `60,000 ms/frame` 降到 `25 ms/frame`
- 在 ADT 上，`Ours tune10s` 达到 `PSNR 31.98 / LPIPS 0.220 / RMSE 0.85`，已经超过优化法参考上界
- 动态前景评估中，`Ours (masked)` 相比静态 LRM 在 ADT/DyCheck 上明显更强，说明显式动态表示确实学到了运动区域
- 运动分割消融中，模型可达 `mIoU 81.2`，甚至优于 `MegaSaM 77.4`，而且速度快约 `200x`

### 对当前 idea 的启发

4DGT 对你当前方向的价值非常直接：

- 它是少见真正把 **feed-forward 4D backbone** 做扎实的工作
- `lifespan / velocity / angular velocity` 这组动态参数化非常适合作为后续 4D 编辑里的时间骨架
- “先粗训、再基于激活模式剪枝并 densify”的训练范式，很适合作为你想做的前向 4D 编辑或前向语义预测模块的效率参考

### 实现约束

- 训练窗口为 `128` 帧，第一阶段输入 `16` 帧，第二阶段增加到 `64` 帧
- 训练使用 `64 x H100`，第一阶段约 `9` 天，第二阶段约 `6` 天
- 推理速度在单张 `80GB A100` 上报告

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/NeurIPS_2025/2025_4DGT_Learning_a_4D_Gaussian_Transformer_Using_Real_World_Monocular_Videos.pdf]]
