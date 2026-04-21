---
title: "An Exploration with Entropy Constrained 3D Gaussians for 2D Video Compression"
venue: ICLR
year: 2025
tags:
  - 3DGS_Reconstruction
  - cross-domain-reference
  - gaussian-splatting
  - video-compression
  - entropy-constrained
  - orthographic-projection
  - deformable-gaussians
  - stream-decoding
  - status/analyzed
core_operator: 用 Toast-like Sliding Window 正交投影把视频的 xy+t 体积改写成可被 3DGS 渲染的显式高斯表示，再结合 time-aware Gaussian generation、光流引导形变和后续 entropy coding，形成可流式解码的视频压缩器 GSVC。
primary_logic: |
  先把视频的空间与时间轴重解释为 3DGS 的 xy+t 坐标，并用 TSW 只渲染时间滑窗内的高斯，
  再基于 HAC 风格压缩框架生成 time-aware deformable Gaussians，
  用 optical flow guidance 和率失真目标联合训练视频重建质量与码率，
  最后对 anchor、属性和网络参数做后压缩与 GPU entropy coding 得到可流式解码的 bitstream。
pdf_ref: paperPDFs/3DGS_Reconstruction/ICLR_2025/2025_An_Exploration_with_Entropy_Constrained_3D_Gaussians_for_2D_Video_Compression.pdf
category: 3DGS_Reconstruction
created: 2026-04-18T16:49
updated: 2026-04-18T16:49
---

# An Exploration with Entropy Constrained 3D Gaussians for 2D Video Compression

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICLR 2025 Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/02cf78fa6cabf269f8d8380355c5f096-Abstract-Conference.html)
> - **Summary**: 这篇论文不是在做传统 3D scene reconstruction，而是在测试一个更激进的想法: 3DGS 能不能直接成为 2D 视频压缩器。核心技巧是把视频看成 `xy+t` 的体积，再用 TSW 正交投影把“时间邻域”变成可局部渲染的高斯滑窗。
> - **Key Performance**:
>   - 论文报告 GSVC 在 UVG 上相对 NeRV 拿到更好的 RD 表现，并且帧重建速度高约 `30%~40%`。
>   - GSVC 支持 stream decoding；附录里典型 `600` 帧视频训练时间约 `3h26m`，对比 NeRV 约 `6h6m`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？
神经视频压缩里一条常见路线是 INR: 直接用神经网络从时间戳回归视频帧。  
这类方法有两个老问题：

- 重建速度不够快，难以真正实用
- 必须先把整个网络或整个表示解码出来，天然不适合 stream decoding

论文提出的问题是：

**3D Gaussian Splatting 这种显式、可局部访问、可部分渲染的表示，能不能被改造为一个实际可用的视频压缩器？**

### 核心能力定义

- **输入**: 单个视频序列
- **输出**: 可解码视频的 Gaussian-based bitstream
- **强项**: 显式表示、局部渲染、随机解码、较高重建 FPS
- **弱项**: 仍然是 per-video 训练式编码，不是通用预训练编码器

### 真正的挑战来源

- 普通 3DGS 假设输入来自真实 3D 场景的多视角观测，这对一般视频并不成立
- 视频需要压缩的是时间冗余，不是真实几何
- 如果沿用原 3DGS 视角模型，会把不满足 3D 结构假设的视频拟合得很差

### 边界条件

- 它更像 `Gaussian-based video representation/compression`，而不是标准意义的 `3DGS_Reconstruction`
- 依赖对单个视频做训练，不适合实时编码
- 强调的是解码与重建效率，而不是几何真实性

## Part II / High-Dimensional Insight

### 方法的整体设计哲学
这篇论文最关键的范式变化是：

**不要执着于“视频必须对应真实 3D 场景”，而是把视频本身当作一个可以被高斯显式表示的时空信号。**

作者把二维平面和时间轴组合成 `x-y-t` 三维空间，然后重新定义相机模型，用一个沿时间轴滑动的窗口去渲染当前帧相关的高斯。这样，3DGS 的显式渲染优势被迁移到了视频压缩。

### The "Aha!" Moment

真正的 aha 是：

**TSW orthographic projection 不要求真实几何，只要求相邻帧相似。**

也就是说，论文并不是把视频硬解释成一个 3D 场景，而是把 3DGS 里的“相机-高斯可见性裁剪”改造成“按时间滑窗访问局部高斯”的机制。  
这一步直接带来了两个收益：

- 只解码 / 渲染当前时间窗需要的高斯，天然支持 stream decoding
- 显式高斯渲染本身很快，重建速度比典型 INR 更有优势

### 为什么这个设计有效？

- TSW 把时间邻域裁剪成局部可渲染窗口，降低了每帧参与渲染的高斯数量
- time-aware Gaussian generation 让高斯生成与时间条件显式绑定，而不是静态参数硬复用
- optical-flow-guided deformation 为动态对象提供运动先验，减少纯数据驱动拟合的难度
- 后压缩把 anchor 坐标、网络参数和属性一起纳入 bitstream，避免训练期 entropy 估计和实际文件大小脱节

### 对我当前方向的价值
虽然它被放在 `3DGS_Reconstruction`，但对你真正有价值的点其实更偏表示工程：

- 对 `4DGS_Editing`，它提供了“编辑后动态内容如何按时间窗快速回放与传输”的思路
- 对 `3DGS_Editing`，它提醒 3DGS 不只是 3D scene primitive，也可以是通用显式信号容器
- 对 feed-forward Gaussians，这篇论文尤其重要，因为前馈模型若要真正上线，stream decoding、partial rendering、post-compression 都是绕不开的问题

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: TSW 的时间滑窗思想可以视为一种低成本播放与回放层，适合承接编辑后的动态 Gaussian 序列
- **与 3DGS_Editing**: 关系中等。它不直接解决编辑问题，但给出了“显式高斯如何变成可部署媒体格式”的方向
- **与 feed-forward Gaussians**: 关系最强。若未来模型直接预测一段动态高斯序列，GSVC 这类后端能提供可压缩、可流式解码的落地路径

### 战略权衡

- 优点: 重新定义了 3DGS 的使用边界，让它从 3D 场景表示扩展到视频压缩
- 代价: 放弃了真实 3D 语义，更多是在做高效显式时空信号建模

## Part III / Technical Deep Dive

### Pipeline

```text
video
-> reinterpret as xy+t volume
-> TSW orthographic projection and temporal culling
-> time-aware Gaussian generation
-> optical-flow-guided deformable Gaussians
-> joint reconstruction + entropy-constrained training
-> post-compression of anchors / attributes / network params
-> stream-decodable Gaussian video bitstream
```

### 关键模块

#### 1. Toast-like Sliding Window (TSW) Projection
这是全文最核心的 operator。  
在渲染时间戳 `t` 的帧时，只取 `z` 坐标落在 `[t-h, t+h]` 内的高斯参与渲染。这样做相当于把普通相机 near/far clipping 改造成时间域局部访问。

#### 2. Time-Aware Gaussian Generation
作者在生成高斯属性的 MLP 中加入 FiLM，使时间编码与高斯相对时间位置信息共同调制属性生成。  
这一步是让静态 3DGS 压缩骨架适配视频任务的关键。

#### 3. Optical Flow Guided Deformation
GSVC 不是只靠 TSW 去吃掉所有动态性，而是额外用光流引导形变高斯去建模动态对象。  
这使它对移动前景的压缩和重建更稳。

#### 4. Post-Training Compression And GPU Entropy Codec
作者指出只在训练期估计 entropy 还不够，因为 anchor 坐标与网络参数在视频任务中占比变高。  
所以他们额外用 G-PCC 压 anchor 坐标、8 bit 量化网络参数，并实现 GPU 上的 ANS codec 来减少 CPU/GPU 来回通信。

### 关键实验信号

- 主表里 GSVC 在 7 个 UVG 视频上都明显优于原始 3DGS / Scaffold-GS / HAC 及其 TSW 变体，说明 “TSW + deformable compression pipeline” 不是小修小补，而是决定性改造
- 论文正文明确给出 GSVC 相对 NeRV 有更好的 RD 表现，并在 `MS-SSIM`、`LPIPS` 上优势更明显
- 实用性指标上，GSVC 比 NeRV 快约 `30%~40%`，比 VCT 快 `40x+`
- GSVC 的 stream decoding 曲线优于 NeRV/HNeRV，因为它不需要先把整个表示完整解码出来

### 少量关键数字

- `Beauty`: `30.25 PSNR / 1.107 MB / 83.37 FPS`
- `Bosphorus`: `33.19 PSNR / 2.050 MB / 67.03 FPS`
- `Yacht`: `28.61 PSNR / 3.7015 MB / 64.00 FPS`
- 典型 `600` 帧视频训练时间: `12.4k s (3h26m)`，NeRV 约 `22k s (6h6m)`

### 局限、风险、可迁移点

- **局限**: 仍然依赖 per-video 训练编码，离真正通用视频 codec 还有距离
- **风险**: TSW 的滑窗宽度 `h` 对视频动态特性敏感；动态过强时大窗口会引入训练不稳和表征混叠
- **可迁移点**: `xy+t` 重解释、时间滑窗局部渲染、stream-decodable Gaussian bitstream、GPU 侧 entropy codec 都很值得迁移到动态 Gaussian 系统

### 实现约束

- 数据集为 UVG 前 7 个视频，分辨率 `1920x1080`
- 训练总步数约 `40000`，使用 VideoFlow 预估光流
- 使用 RTX 3090；重建速度只统计 rendering，不包含可并行重叠的 entropy decoding

## Local Reading / PDF 参考
![[paperPDFs/3DGS_Reconstruction/ICLR_2025/2025_An_Exploration_with_Entropy_Constrained_3D_Gaussians_for_2D_Video_Compression.pdf]]
