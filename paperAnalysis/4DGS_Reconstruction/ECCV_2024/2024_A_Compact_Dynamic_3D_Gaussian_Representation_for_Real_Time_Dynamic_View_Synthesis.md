---
title: "A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis"
venue: ECCV
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - compact-model
  - dynamic-scene
  - real-time
  - monocular-video
  - status/analyzed
core_operator: 只让位置和旋转随时间变化，并用少量参数近似这些时变分量，同时固定尺度、颜色和透明度，以更紧凑的 3DGS 形式表示动态场景。
primary_logic: |
  输入单目或多视图动态观测，先建立一组共享外观属性的 3D Gaussians，
  再把位置与旋转建模成时间函数并用少量参数近似，
  从而在降低内存占用和弱化每时刻多视图需求的同时，保持高质量动态新视角合成。
pdf_ref: paperPDFs/4DGS_Reconstruction/ECCV_2024/2024_A_Compact_Dynamic_3D_Gaussian_Representation_for_Real_Time_Dynamic_View_Synthesis.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T14:52
updated: 2026-04-18T14:52
---

# A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2311.12897) / [PDF](https://arxiv.org/pdf/2311.12897.pdf)
> - **Summary**: 这篇论文的价值在于它把动态 3DGS 的时间自由度压缩得很干净。不是所有参数都该随时间漂移，作者只让位置和旋转随时间变化，这让动态 3DGS 变得更省内存、更轻量，也更适合作为后续编辑或前馈建模的底座。
> - **Key Performance**:
>   - 文中在单 GPU 上报告了约 118 FPS 的渲染速度。
>   - 同时保持了与更重的动态方法相当甚至更好的渲染质量。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

动态 3DGS 的一个直观难点是：如果每个时间都要更新所有高斯参数，内存和数据需求都会迅速升高。

这会带来两个问题：

- 表示很重
- 每个时间点都更依赖足够多的观测视角

这篇论文的目标很明确：
**把动态高斯表示压缩到最小必要变化量。**

## Part II / High-Dimensional Insight

### 方法真正新在哪里

作者的判断是：

- 很多动态场景的尺度、颜色、透明度未必需要频繁变化
- 真正主要随时间改变的往往是位置和旋转

所以方法只对这两个量做时间函数建模，而其余属性保持不变。

### The "Aha!" Moment

真正的 aha 是：

**动态 3DGS 的困难不一定来自“时间太复杂”，而是来自“我们让太多不该变化的参数也跟着时间变了”。**

这篇论文通过缩小时间作用域，获得了：

- 更小的模型
- 更低的内存需求
- 更好的实时性

对你后续做 4DGS 编辑很有参考价值，因为这对应一个核心设计问题：
**哪些量应该被编辑，哪些量应该尽量稳定地被继承。**

### 权衡与局限

- 优势：紧凑、高速、对观测需求更友好
- 局限：如果外观和透明度本身强时变，这种固定假设会受限

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic observations
-> shared 3D Gaussians with static appearance attributes
-> time functions for positions and rotations
-> compact dynamic representation
-> real-time dynamic view synthesis
```

### 关键技术点

#### 1. Selective Temporal Parameterization

论文没有把所有属性都做成时变，而是把时间变化集中在位置和旋转上，这就是紧凑性的来源。

#### 2. Relaxing Multi-View Assumptions

由于表示更省、更稳，它对每时刻多视图的依赖也被弱化，所以单目设置下也能更好工作。

### 关键信号

- 118 FPS 是非常强的工程信号。
- 论文强调既省内存又保持质量，这比单纯追速度更值得关注。

### 对你的价值

这篇论文很适合放进 4DGS 编辑的支撑层阅读，因为：

- 它给出了“最小时间变化参数集”的思路
- 对后续做 editable dynamic representation 很有启发

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/ECCV_2024/2024_A_Compact_Dynamic_3D_Gaussian_Representation_for_Real_Time_Dynamic_View_Synthesis.pdf]]
