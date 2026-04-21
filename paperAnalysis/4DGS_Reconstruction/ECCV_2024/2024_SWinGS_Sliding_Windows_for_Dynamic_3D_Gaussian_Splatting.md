---
title: "SWinGS: Sliding Windows for Dynamic 3D Gaussian Splatting"
venue: ECCV
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - sliding-windows
  - dynamic-scene
  - adaptive-windowing
  - temporal-consistency
  - self-supervision
  - status/analyzed
core_operator: 用 sliding-window 训练把长动态序列切成局部时窗，并允许每个窗口拥有局部 canonical 表示，再用自监督一致性把窗口之间重新对齐。
primary_logic: |
  输入任意长度动态序列，先按运动强度自适应选择窗口大小并为每个窗口单独训练动态 3D Gaussian 模型，
  再通过 temporally-local canonical representation 和 static/dynamic disentanglement 建模局部运动，
  最后用随机新视角上的自监督一致性微调连接不同窗口，得到可扩展到长序列的动态高斯重建。
pdf_ref: paperPDFs/4DGS_Reconstruction/ECCV_2024/2024_SWinGS_Sliding_Windows_for_Dynamic_3D_Gaussian_Splatting.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T15:06
updated: 2026-04-18T15:06
---

# SWinGS: Sliding Windows for Dynamic 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2312.13308) / [PDF](https://arxiv.org/pdf/2312.13308.pdf)
> - **Summary**: SWinGS 的价值在于它不再假设一个长视频可以被单个 canonical 动态模型优雅地吃下，而是把长时序拆成多个可控局部窗口，再用一致性把它们拼回去。这对长视频 4DGS 很实用。
> - **Key Performance**:
>   - 论文强调在任意长度动态场景上仍能保持高质量渲染。
>   - 同时保留了实时交互查看的优势。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

长序列动态 3DGS 常见的问题是：

- 单个 canonical 表示难以覆盖长时间剧烈几何变化
- 训练长视频会越来越重
- 一旦局部时段运动模式不同，统一 deformation 容易失效

SWinGS 的核心任务是让动态 3DGS 能扩展到更长、更复杂的序列。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

它真正新的地方不是单纯把序列切块，而是允许 **每个窗口拥有自己的局部 canonical 表示**。

这意味着论文承认一件很重要的事实：
长动态序列未必适合共享同一个时空模板。

### The "Aha!" Moment

真正的 aha 是：

**很多动态重建失败，不是因为局部运动不会建模，而是因为我们错误地要求整段视频共享同一套动态坐标系。**

SWinGS 通过 sliding windows 让 canonical representation 也变成局部对象：

- 局部窗口内更容易学习稳定形变
- 窗口间再用一致性损失对齐

这对你后面做 4DGS 编辑也很有启发，因为长时序编辑同样可能需要“局部编辑接口 + 全局拼接一致性”。

### 权衡与局限

- 优势：适合长序列、大变化动态场景
- 局限：窗口之间仍需额外对齐；不是一个最简洁的全局表示

## Part III / Technical Deep Dive

### Pipeline

```text
long dynamic sequence
-> adaptive sliding-window partition
-> train local dynamic 3DGS per window
-> local canonical representation and static/dynamic disentanglement
-> self-supervised consistency across windows
-> scalable dynamic scene reconstruction
```

### 关键技术点

#### 1. Sliding-Window Dynamic 3DGS

每个窗口单独训练动态高斯模型，降低长时序建模难度。

#### 2. Adaptive Sampling Strategy

窗口大小不是固定的，而是根据场景运动强度调节，平衡训练成本与视觉质量。

#### 3. Consistency Fine-Tuning

窗口切分后还需要重新拼回统一时序，所以作者用随机新视角上的自监督一致性进一步细化。

### 对你的价值

如果你之后做更长时序的 4DGS 编辑，SWinGS 最值得迁移的不是具体损失，而是：

- **局部时序建模**
- **跨窗口一致性恢复**

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/ECCV_2024/2024_SWinGS_Sliding_Windows_for_Dynamic_3D_Gaussian_Splatting.pdf]]
