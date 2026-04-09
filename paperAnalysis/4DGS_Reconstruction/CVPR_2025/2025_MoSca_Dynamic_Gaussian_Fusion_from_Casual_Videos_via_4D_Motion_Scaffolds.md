---
title: "MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds"
venue: CVPR
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - monocular-video
  - casual-video
  - motion-scaffold
  - foundation-model-priors
  - disentangled-deformation
  - bundle-adjustment
  - wild-reconstruction
  - status/analyzed
core_operator: 先用 foundation vision models 从 casual monocular video 中提升出紧凑平滑的 4D Motion Scaffold，再将场景几何与外观作为锚定其上的高斯进行全局融合，从而把困难的单目动态重建问题转成“运动骨架 + 高斯融合”的分解问题。
primary_logic: |
  输入野外单目视频后，先借助预训练视觉基础模型估计支撑动态的 4D Motion Scaffold，
  再将高斯锚定到该 scaffold 上并做全局融合，把几何与外观从 deformation field 中解耦出来，
  同时通过 bundle adjustment 联合求解相机位姿和焦距，最终得到可渲染的动态高斯场景。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2025/2025_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_Scaffolds.pdf
category: 4DGS_Reconstruction
created: 2026-04-09T17:08
updated: 2026-04-09T17:08
---

# MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.html) · [Project Page](https://jiahuilei.com/projects/mosca/) · [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.pdf) · [Code](https://github.com/JiahuiLei/MoSca)
> - **Summary**: MoSca 不是在多视角受控数据上做标准 4D 重建，而是直面“casual monocular video in the wild”这种更难、更弱约束的输入。它的关键思想是先从基础视觉模型中提取先验，提升出一个紧凑平滑的 4D Motion Scaffold，再把高斯锚定到这个 scaffold 上进行全局融合。这样做等于先把运动骨架理顺，再去重建几何和外观。
> - **Key Performance**:
>   - 项目页与 CVPR 页面都强调它能直接处理野外单目 casual video。
>   - 论文报告在动态渲染 benchmark 上达到 SOTA，并展示了真实视频场景中的实用性。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

MoSca 解决的是一个比标准 4D 新视角合成更难的问题：

**如何从 casual monocular video 中重建可渲染的动态 4D 场景。**

这里的难点不只是在单目，而在于：

- 视频随手拍，约束弱
- 相机内参与位姿不一定已知
- 动态物体、背景、遮挡都更复杂

很多传统 4D 重建方法默认多视角、同步、相对干净的数据。  
MoSca 明显更面向真实世界输入。

### 它的方法直觉

方法的核心直觉是：

1. **对于这种弱约束输入，直接优化高斯场太难**
2. **应该先把“运动/形变骨架”提出来**
3. **再把几何与外观作为锚定在这个骨架上的高斯做全局融合**

所以它不是直接从视频到高斯，而是先到 motion scaffold，再到 Gaussian fusion。

### 一句话能力画像

- **输入**：wild monocular video
- **先验来源**：foundation vision models
- **中间表示**：4D Motion Scaffold
- **场景表示**：锚定于 scaffold 的 dynamic Gaussians
- **附加优势**：还能联合求相机 pose 和 focal length

### 对你的研究最重要的启发

MoSca 对你最值得保留的启发是：

**在弱约束动态重建中，先把运动结构抽出来，往往比直接做完整 4DGS 优化更稳。**

这对以后做：

- 单目 4DGS
- in-the-wild 4D 编辑
- 结合 foundation model 的动态场景理解

都会很有帮助。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

它真正的新意不是“又提出一个 4DGS”，而是：

- 把 foundation model priors 系统地引入 4D 重建
- 用 Motion Scaffold 做运动层表示
- 再把几何和外观与 deformation 解耦

这意味着 MoSca 更像一个“先验增强的动态重建框架”，而不只是单纯的渲染表示。

### The "Aha!" Moment

真正的 aha moment 是：

**在 casual monocular video 这种严重欠约束的设置里，最先应该稳定下来的不是完整几何，而是运动结构。**

只要 motion scaffold 足够紧凑、平滑、可信：

- 高斯融合就更容易
- 时序一致性也更自然
- 视角外推会更稳

### 为什么这个思路值得长期记住

MoSca 很重要，因为它把 4DGS 从“受控多视角动态场景”往“真实视频输入”推进了一大步。  
对于你的知识库来说，它是连接以下两条线的关键节点：

- 4DGS reconstruction
- foundation-model-assisted dynamic scene understanding

### 它的边界

- 依赖基础视觉模型先验的质量
- 单目输入决定了问题本身仍然非常欠约束
- scaffold 若出错，后面的 Gaussian fusion 也会被系统性带偏

---

## Part III / Technical Deep Dive

### Pipeline

```text
casual monocular video
-> foundational vision priors
-> 4D Motion Scaffold
-> Gaussians anchored on scaffold
-> global Gaussian fusion
-> disentangled geometry/appearance vs deformation
-> bundle adjustment for pose + focal
-> dynamic scene reconstruction
```

### 关键模块

#### 1. 4D Motion Scaffold

这是整篇论文的灵魂。  
项目页明确把 MoSca 定义为：

- compact
- smooth
- encoding underlying motions/deformations

也就是说，它是一个运动骨架式中间表示，而不是直接用于最终渲染的完整场。

#### 2. Gaussian Fusion Anchored On Scaffold

高斯不是独立自由漂浮的，而是锚定在 Motion Scaffold 上，再做全局融合。  
这一步的好处是把：

- motion/deformation
- geometry/appearance

分开处理，从而提高优化稳定性。

#### 3. Built-in Camera Recovery

项目页还特别强调：

- focal length
- camera poses

都可以通过 bundle adjustment 联合求解，而不需要额外 pose estimation 工具。  
这对 casual video 非常关键，因为真实视频通常不会有干净标定。

### 关键实验信号

- 项目页展示覆盖 vertical short videos、robot videos、movie clips 等更野的输入。
- 论文强调对 real videos 的有效性，而不仅是 benchmark。
- 这说明 MoSca 的核心卖点不只是渲染指标，而是“输入条件更现实”。

### 对你可迁移的 operator

- **motion scaffold before Gaussian fusion**
- **foundation-model priors for underconstrained 4D reconstruction**
- **joint camera and scene optimization in monocular settings**

如果你以后打算做从单目视频到可编辑 4DGS 的链路，MoSca 会非常关键。

---

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/CVPR_2025/2025_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_Scaffolds.pdf]]
