---
title: "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"
venue: CVPR
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - dynamic-scene-rendering
  - neural-voxels
  - hexplane-inspired
  - deformation-decoder
  - explicit-representation
  - real-time-rendering
  - status/analyzed
core_operator: 用 3D Gaussians 承载显式空间表示，再用 HexPlane 启发的 4D neural voxels 与轻量 MLP deformation decoder 在任意时间戳生成动态高斯，从而把动态场景重建转成显式高斯加时空特征场的组合问题。
primary_logic: |
  输入一组基础 3D Gaussians 及时间戳，
  先通过空间-时间结构编码器从 4D neural voxels 中提取 Gaussian 特征，
  再经多头 Gaussian deformation decoder 预测该时间点的高斯形变与属性，
  最终得到可实时渲染的动态 3D Gaussians 表示。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_4D_Gaussian_Splatting_for_Real_Time_Dynamic_Scene_Rendering.pdf
category: 4DGS_Reconstruction
created: 2026-04-09T17:08
updated: 2026-04-09T17:08
---

# 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html) · [Project Page](https://guanjunwu.github.io/4dgs/) · [arXiv 2310.08528](https://arxiv.org/abs/2310.08528) · [Code](https://github.com/hustvl/4DGaussians)
> - **Summary**: 4D-GS 是当前很多 4DGS 方法的基础母体之一。它不是简单对每一帧单独训练 3DGS，而是把动态场景表示成“基础 3D Gaussians + 4D neural voxels + deformation decoder”的整体系统。这样做的意义是同时保留显式高斯渲染的速度优势，以及时空特征场对复杂运动的建模能力。
> - **Key Performance**:
>   - 项目页与论文都报告在 RTX 3090 上可达到高分辨率实时渲染；项目页给出 `82 FPS @ 800x800`。
>   - 论文强调可以在约 30 分钟内完成高分辨率动态场景学习。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

4D-GS 针对的是动态场景表示里经典的两难：

- 想要高质量复杂动态
- 又想要高效率和实时渲染

在它之前，很多动态表示方法要么：

- 质量不错但推理慢
- 要么速度快但难以建模复杂形变

4D-GS 的核心目标就是把 3DGS 的显式高效渲染优势迁移到动态场景里。

### 它的方法直觉

它的方法直觉非常经典也非常有影响力：

1. **基础显式几何仍然交给 3D Gaussians 表示**
2. **动态变化不直接编码进每一帧的独立高斯，而是交给时空特征场与 deformation decoder**
3. **这样既避免逐帧单独训练，也避免完全隐式表示的高昂代价**

### 一句话能力画像

- **基础表示**：3D Gaussians
- **时空特征场**：4D neural voxels
- **动态建模**：lightweight deformation MLP
- **优势**：实时、显式、训练和存储效率高

### 对你的研究最重要的启发

4D-GS 对你最大的意义，不只是它是早期基线，而是它定义了一种范式：

**显式高斯负责渲染，隐式/半隐式时空特征负责驱动动态。**

后面很多 4DGS 论文，包括编辑方法，本质上都是在这个范式上继续加模块。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

它真正新的地方在于没有把动态 3DGS 做成“每帧一个 3DGS”的简单扩展，而是提出了一个整体 4D 表示：

- 显式空间粒子：3D Gaussians
- 时空上下文：4D neural voxels
- 时间解码器：deformation decoder

这让它既不像传统 NeRF 那么完全隐式，也不像逐帧 3DGS 那样参数爆炸。

### The "Aha!" Moment

真正的 aha moment 是：

**动态场景不一定要靠为每个时间点维护一整套独立显式表示来建模，只要有一个共享的基础高斯集合，再配一个时间条件驱动的形变机制，就已经能表达很多动态。**

这个思路后来几乎成为 4DGS 领域的标准起点。

### 为什么这个思路值得长期记住

因为它很像一个通用骨架：

- 如果你做 reconstruction，它是动态建模器
- 如果你做 editing，它是可被编辑的基础场
- 如果你做 language / feature extension，它是可挂接语义或特征的几何载体

也就是说，4D-GS 的价值不只是一个 benchmark 方法，而是一个**平台型表示**。

### 它的边界

- 形变由轻量 decoder 预测，极复杂或拓扑变化剧烈的场景可能仍有挑战
- 其低秩/分解式时空编码虽然高效，但表达能力并非无限
- 更复杂的动态语义控制不是它原始设计的重点

---

## Part III / Technical Deep Dive

### Pipeline

```text
base 3D Gaussians + timestamp t
-> spatial-temporal structure encoder on 4D neural voxels
-> Gaussian features
-> multi-head Gaussian deformation decoder
-> deformed Gaussians at time t
-> real-time rendering
```

### 关键模块

#### 1. 3D Gaussians As Base Particles

论文并没有抛弃 3DGS，而是保留其作为显式空间载体。  
这使得渲染仍然享受高斯 splatting 的高速优势。

#### 2. 4D Neural Voxels

这部分是动态能力的主要来源。  
论文与项目页都强调，它借鉴了 HexPlane 风格的分解式编码，以更高效地构建时空特征。

#### 3. Deformation Decoder

给定高斯中心与时间戳，模型从时空结构编码器提取特征，再通过多头 deformation decoder 预测该时刻的高斯状态。  
这说明 4D-GS 的动态不是“显式存全部帧”，而是“按时间查询并解码”。

### 关键实验信号

- 项目页给出 `82 FPS @ 800x800` 的实时性能。
- 论文强调训练效率和存储效率都较强。
- 其重要贡献还在于给后续 4DGS 社区提供了一个广泛使用的代码和表示起点。

### 对你可迁移的 operator

- **base-Gaussian + deformation-field factorization**
- **hexplane-inspired spatio-temporal encoding**
- **time-conditioned deformation decoding**

你后面看很多 4DGS 编辑或语义扩展论文时，都可以回头问一句：它到底是在 4D-GS 这个骨架上的哪一层做修改。

---

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/CVPR_2024/2024_4D_Gaussian_Splatting_for_Real_Time_Dynamic_Scene_Rendering.pdf]]
