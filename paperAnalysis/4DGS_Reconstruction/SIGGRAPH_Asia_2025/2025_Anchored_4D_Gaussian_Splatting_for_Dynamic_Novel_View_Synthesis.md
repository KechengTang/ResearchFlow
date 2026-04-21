---
title: "Anchored 4D Gaussian Splatting for Dynamic Novel View Synthesis"
venue: SIGGRAPH Asia
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - anchor-based
  - model-compression
  - dynamic-anchor-control
  - anchor-stabilization
  - neural-gaussians
  - storage-efficient
  - status/analyzed
core_operator: 用 4D anchor points 作为动态高斯的上层组织单元，由 anchor latent feature 经 MLP 解码局部 neural Gaussians 的属性，再通过 dynamic anchor control 把 anchor 增量集中在动态欠重建区域，并通过 anchor stabilization 冻结静态区域的 anchor，兼顾渲染质量与存储效率。
primary_logic: |
  先在 4D 时空中初始化 anchors，并让每个 anchor 生成一组局部 neural Gaussians，
  再在给定时间切片得到对应的 3D Gaussian 用于渲染，
  同时根据动态区域的可见性与时间梯度触发新的 dynamic anchor growth，
  并对低速度 static anchors 进行稳定化冻结，防止静态区域持续过生长，
  最终实现高质量、低存储的 dynamic novel view synthesis。
pdf_ref: paperPDFs/4DGS_Reconstruction/SIGGRAPH_Asia_2025/2025_Anchored_4D_Gaussian_Splatting_for_Dynamic_Novel_View_Synthesis.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T21:12
updated: 2026-04-18T21:12
---

# Anchored 4D Gaussian Splatting for Dynamic Novel View Synthesis

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [DOI](https://doi.org/10.1145/3757377.3763898)
> - **Metadata Note**: 当前 PDF 标注为 `SIGGRAPH Asia 2025` 正式论文，同时首页带有 “Corrected VoR published on January 2, 2026” 说明。这不影响现有 `analysis_log.csv` 中的 `SIGGRAPH_Asia 2025` 归档。
> - **Summary**: Anchored 4DGS 的核心不是再做一种更大的 4D Gaussian，而是用 4D anchor 去组织和约束局部 neural Gaussians。这样既能保留 4D Gaussian 的表达力，又能避免 direct 4D expansion 带来的巨大存储开销。
> - **Key Performance**:
>   - 在 N3DV 上达到 `32.01 PSNR / 0.948 SSIM / 0.136 LPIPS / 126 MB`。
>   - 在 Technicolor 上达到 `33.59 PSNR / 0.920 SSIM / 0.169 LPIPS / 96 MB`。
>   - 相比 Realtime4DGS，存储从 `6882 MB` 降到 `126 MB`；相比 4d-rotor-gs，从 `437 MB` 降到 `126 MB`，约为 `98%` 和 `70%` 压缩。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

Anchored 4DGS 解决的是：

**4D Gaussian 虽然表达力强，但直接把高斯扩到 4D 会带来很高的存储开销；而基于 MLP 的 time-varying Gaussians 又常有质量瓶颈。**

所以问题不是单纯的“质量 vs 速度”，而是：

- 怎么保留 4D Gaussian 的时空表达力
- 怎么避免 per-frame / per-Gaussian 属性膨胀
- 怎么把模型容量更精准地放到真正动态的区域

### 它的方法直觉

论文的直觉是：

**动态场景里相邻 Gaussian 往往共享相近的运动、旋转和尺度属性，因此没必要让所有 transient Gaussians 都完全独立存在。**

因此作者引入 anchor-based hierarchy：

- 上层是 4D anchors
- 下层是由 anchor feature 解码出的 local neural Gaussians

这样动态高斯的属性不再是逐点独立存储，而是由 anchor latent feature 组织起来。

### 一句话能力画像

- **输入**：动态场景多视角视频
- **核心表示**：4D anchors + neural Gaussians
- **关键增量机制**：dynamic anchor control
- **关键压缩机制**：anchor stabilization
- **目标**：高质量 dynamic NVS with much lower storage

### 对我当前方向的价值

这篇对你当前方向非常有价值，因为它打到一个很实际的问题：

**4DGS 想落地，不能只追求质量，还必须解决表示冗余和存储膨胀。**

对你的几条路线来说：

- 对 `4DGS_Reconstruction`：它提供了一个很强的 representation compression / organization 方案。
- 对 `4DGS_Editing`：anchor 作为更高层语义单位，天然比逐高斯编辑更适合作为编辑控制柄。
- 对 `feed-forward Gaussians`：anchor-latent -> local Gaussian 的生成方式非常接近未来可前馈预测的层级 Gaussian 表示。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

主要创新点有三层：

- **4D anchor-based representation**：不是直接存所有 4D Gaussians，而是通过 anchor feature 解码局部 Gaussian 属性。
- **dynamic anchor control**：专门针对动态欠重建区域生成更多 anchor，解决原始 Scaffold-GS 增长策略在 4D 上对 temporally sparse regions 不敏感的问题。
- **anchor stabilization**：将低速度、近似静态的 anchors 冻结，防止静态区域反复长新 anchor。

这套设计本质上是在做“时空容量分配”。

### The "Aha!" Moment

最值得记住的 aha 是：

**4DGS 的表达力问题和存储问题，不一定要靠更强压缩算法解决，也可以靠更合理的层级组织去解决。**

也就是说，先用 anchor 把局部高斯组织起来，再决定哪些区域继续增长、哪些区域冻结，  
这比让所有高斯无差别地 densify 要聪明得多。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**：关系很近。anchor 是天然的高层编辑单元，未来无论是局部时空编辑还是 motion-aware editing，都可能比逐高斯更稳。
- **与 3DGS_Editing**：Scaffold / anchor 思路本来就和静态场景 editing 很兼容，这篇只是把它推进到 4D。
- **与 feed-forward Gaussians**：关系非常直接。未来完全可以让网络前向预测 anchor latent，再解码出局部动态高斯。

### 局限、风险、可迁移点

- **局限**：anchor 与 Gaussian 的绑定关系本身会引入额外优化耦合，论文也提到这可能带来训练不稳定性。
- **风险**：dynamic anchor control 与 stabilization 的阈值选择会影响动态/静态区域的划分。
- **可迁移点**：
  - hierarchical anchor-based 4D Gaussian representation
  - capacity allocation toward dynamic regions
  - freezing static support units to stop redundancy growth

---

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic scene video
-> initialize 4D anchors in spatiotemporal space
-> decode local neural Gaussians from anchor features
-> slice 4D representation at target timestamp
-> render 3D Gaussians for that time
-> dynamic anchor control grows anchors in under-reconstructed dynamic regions
-> anchor stabilization freezes static anchors
-> efficient dynamic novel view synthesis
```

### 关键模块

#### 1. 4D Anchors

每个 anchor 存一个 latent feature，并负责生成一组局部 neural Gaussians。  
这样邻近高斯的属性通过共享 anchor feature 被隐式耦合起来。

#### 2. Dynamic Anchor Control

原始 Scaffold-GS 的 anchor growth 直接搬到 4D 不够，因为很多动态高斯只在少数时间点可见。  
作者因此引入针对动态区域的可见性与时间梯度策略，专门在 under-reconstructed dynamic regions 增长新 anchor。

#### 3. Anchor Stabilization

静态区域的 anchor 往往在几乎所有时间都可见，如果不管，会持续累计增长。  
作者通过 anchor velocity 识别近似静态 anchor，并将其冻结，阻止冗余扩张。

### 关键实验信号

- N3DV 和 Technicolor 两个 benchmark 上都领先或持平最强基线，同时大幅降低存储。
- 对 Realtime4DGS 和 4d-rotor-gs 的存储压缩非常显著，这证明 anchor organization 不是纸面小优化，而是 representation-level 改进。
- Ablation 里：
  - 只用动态阈值策略提升有限
  - 再加 dynamic anchor control 有明显增益
  - 再加 anchor stabilization 达到最好  
  说明“动态区域加容量、静态区域减冗余”这两个动作必须配合。

### 对当前研究最可迁移的 operator

- **anchor-latent to local dynamic Gaussians**
- **dynamic-region-aware capacity growth**
- **static-region freezing for efficient 4DGS**

如果你后面要做可扩展的 4DGS 库或者更可控的 4D 编辑表示，这篇非常值得长期保留。

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/SIGGRAPH_Asia_2025/2025_Anchored_4D_Gaussian_Splatting_for_Dynamic_Novel_View_Synthesis.pdf]]
