---
title: "GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes"
venue: arXiv
year: 2026
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - monocular-video
  - probabilistic-modeling
  - gaussian-process
  - variational-inference
  - uncertainty-quantification
  - temporal-extrapolation
  - motion-prior
  - status/analyzed
core_operator: 把 4DGS 里的 primitive deformation 改写成定义在 canonical 位置与时间上的 variational Gaussian Process 后验，用时空核学习 motion prior，再通过 GP-GS 双阶段优化把不确定性、稀疏区域补全和未来时刻外推一起带进动态重建。
primary_logic: |
  输入 monocular video 与 canonical 4D Gaussian primitives 后，先依据渲染贡献筛出高置信高斯轨迹训练多输出 variational Gaussian Processes，
  再用空间 Matérn 核与时间 periodic 核建模 deformation 的时空相关性，并周期性缓存 GP 后验均值/方差作为 motion prior 与 uncertainty 来源，
  最后在 4DGS 优化时用 GP guidance regularize 难观测区域，从而同时提升重建、估计不确定性并支持未来时刻外推。
pdf_ref: paperPDFs/4DGS_Reconstruction/arXiv_2026/2026_GP_4DGS_Probabilistic_4D_Gaussian_Splatting_from_Monocular_Video_via_Variational_Gaussian_Processes.pdf
category: 4DGS_Reconstruction
created: 2026-04-21T16:00
updated: 2026-04-21T16:00
---

# GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2604.02915](https://arxiv.org/abs/2604.02915)
> - **Summary**: GP-4DGS 的关键不是再发明一种 deterministic deformation，而是把 4DGS 的运动建模显式概率化，让 deformation 本身携带后验均值与方差，从而同时获得更稳的 sparse-view 动态重建、可解释的不确定性估计和超出训练时刻的未来运动外推。
> - **Key Performance**:
>   - 在 `DyCheck` 全 7 场景上达到 `17.38 mPSNR / 0.65 mSSIM / 0.37 mLPIPS`，优于 `SoM` 的 `17.09 / 0.65 / 0.39`；在 `Challenging subset` 上达到 `15.02 / 0.46 / 0.51`，优于 `SoM` 的 `14.56 / 0.46 / 0.53`。
>   - 在未来运动外推上，周期性场景保留最后 `5 / 15` 帧测试时达到 `17.62 / 16.65 PSNR`，明显高于 naive linear extrapolation 的 `11.55 / 8.11`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？

GP-4DGS 瞄准的是一个现有 4DGS 普遍回避的问题：
**现在的 4DGS 几乎都把 motion 当作 deterministic optimization 问题，因此既不会告诉你哪里不可靠，也不会在训练时间之外自然外推。**

这会带来三个直接后果：

- 遮挡或稀疏观测区域只能靠手工先验硬压。
- 模型无法给出 principled uncertainty。
- 对未来时刻或训练帧之外的 motion prediction 几乎没有能力。

### 核心能力定义

- **输入**：单目动态视频和可优化的 4D Gaussian scene。
- **输出**：带 probabilistic motion prior 的 4DGS 重建结果，以及与 primitive motion 对应的不确定性估计。
- **能做**：dynamic reconstruction、uncertainty quantification、future motion extrapolation。
- **不主打**：跨场景 feed-forward inference；它仍然是 per-scene optimization 路线。

### 真正的挑战来源

- 标准 4DGS deformation 先验是固定的，难以根据 observation density 自适应变化。
- 直接对所有 primitives 做 GP 后验推断，计算量会爆炸。
- 时空相关性并不是各向同性的，空间和时间的 kernel 结构不能简单混写。

### 边界条件

GP-4DGS 依赖 canonical-space deformation 框架，核心改造点是 motion prior，而不是渲染器本体。它更像是在 4DGS 上加一个 probabilistic motion layer，而不是替换整条 4DGS pipeline。

## Part II / High-Dimensional Insight

### 方法真正新的地方

这篇工作真正新的地方，是把 4DGS 里最常被写成“一个 MLP 或一个显式函数”的 deformation，改写成：

**定义在 canonical position × time 上的 GP posterior。**

于是 motion 不再只是一个点估计，而是一个既有 mean、也有 variance 的函数分布。

### The "Aha!" Moment

真正的 aha 是：

**动态高斯的运动先验，不一定要靠手工 rigidity 或固定 parametric deformation 硬塞进去，也可以从“高置信观测到的 primitive trajectories”里学出来，并再反过来约束难观测区域。**

这个设计的因果链很清楚：

- 先从高置信 primitives 提取更可信的 motion samples。
- 用 variational GP 学到 motion posterior。
- 再把 posterior mean 当作 guidance，posterior variance 当作 uncertainty。
- 于是 4DGS 在 sparse / occluded regions 的优化不再完全盲目。

### 为什么这个设计有效？

- GP 天然提供 uncertainty 和 extrapolation，这两点不是额外外挂模块，而是 probabilistic formulation 的直接产物。
- 通过把空间和时间分开建核，模型能同时保住 canonical geometry 上的局部平滑性和时间维度上的周期结构。
- 通过 confidence-weighted sampling 先只信可靠 observations，motion prior 的质量会明显高于对所有 noisy primitives 一视同仁。

### 战略权衡

- **优点**：让 4DGS 首次有了 principled uncertainty 和真正像样的 future motion prediction。
- **代价**：仍是 per-scene optimization；而且 GP 只是 motion layer，不会自动解决 appearance 建模与大场景泛化。

## Part III / Technical Deep Dive

### Pipeline

```text
monocular video + canonical 4D Gaussians
-> compute response/confidence of primitives from rendering contribution
-> select confident primitive trajectories
-> train multi-output variational GPs on canonical position + time
-> use spatial Matérn kernels + periodic temporal kernels
-> infer posterior mean / variance for all primitives
-> use posterior mean as GP guidance during GS optimization
-> render uncertainty maps and query future timesteps for extrapolation
```

### 关键机制

#### 1. Spatio-temporal GP kernel

作者没有把 `(x, y, z, t)` 粗暴地塞进一个 isotropic kernel，而是显式分成：

- 空间 Matérn kernel：负责 canonical geometry 上的局部相关性
- 时间 periodic kernel：负责周期性 motion pattern 与未来时刻外推

这使模型更像在学习“哪些 primitives 在空间上应该一起动，哪些 temporal pattern 会重复出现”。

#### 2. Variational GP with inducing points

因为 primitive 数量通常是万级，exact GP 不现实，所以论文用 inducing points 做 variational approximation，把复杂度降到 `O(NM^2 + M^3)`。另外它还用时间序列特征来初始化 inducing points，而不是简单 random pick。

#### 3. GP-GS dual optimization

最关键的不是“先训好 GP 再用一次”，而是一个循环：

- 阶段一：从当前 4DGS 里挑 confident trajectories 训练 GP
- 阶段二：用 GP posterior mean regularize 4DGS 的 deformation

这形成了一个自强化闭环：越稳定的 primitives 越能提供好 prior，越好的 prior 又越能修 sparse / occluded regions。

### 关键实验信号

- 在 `DyCheck` 上，GP-4DGS 在全 7 场景、SoM 5 场景和 reduced-overlap `Challenging subset` 上都优于 Gaussian Marbles / SoM，尤其 challenging subset 的优势更明显，说明概率化 motion prior 对 sparse observation 很有帮助。
- 在 future motion extrapolation 上，保留最后 `5 / 15` 帧测试时，GP-4DGS 在 periodic scenes 上远强于 linear extrapolation，说明时间 kernel 真正学到了周期结构，而不只是 interpolation。
- 在 uncertainty quantification 上，`AUSE-MSE` 在 `Top-20 / Top-40 / All frames` 三种设置下都优于 `UA-4DGS`，说明 uncertainty 不是“看起来合理”，而是和真实 error 对齐得更好。

### 你最值得记住的点

对你后续 `4DGS editing + uncertainty-aware propagation` 方向，这篇最值钱的不是它的重建分数，而是它证明了：

- uncertainty 可以直接挂在 primitive motion 上，而不一定只做 image-space score
- future motion prediction 可以从 probabilistic motion prior 自然长出来
- “高置信轨迹蒸馏出 prior，再反过来管 sparse region” 这个闭环很适合迁移到 edit propagation 场景

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/arXiv_2026/2026_GP_4DGS_Probabilistic_4D_Gaussian_Splatting_from_Monocular_Video_via_Variational_Gaussian_Processes.pdf]]
