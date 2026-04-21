---
title: "F4Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting"
venue: arXiv
year: 2026
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - feed-forward
  - densification
  - budget-control
  - compact-3dgs
  - uncalibrated-reconstruction
  - multiscale-prediction
  - vggt
  - status/analyzed
core_operator: 把传统 3DGS 的 densification 改写成一个前向可预测的多尺度 allocation 问题，先预测 multi-scale Gaussian maps 与 densification score maps，再通过 threshold-controlled exclusive level selection 和 budget matching，在不重训的前提下根据目标 Gaussian budget 自适应分配 primitives。
primary_logic: |
  输入未标定的多视图 context images 与用户指定的 target Gaussian budget，
  先用 VGGT 风格的 geometry backbone 预测相机参数并编码几何特征，
  再由 Gaussian center / parameter heads 输出多尺度 Gaussian parameter maps 和 densification score maps，
  训练时从 novel-view rendering loss 反传得到 view-space gradient，把它压缩成 densification supervision，
  推理时根据阈值在不同尺度之间做互斥分配，并用 budget-matching algorithm 找到满足目标 budget 的阈值，
  最终得到紧凑但细节更集中的 feed-forward 3DGS 表示。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/arXiv_2026/2026_F4Splat_Feed_Forward_Predictive_Densification_for_Feed_Forward_3D_Gaussian_Splatting.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-10T20:36
updated: 2026-04-10T20:36
---

# F4Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://mlvlab.github.io/F4Splat/) | [arXiv 2603.21304](https://arxiv.org/abs/2603.21304)
> - **Summary**: F4Splat 的核心贡献不是又做了一个 feed-forward 3DGS，而是把传统 3DGS 中最难迁移到前向框架的一步 `adaptive density control`，改写成了可学习、可预算控制、可跨场景泛化的 `predictive densification`。它让“最终要放多少个 Gaussians、放在哪些区域”第一次成为 feed-forward 模型在推理时可直接控制的变量。
> - **Key Performance**:
>   - 在 `RE10K` 多视图未标定设置下，`F4Splat τ-` 以 `1142K` Gaussians 达到 `LPIPS 0.119 / SSIM 0.870 / PSNR 26.18`，优于 `AnySplat` 的 `PSNR 25.40`；而 `F4Splat τ+` 只用 `315K` Gaussians 仍有 `PSNR 25.85`。
>   - 在 `ACID` 两视图未标定设置下，`F4Splat τ+` 仅用 `52K` Gaussians 就达到 `PSNR 26.028`，已超过 `VicaSplat` 用 `131K` Gaussians 的 `24.548`；`τ-` 在同等 `131K` 预算下达到 `PSNR 26.282`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

F4Splat 解决的是 feed-forward 3DGS 里一个非常基础、但过去几乎都被默认绕开的核心问题:
**如果不再做 per-scene iterative optimization，那 3DGS 里最关键的 densification 机制要怎么保留下来？**

此前的 feed-forward 方法虽然快，但大多有两个结构性问题:

- `pixel-to-Gaussian` 路线通常一像素一个高斯，导致高斯数量和输入分辨率强绑定。
- `voxel-to-Gaussian` 路线虽然能改预算，但仍然是空间均匀分配，简单区域浪费、高频区域不够，且经常要为新预算重训。

所以 F4Splat 真正关心的，不只是“前向预测出一堆高斯”，而是:
**能否让前向模型像优化式 3DGS 一样，根据场景复杂度和多视图重叠，自适应决定哪里该 densify，哪里该省。**

### 核心能力定义

- **输入**: 未标定的 context images，加一个用户指定的 target Gaussian budget
- **输出**: 对应预算下的 3D Gaussian scene 与预测相机参数
- **擅长**: 在不重训的前提下做 budget-aware、spatially adaptive 的高斯分配
- **不擅长**: 动态场景、显式语义编辑、真正连续的任意精度控制

### 真正的挑战来源

- feed-forward 管线里没有 iterative densification，自然也失去了 ADC 的自适应分配能力。
- 不同视角会覆盖同一空间区域，若仍按像素或体素均匀出点，就会产生大量重复 Gaussians。
- 真正有用的 densification signal 应该和“质量收益”相关，但推理时又不能依赖 GT novel views 或反向优化。

### 边界条件

- 论文聚焦的是静态 3D scene reconstruction，而不是 4D 动态场景。
- Gaussian 数量控制范围受 multi-scale levels 决定，不是完全无界。
- 方法假设可以从输入图像中学到稳定的 densification score，而不是显式几何推理或语义推理。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

F4Splat 的整体哲学可以概括成一句话:
**把 densification 从“训练中边优化边长点”的过程，改写成“网络一次前向就预测出该在哪里长点”的问题。**

这一步很重要，因为它没有试图在 feed-forward 里硬模拟传统 3DGS 的完整优化流程，而是只迁移其中最关键的控制信号:

- 哪些区域复杂，值得更高密度
- 哪些区域多视角已经充分覆盖，不该重复出点
- 在给定预算下，哪些位置应该留在 coarse level，哪些位置应该升到 finer level

### The "Aha!" Moment

真正的 aha 在于:
**把 3DGS 里“需要 densify 的程度”提炼成一个可学习的 densification score，并让这个 score 来驱动多尺度高斯分配与预算匹配。**

作者不是凭启发式去算 score，而是从 rendering loss 的 `view-space positional gradient` 里提炼 supervision:

- 对某个 Gaussian，如果它对 novel-view rendering loss 的梯度大，说明这个区域当前表示不足。
- 于是这个梯度就可以作为 densification 的 teacher signal。
- 再训练网络直接从输入图像预测这个 score，推理时就不必再依赖 GT 和反向传播。

这样一来，`threshold τ` 就从普通超参数变成了一个真正可用的“细节/预算旋钮”。

### 为什么这个设计有效？

- **densification signal 与质量收益绑定**: score 不是纯视觉边缘启发式，而是来自 rendering error 对表示不足的反馈。
- **multi-scale exclusive allocation 很干净**: 每个区域只从一个尺度选高斯，避免不同 level 重复占预算。
- **novel-view supervision 避免记忆输入视角**: 模型学到的是可泛化的分配策略，而不是把 observed view 原样抄下来。
- **budget matching 可推理期直接使用**: 用户给预算，系统找阈值，不需要为了新预算重新训练模型。

### 战略权衡

- **优点**: 是当前 feed-forward 3DGS 里非常少见、真正把“Gaussian budget”变成 first-class control variable 的工作。
- **局限**: 它解决的是表示分配问题，不是时序建模、语义定位或编辑传播问题；另外多尺度设计也限制了可控范围和粒度。

## Part III / Technical Deep Dive

### Pipeline

```text
uncalibrated context images + target Gaussian budget
-> geometry backbone encodes image tokens and predicts camera parameters
-> Gaussian heads predict multi-scale Gaussian parameter maps + densification score maps
-> during training, render novel views and backprop view-space gradients
-> convert gradients into densification supervision
-> during inference, threshold densification scores to choose one scale per region
-> run budget matching to find the threshold for the requested Gaussian count
-> output compact budget-aware 3DGS
```

### 关键模块

#### 1. Geometry Backbone

作者沿用 `VGGT` 风格的几何主干:

- 先用 `DINOv2` 提取 patch tokens
- 再拼接 camera tokens 和 register tokens
- 通过 frame-wise / global self-attention 聚合多视图信息
- 最终同时输出 scene 几何表征与相机参数预测

这使得 F4Splat 不只是分配高斯数量，也在 uncalibrated setting 下联合学 pose。

#### 2. Multi-Scale Gaussian Prediction

模型预测多尺度的:

- Gaussian center maps
- Gaussian parameter maps
- densification score maps

随着 level 提升，空间分辨率翻倍。这样每个局部区域都可以有“粗尺度少量高斯”与“细尺度更多高斯”两种候选表达，后续由 score 决定到底用哪一层。

#### 3. Spatially Adaptive Gaussian Allocation

这是推理时真正起作用的分配模块。

基本规则是:

- 从最粗层开始
- 如果某区域 score 高于阈值 `τ`，就升级到更细层分配更多 Gaussians
- 不同尺度之间采用互斥 mask，确保不会重复选中同一片空间

最后得到的不是“每层都放一些”，而是一个在不同位置自适应切换分辨率的紧凑表示。

#### 4. Feed-Forward Predictive Densification

训练时作者用 novel-view rendering loss 反传，计算每个 Gaussian 的 `homodirectional view-space positional gradient`。

这个梯度被压成一个 log-scaled densification target:

- 梯度大，说明该区域目前表示不足，应该允许更多高斯
- 梯度小，说明区域已经表达充分，不需要继续加密

随后网络直接回归该 densification score，于是 ADC 风格的“该不该 densify”被转译成前向预测任务。

#### 5. Novel-View Training and Scene-Scale Regularization

两个训练细节很关键:

- **novel-view supervision**: 不只监督 context views，而是对齐到预测坐标系后监督 novel view，避免 trivial memorization。
- **scene-scale regularization**: 强制 Gaussian centers 的平均尺度稳定，否则在未标定设置里训练会直接崩掉。

### 关键实验信号

- 在 `RE10K` 和跨数据集泛化到 `ACID` 的实验里，F4Splat 在同等预算下优于 `AnySplat`，而在大幅压缩预算时依然维持很强质量。
- 两视图 `ACID` 结果说明它在非常稀疏输入下也有竞争力，且比未标定 baseline 更稳。
- ablation 很关键:
  - 用随机 allocation 或频率启发式替代 learned densification score，性能都会掉。
  - 去掉 level-wise Gaussian supervision，训练稳定性和最终质量都下降。
  - 去掉 scene-scale regularization 会直接训练失败。
- supplementary 还说明自适应 allocation 的代价并不大: 只带来约 `1.8%` VRAM 和 `10.1%` inference overhead。

### 对当前 idea 的启发

这篇对你当前 idea 的参考价值其实非常高，几乎就是“前向预测模块里的 densification operator”直接参考:

- 如果你要做 **semantic feed-forward 4D editing**，F4Splat 提供了一个非常明确的答案: `densification` 不必再依赖后期优化，而可以被网络直接预测。
- 其 `densification score` 未来可以扩展成 **语义不确定性 + 运动复杂度 + 多视图覆盖度** 的联合分数，用来决定哪里需要更高 4D Gaussian 密度。
- `budget matching + threshold control` 这套设计也很适合作为未来 4D 模型的 **可控推理接口**，例如按显存预算、交互时延或编辑区域大小动态调整表示密度。

### 实现约束

- 当前方法还是静态 3D，不包含时间维 densification，也没有 motion-aware allocation。
- 它的 score teacher 来自 novel-view rendering gradient，因此仍依赖训练阶段的可微渲染闭环。
- 预算控制是离散多尺度式的，不是完全连续的 density field control。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/arXiv_2026/2026_F4Splat_Feed_Forward_Predictive_Densification_for_Feed_Forward_3D_Gaussian_Splatting.pdf]]
