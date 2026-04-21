---
title: "DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction"
venue: arXiv
year: 2026
tags:
  - 3DGS_Reconstruction
  - gaussian-splatting
  - 3dgs
  - feed-forward
  - noise-robust-reconstruction
  - geometry-appearance-decoupling
  - multi-view-consistency
  - boundary-guided-refinement
  - status/analyzed
core_operator: 在 MVSplat 风格前向 3DGS 上，把 Gaussian 参数预测拆成 geometry branch 和 appearance branch 两个轻量分支，并利用几何分支导出的边界强度与置信度构造 CBC 门控残差，只在边界低置信区域修正外观分支，直接从 noisy multi-view 输入恢复 clean 3DGS。
primary_logic: |
  输入带相机参数的 noisy multi-view images，
  先用共享 2D encoder 和 geometry-aligned plane-sweep aggregation 提取跨视图 3D 特征，
  再分别用几何分支预测位置、旋转、尺度和透明度，用外观分支预测颜色与 SH 系数，
  随后根据几何分支的 disparity 边界强度与置信度构造 CBC gating，只在边界低置信区域做外观残差修正，
  最终通过可微 Gaussian splatting 渲染 clean views，并仅用 clean 2D renderings 监督训练整个网络。
pdf_ref: paperPDFs/3DGS_Reconstruction/arXiv_2026/2026_DenoiseSplat_Feed_Forward_Gaussian_Splatting_for_Noisy_3D_Scene_Reconstruction.pdf
category: 3DGS_Reconstruction
created: 2026-04-18T19:00
updated: 2026-04-18T19:00
---

# DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2603.09291](https://arxiv.org/abs/2603.09291)
> - **Summary**: DenoiseSplat 研究的是一个很现实但过去常被忽略的问题: feed-forward 3DGS 在干净输入上很强，但真实多视图数据常带噪声、压缩和低照度伪影。作者没有把 denoising 放在 3D 之前，而是把去噪直接内化进 3D Gaussian 表示学习本身。
> - **Key Performance**:
>   - 在 noisy RE10K 上，DenoiseSplat 达到 `25.05 PSNR / 0.814 SSIM / 0.260 LPIPS`，优于 `Denoise-Then-MVSplat` 的 `24.77 / 0.788 / 0.272` 和直接输入噪声的 `MVSplat-Noisy`。
>   - 编码端推理时间仅 `60.74 ms`，明显低于两阶段基线的 `86.90 ms`，而峰值显存仅从 `1.03 GiB` 增到 `1.07 GiB`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

这篇论文处理的是一个非常具体的 reconstruction 痛点：

**当多视图输入本身有噪声时，feed-forward 3DGS 是否还能稳定恢复干净几何和纹理，而不是先做逐视图 2D 去噪再重建？**

传统做法通常有两种：

- 直接把 noisy 图像喂给原始 MVSplat，这会很快损坏几何和纹理一致性；
- 先逐帧做 2D denoise，再送进 3D 重建，但这往往带来过平滑和视图间不一致。

DenoiseSplat 的主张是：噪声问题应该在 3D 表示层内部解决，而不是作为独立预处理模块外挂。

### 核心能力定义

- **输入**：带相机参数的 noisy multi-view images。
- **输出**：clean 3D Gaussian scene 及其 novel-view renderings。
- **擅长**：Gaussian、Poisson、speckle、salt-and-pepper 四类噪声下的 feed-forward 静态场景重建。
- **不擅长**：真实相机复杂噪声、运动模糊、压缩伪影以及动态场景。

### 真正的挑战来源

- 2D 去噪会移除高频细节，而这些细节恰恰常是多视图匹配和 3D 恢复所必需的。
- 噪声不仅污染外观，也会间接损坏几何推断和跨视图融合。
- 在边界和遮挡区域，几何估计稍有不稳，渲染出来的外观伪影就会非常显眼。

### 边界条件

- 论文数据主要是基于 RE10K 的合成噪声 benchmark。
- 相机参数是已知的，因此这不是无位姿的鲁棒重建方法。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

DenoiseSplat 的哲学非常直接：

**不要把 denoising 和 reconstruction 串成两个模块，而要把它们放进同一个前向 3DGS 网络里联合学习。**

但作者并没有简单地“让网络自己学去噪”，而是进一步把问题拆成两层：

- 几何应当主要依赖更稳定、更抗噪的结构 cue；
- 外观可以吸收剩余噪声，但需要和几何分开建模。

### The "Aha!" Moment

真正的 aha 是：

**噪声恢复里最容易被污染的是 appearance，而真正应该尽量稳定下来的则是 geometry，所以两者不能共用一个 Gaussian head。**

于是作者把原始单头预测改成：

- `geometry branch` 负责中心、旋转、尺度、不透明度；
- `appearance branch` 负责颜色和 SH 系数。

这相当于承认 noisy reconstruction 的核心矛盾不是“所有参数都一起更稳”，而是“几何和外观对噪声的敏感性本来就不同，必须解耦”。

### 为什么这个设计有效？

- geometry branch 可以优先利用跨视图稳定结构，而不被颜色噪声过度干扰。
- appearance branch 独立建模残余噪声和颜色波动，减少其反向污染几何。
- `CBC` 进一步用几何边界强度和置信度去门控 appearance 残差，只在真正危险的边界低置信区域做修正，避免全局过修。

### 战略权衡

- **优点**：完全保留 feed-forward 3DGS 的高效推理形式，同时显著提升噪声鲁棒性。
- **局限**：对真实世界复杂退化的泛化仍未充分验证，而且当前主要针对静态场景。

## Part III / Technical Deep Dive

### Pipeline

```text
noisy multi-view images + cameras
-> shared 2D feature encoder
-> plane-sweep / geometry-aligned 3D aggregation
-> geometry branch predicts Gaussian structure
-> appearance branch predicts SH/color
-> CBC uses geometry boundary + confidence to gate appearance residuals
-> Gaussian splatting renderer outputs clean views
-> train with clean 2D rendering supervision only
```

### 关键模块

#### 1. Scene-consistent multi-noise benchmark

作者在 RE10K 上构建了 scene-consistent 噪声数据：同一 scene 的所有视图共享同一种噪声类型和强度。  
这比逐视图随机噪声更贴近真实采集条件，也更能暴露跨视图一致性问题。

#### 2. Dual-branch Gaussian Head

这是整篇最核心的结构设计。  
单头预测在 noisy setting 下容易让外观噪声干扰几何恢复，而双分支能显式降低这种耦合。

#### 3. CBC

`Cross-Branch Boundary-Guided Appearance Correction` 并不是简单再加一个 denoiser。  
它从几何分支导出 disparity 边界强度和置信度，并只在“边界强且低置信”的区域去修 appearance residual，因此更像一个精确的 boundary repair module。

### 关键实验信号

- 总体表 2 中，DenoiseSplat 比两阶段 `IDF + MVSplat` 在三项指标都更好，说明“先 2D 去噪再 3D 重建”不是最优解。
- 在 Gaussian 噪声从 `σ=0.05` 增加到 `0.15` 的消融里，DenoiseSplat 的退化曲线最平滑，尤其在强噪声下 `LPIPS` 和 `SSIM` 优势更明显。
- 对 Poisson、speckle、salt-and-pepper 的不同强度设置，模型也保持了相对稳定的表现，说明不是只对单一种噪声过拟合。
- 论文强调 novel views 上的提升更能说明问题，因为跨视图一致性受噪声影响最大，而 DenoiseSplat 在这一点上优于两阶段方法。

### 方法边界

- 当前噪声仍是合成噪声，真实 camera noise、压缩伪影和 motion blur 没有系统覆盖。
- 方法仍建立在 MVSplat 这一类前向多视图骨干上，并没有改变其对输入视角和相机信息的基本依赖。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Reconstruction/arXiv_2026/2026_DenoiseSplat_Feed_Forward_Gaussian_Splatting_for_Noisy_3D_Scene_Reconstruction.pdf]]
