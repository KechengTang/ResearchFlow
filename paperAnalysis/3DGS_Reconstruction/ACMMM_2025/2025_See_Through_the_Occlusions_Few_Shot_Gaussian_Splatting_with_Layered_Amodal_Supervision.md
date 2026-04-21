---
title: "See Through the Occlusions: Few-Shot Gaussian Splatting with Layered Amodal Supervision"
venue: ACMMM
year: 2025
tags:
  - 3DGS_Reconstruction
  - gaussian-splatting
  - few-shot
  - sparse-view
  - occlusion-aware
  - amodal-supervision
  - STO-GS
  - diffusion-inpainting
  - layered-opacity
  - status/analyzed
core_operator: 借用 4DGS 的时间建模思想，把 time 重解释为 occlusion depth，用扩散式 amodal inpainting 生成分层 amodal views，再通过 STO-MLP 按 occlusion layer 调制 Gaussian opacity，从而让 few-shot 3DGS 能显式学习被前景遮住的隐藏结构。
primary_logic: |
  先基于稀疏输入图像生成多层 amodal views，
  再把 occlusion layer 当成类似时间的可学习维度，为每个 Gaussian 预测与层相关的 opacity 调制，
  接着用两阶段训练和 layer-aware sampling 在 modal / amodal 监督之间逐步过渡，
  最终把原本不可见的背景与被遮挡结构纳入 few-shot 3DGS 的学习过程。
pdf_ref: paperPDFs/3DGS_Reconstruction/ACMMM_2025/2025_See_Through_the_Occlusions_Few_Shot_Gaussian_Splatting_with_Layered_Amodal_Supervision.pdf
category: 3DGS_Reconstruction
created: 2026-04-18T21:12
updated: 2026-04-18T21:12
---

# See Through the Occlusions: Few-Shot Gaussian Splatting with Layered Amodal Supervision

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [DOI](https://doi.org/10.1145/3746027.3755801)
> - **Summary**: 这篇工作很有意思，它把 4DGS 的时间维借来做静态 few-shot 3DGS 的遮挡建模。具体做法是把 occlusion depth 当成类时间维度，再用 diffusion inpainting 生成 layered amodal views，当作显式监督去训练高斯在不同遮挡层上的 opacity 调制。
> - **Key Performance**:
>   - 在 Mip-NeRF360 12-view 设定下，相比 CoR-GS 大约提升 `0.48 dB` 和 `0.51 dB` PSNR。
>   - 在 LLFF 3-view `1/4` 下达到 `20.02 PSNR / 0.680 SSIM / 0.264 LPIPS`，`1/8` 下达到 `20.78 / 0.741 / 0.190`。
>   - 在 Shiny 数据集上达到 `19.23 / 0.576 / 0.309` 和 `20.51 / 0.683 / 0.210`，说明不仅对强遮挡有效，对反射场景也有帮助。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

STO-GS 解决的是：

**few-shot 3DGS 在强遮挡和稀疏视角条件下，如何恢复那些从来没被直接看到的隐藏区域。**

标准 few-shot 3DGS 常见问题是：

- central region 还原得还可以
- 被前景挡住的背景结构学不到
- 视角一偏，隐藏区域直接碎掉

其根因不是渲染器不够强，而是：

**hidden content 根本没有显式 supervision。**

### 它的方法直觉

这篇的直觉非常巧妙：

**把 4DGS 里 “时间变化” 的建模框架，重解释成 “遮挡深度层” 的建模框架。**

于是作者把 occlusion layer 当成一个新的可学习维度，并做三件事：

- 生成多层 amodal views
- 用 STO-MLP 按层调制 Gaussian opacity
- 用两阶段训练逐渐从 visible region 扩展到 hidden region

### 一句话能力画像

- **输入**：few-shot sparse views
- **新增监督**：diffusion-generated layered amodal views
- **核心 operator**：layer-wise opacity modulation
- **目标**：occlusion-aware reconstruction of hidden structures

### 对我当前方向的价值

这篇对你当前方向很有价值，因为它触到一个编辑和重建都很痛的点：

**被遮挡区域怎么显式建模。**

对你的路线具体价值在：

- 对 `3DGS_Reconstruction`：它是 few-shot + occlusion-aware 的很强参考。
- 对 `3DGS_Editing`：如果编辑涉及遮挡后区域的补全或 reveal，这篇的 amodal supervision 思路很值得迁移。
- 对 `4DGS_Editing`：虽然论文是静态场景，但它借用了 4DGS 的维度重解释思想，这对未来把 temporal axis 改造成 control axis 很有启发。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

这篇的真正新意有三层：

- **把 4DGS 的 time dimension 重解释为 occlusion depth**
- **用 diffusion-based amodal completion 生成显式 hidden-content supervision**
- **按 occlusion layer 调制 Gaussian opacity，而不是只做普通 pseudo-view augmentation**

这让它和 FSGS、CoR-GS、CoMapGS 的区别很明显：  
它不是只补更多视角，而是补“更多层的可见性结构”。

### The "Aha!" Moment

最值得记住的 aha 是：

**遮挡问题不一定要通过更强几何先验硬解，也可以通过构造分层 amodal supervision，让模型“学会逐层看穿”。**

这其实是一种很像 4D 建模的思想迁移：

- 在 4DGS 里，不同时间层对应不同状态
- 在 STO-GS 里，不同遮挡层对应不同可见性状态

这种“把一个难变量改写成分层状态空间”的思路非常值得保留。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**：它不是动态编辑论文，但“把 time 当 control-like axis 用”的思路，对多层显隐、层次 reveal editing 很有启发。
- **与 3DGS_Editing**：关系很直接。编辑被遮挡对象时，amodal supervision 和 layer-wise opacity modulation 都可以变成很有用的 operator。
- **与 feed-forward Gaussians**：这篇不是前馈预测，但 diffusion inpainting 生成的 amodal views 可以作为前馈 few-shot Gaussian model 的 teacher supervision。

### 局限、风险、可迁移点

- **局限**：依赖 segmentation-based diffusion inpainting 生成 amodal views；当前景物体太大时，背景补全质量会下降。
- **风险**：如果 amodal image 生成错误，模型会学到错误的 hidden structure。
- **可迁移点**：
  - reinterpret a missing dimension as learnable layered states
  - amodal completion as Gaussian supervision
  - layer-wise opacity control for hidden-space reconstruction

---

## Part III / Technical Deep Dive

### Pipeline

```text
sparse modal views
-> segmentation + diffusion inpainting
-> layered amodal views
-> assign occlusion-layer index
-> STO-MLP modulates Gaussian opacity per layer
-> two-stage training with layer-aware sampling
-> occlusion-aware few-shot 3DGS reconstruction
```

### 关键模块

#### 1. Layered Amodal View Generation

作者不是只生成一张补全图，而是生成多层 amodal views。  
每一层都对应“再去掉一层前景遮挡之后”的场景可见性。

#### 2. STO-MLP for Layered Opacity Modulation

每个 Gaussian 的 opacity 不再是固定的，而是随 occlusion layer 改变。  
这让模型能在更深层中逐步 suppress occluders，显露 hidden structures。

#### 3. Two-stage Training

先用原始 modal views 学一个可用的基础场景，  
再在第二阶段引入 amodal views 和 layer-aware sampling 做 refinement。  
这比一开始就混合所有 supervision 更稳。

#### 4. Adaptive Amodal Loss Weight

作者还对 amodal loss 做了自适应衰减，避免高层 hidden supervision 在训练早期过强或过弱。

### 关键实验信号

- 在 Mip-NeRF360 这种遮挡和未观测区域都很多的数据集上收益最明显。
- 在 LLFF、Shiny 这种前向视角场景也仍有提升，说明它并不只是 “专门为极端遮挡” 才有效。
- Ablation 证明 two-stage training、layer-aware sampling、adaptive amodal loss 都有贡献。

### 对当前研究最可迁移的 operator

- **layered amodal supervision**
- **opacity modulation across hidden-depth layers**
- **4DGS-inspired dimension reinterpretation for static occlusion**

如果你以后在 few-shot 资产补全、遮挡恢复或 reveal-based editing 上做库建设，这篇很值得反复翻。

## Local Reading / PDF 参考

![[paperPDFs/3DGS_Reconstruction/ACMMM_2025/2025_See_Through_the_Occlusions_Few_Shot_Gaussian_Splatting_with_Layered_Amodal_Supervision.pdf]]
