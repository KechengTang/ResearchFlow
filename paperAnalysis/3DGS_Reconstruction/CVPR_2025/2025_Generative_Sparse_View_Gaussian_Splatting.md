---
title: "Generative Sparse-View Gaussian Splatting"
venue: CVPR
year: 2025
tags:
  - 3DGS_Reconstruction
  - gaussian-splatting
  - sparse-view
  - diffusion-prior
  - pseudo-view-hallucination
  - geometry-aware-finetuning
  - depth-regularization
  - dynamic-scene-reconstruction
  - 4dgs
  - status/analyzed
core_operator: 通过“GS 渲染伪视角 -> depth adapter 条件扩散 hallucinate 新图 -> scene-specific LoRA 与 GS 交替优化”的闭环，把预训练 diffusion prior 变成 sparse-view 3D/4DGS 的数据扩增器，并用 geometry-aware feature correspondence 与 depth regularization 限制 hallucination 漂移。
primary_logic: |
  先用稀疏训练视角初始化 3DGS / 4DGS，
  再在伪视角渲染图像并估计深度，送入带 LoRA 的预训练 diffusion model 生成 scene-specific pseudo views，
  同时通过已知相机位姿把伪视角图像回投到训练视角，在 diffusion feature space 中施加 geometry-aware consistency，
  最后用真实视角与 hallucinated 视角共同优化 Gaussian 表示，从而显著提升 sparse-view 静态与动态重建。
pdf_ref: paperPDFs/3DGS_Reconstruction/CVPR_2025/2025_Generative_Sparse_View_Gaussian_Splatting.pdf
category: 3DGS_Reconstruction
created: 2026-04-18T20:35
updated: 2026-04-18T20:35
---

# Generative Sparse-View Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Kong_Generative_Sparse-View_Gaussian_Splatting_CVPR_2025_paper.pdf)
> - **Summary**: GS-GS 的核心不是“用 diffusion 替代 GS”，而是让 diffusion 成为 sparse-view GS 的伪视角生成器，并且通过几何一致性约束把 hallucination 驯服到可用于重建的程度。它是一条很典型的 generative prior + explicit Gaussian representation 的混合路线。
> - **Key Performance**:
>   - 静态场景上，Blender / LLFF / Mip-NeRF360 分别达到 `28.57 / 0.923 / 0.055`、`24.82 / 0.737 / 0.105`、`25.87 / 0.745 / 0.182`。
>   - 动态场景 Neural 3D Video 上，仅 3 视角就达到 `27.13 / 0.907 / 0.135`，显著优于 SpacetimeGS 的 `14.98 / 0.774 / 0.327`。
>   - Ablation 显示 LLFF 上 PSNR 从无 hallucination 的 `17.43`，到有 diffusion 但无 geometry-aware fine-tuning 的 `22.71`，再到 full model 的 `24.82`。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

这篇解决的是：

**当训练相机视角极少时，3DGS / 4DGS 如何补足欠采样区域的信息缺口。**

Sparse-view reconstruction 的核心困难是：

- 几何和外观都欠约束
- textureless 区域更容易漂
- 靠普通 regularization 往往不够
- 纯生成方法又常常丢失几何一致性

GS-GS 的目标不是只生成好看的新视角，而是让生成出的伪视角真正能反过来提升 GS 重建。

### 它的方法直觉

方法直觉是：

**让 GS 和 diffusion 互相喂数据，但必须加一个几何一致性闸门。**

作者做了一个交替优化闭环：

- GS 提供当前场景的伪视角渲染
- diffusion 用这些渲染和深度条件 hallucinate 更真实的新视角
- 新视角再反馈给 GS 继续优化

如果没有 geometry-aware fine-tuning，这个闭环很容易越训越偏；  
因此作者额外用 feature correspondence 去钉住跨视角几何。

### 一句话能力画像

- **输入**：少量已知位姿视角
- **Gaussian 基座**：3DGS for static，SpacetimeGS for dynamic
- **生成模块**：pretrained diffusion + depth adapter + scene-specific LoRA
- **关键补丁**：geometry-aware diffusion fine-tuning + depth regularization
- **输出**：静态与动态 sparse-view novel view synthesis

### 对我当前方向的价值

这篇对你当前方向的价值很直接：

- 对 `3DGS_Reconstruction`：它是 sparse-view GS 的强 generative prior 路线。
- 对 `4DGS_Reconstruction`：作者明确展示这条 pipeline 也能套到 4DGS 上。
- 对 `3DGS_Editing` / `4DGS_Editing`：很多编辑系统的前提是先拿到一个可用的 Gaussian 底座；GS-GS 提供了一个在少视角条件下补底座的有效方案。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

论文最重要的创新点有三层：

- **alternating optimization of GS and diffusion**：不是一次性生成伪视角，而是让两者循环改进。
- **geometry-aware diffusion fine-tuning**：通过已知相机位姿把伪视角 warp 回训练视角，并在 diffusion feature space 中做一致性约束。
- **同一条 pipeline 同时适配 3D 和 4D**：这让它不只是 static sparse-view trick，而是一个更普适的 GS enhancement recipe。

### The "Aha!" Moment

最重要的 aha 是：

**稀疏视角场景下，真正需要的不是更多随机生成图，而是“可被几何解释的新观测”。**

也就是说，diffusion 不能只负责视觉补全，必须被 explicit scene structure 驯化。  
GS-GS 的贡献就在于把这一点做成了一个可训练闭环。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**：间接但重要。它能在少视角/少相机条件下先补一个更完整的 dynamic Gaussian 底座，再做后续编辑。
- **与 3DGS_Editing**：关系更直接。很多静态编辑任务面临资产采集不足，GS-GS 适合做编辑前的资产补全。
- **与 feed-forward Gaussians**：它不是纯 feed-forward 预测，但它已经非常接近“让生成模型给 Gaussian 重建提供前馈先验”的范式，可视为 feed-forward Gaussian 方向的过渡方案。

### 局限、风险、可迁移点

- **局限**：依赖预训练 diffusion model 与 depth adapter，训练链路较重。
- **风险**：如果 hallucinated pseudo views 与真实几何偏差过大，可能把错误观测反向灌入 GS。
- **可迁移点**：
  - scene-specific LoRA for reconstruction
  - geometry-aware feature-level consistency
  - pseudo-view hallucination as reconstruction supervision
  - static / dynamic unified sparse-view recipe

---

## Part III / Technical Deep Dive

### Pipeline

```text
sparse input views
-> initialize 3DGS / 4DGS
-> render pseudo-view images
-> estimate pseudo-view depth
-> diffusion + depth adapter hallucinate refined pseudo views
-> warp pseudo-view render to train views for feature correspondence
-> optimize LoRA and Gaussian representation alternately
-> sparse-view static / dynamic reconstruction
```

### 关键模块

#### 1. Scene-specific LoRA Adaptation

作者不是直接用通用 diffusion model，而是用当前场景渲染图持续 fine-tune LoRA。  
这样 hallucination 会逐渐从通用先验变成 scene-specific prior。

#### 2. Geometry-aware Diffusion Fine-Tuning

这是整篇最关键的稳定器。  
给定伪视角渲染图，先利用已知相机位姿 warp 回训练视角，再约束两者的 diffusion feature 尽量一致，从而减少跨视角 hallucination 漂移。

#### 3. Depth Regularization for GS

除了伪视角图像，作者还把深度约束引入 GS 优化。  
这一步尤其帮助 sparse-view 下的几何细节恢复，ablation 也证明确实有效。

### 关键实验信号

- 表 1 显示在 Blender、LLFF、Mip-NeRF360 三个静态数据集上都领先现有 sparse-view baseline。
- 表 2 说明动态场景上也成立，而且在极端 3-view 条件下优势尤其大。
- 表 3 是最有启发性的：
  - 只用 vanilla GS：很差
  - 加 diffusion hallucination：明显提升
  - 再加 geometry-aware fine-tuning：进一步稳定
  - 最后再加 depth regularization：最好  
  这清楚说明整套 pipeline 是逐层搭起来的。

### 对当前研究最可迁移的 operator

- **Gaussian reconstruction + diffusion hallucination closed loop**
- **feature-space geometric consistency instead of pixel-only filtering**
- **pseudo-view supervision for sparse 4DGS**

如果你接下来要继续扩 sparse-view 或 low-data 的 Gaussian 库，这篇属于很值得保留的桥梁型论文。

## Local Reading / PDF 参考

![[paperPDFs/3DGS_Reconstruction/CVPR_2025/2025_Generative_Sparse_View_Gaussian_Splatting.pdf]]
