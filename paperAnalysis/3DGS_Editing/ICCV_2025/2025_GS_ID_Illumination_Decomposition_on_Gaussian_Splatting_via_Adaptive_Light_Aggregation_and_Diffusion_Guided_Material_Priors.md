---
title: "GS-ID: Illumination Decomposition on Gaussian Splatting via Adaptive Light Aggregation and Diffusion Guided Material Priors"
venue: ICCV
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - relighting
  - illumination-decomposition
  - inverse-rendering
  - adaptive-light-aggregation
  - diffusion-guided-material-priors
  - scene-composition
  - status/analyzed
core_operator: 在 3DGS 上联合建模环境光、局部各向异性球面高斯光照、逐 Gaussian 阴影方向和 diffusion-guided 材质先验，从而分解光照与材质并支持可编辑 relighting。
primary_logic: |
  输入场景图像并构建 3DGS，
  先用环境光贴图与自适应局部 SGM 光照共同建模复杂照明，
  再为每个 Gaussian 学习 shadow-aware 可见性方向以近似多光源阴影，
  同时引入 diffusion 先验约束 albedo 与 roughness，
  最终实现更可靠的 illumination decomposition，并支持 relighting 与 scene composition。
pdf_ref: paperPDFs/3DGS_Editing/ICCV_2025/2025_GS_ID_Illumination_Decomposition_on_Gaussian_Splatting_via_Adaptive_Light_Aggregation_and_Diffusion_Guided_Material_Priors.pdf
category: 3DGS_Editing
created: 2026-04-18T13:09
updated: 2026-04-18T13:09
---

# GS-ID: Illumination Decomposition on Gaussian Splatting via Adaptive Light Aggregation and Diffusion Guided Material Priors

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Du_GS-ID_Illumination_Decomposition_on_Gaussian_Splatting_via_Adaptive_Light_Aggregation_ICCV_2025_paper.pdf)
> - **Summary**: GS-ID 虽然不是传统的 text-driven scene editing，但它直接解决了 3DGS 里“几何、材质、光照纠缠在一起，导致 relighting/editing 做不准”的基础瓶颈。对做 3DGS 编辑的人来说，它更像是一个把 scene editing 推向物理可控的 decomposition backbone。
> - **Key Performance**:
>   - 在 TensoIR Synthetic 上，达到 `33.49` 的 albedo `PSNR`、`39.13 / 0.984 / 0.020` 的 novel-view synthesis 指标，以及 `28.69 / 0.947 / 0.075` 的 relighting 指标。
>   - 以约 `40 分钟` 训练时间实现近 `60 FPS` 渲染，相比 vanilla 3DGS 仅约 `1.5×` 训练开销。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

3DGS 擅长逼真渲染，但原始表示把 **geometry / material / lighting** 强烈纠缠在一起，因此一旦要做 relighting、scene composition 或物理一致编辑，就会失真。GS-ID 要解决的是：**如何在 3DGS 上做可用的 illumination decomposition，尤其是在高光、阴影和非朗伯条件下。**

### 核心能力定义

- **输入**: 多视图场景图像。
- **输出**: 可分解的材质、光照与几何，以及支持 relighting 的 3DGS。
- **擅长**: relighting、scene composition 这类物理可控编辑。
- **边界**: 不是语义文本编辑器；更偏 editing-enabling foundation。

## Part II / High-Dimensional Insight

### 方法设计的核心判断

GS-ID 的核心判断是：**只靠简单环境光或单一逆渲染先验，无法在 3DGS 上稳定拆开复杂光照与材质；必须同时增强光照表达能力、阴影建模能力和材质先验。**

### The "Aha!" Moment

真正的 aha 是三件事一起做：

1. 用 **adaptive anisotropic SGMs** 表达局部复杂光照；
2. 给每个 splat 学 **shadow direction vectors**，显式近似多光源阴影；
3. 用 **diffusion-guided albedo / roughness priors** 缩小光材歧义空间。

单独任何一项都不够，三者组合后才让 3DGS 具备更可信的 illumination decomposition。

### 为什么这个设计有效？

- SGM 光照比简单环境贴图更能覆盖空间变化光源。
- per-splat 阴影方向让 cast shadows 不再只能被当成材质噪声吸收。
- diffusion 先验为材质分解提供外部统计约束，减少 light-material ambiguity。

### 局限

- 主要是 inverse rendering / relighting 场景，不是开放词汇语义编辑。
- 系统中包含较多物理项与先验项，工程复杂度高于普通 3DGS。

## Part III / Technical Deep Dive

### Pipeline

```text
scene images
-> build 3DGS
-> model ambient light with learnable environment map
-> model local light with adaptive anisotropic SGMs
-> learn per-splat shadow-aware visibility vectors
-> regularize albedo and roughness with diffusion priors
-> obtain decomposed scene for relighting and composition
```

### 关键机制

#### 1. Adaptive light aggregation

核心是局部各向异性 SGM 混合光照。它比只用全局环境光更适合处理空间变化大的真实照明。

#### 2. Visibility-aware deshadowing

给每个 Gaussian 绑定一个可学习单位向量来表示来自多光源的阴影方向，这一步对 disentangle 阴影和材质非常关键。

#### 3. Diffusion-guided material priors

用预训练 diffusion 模型提供 albedo 与 roughness 先验，不让优化过程把高光、阴影错误地解释成材质本体。

### 关键实验信号

- TensoIR 上各项指标整体优于 GSshader、RelightGS 和 GS-IR。
- 除了 decomposition 指标，论文还展示了 downstream relighting 与 scene composition 结果，说明分解结果是“可编辑”的，不只是分析用。

### 实现约束

- 主要评测数据集包括 `TensoIR Synthetic`、`ADT` 和 `Mip-NeRF 360`。
- 通过 CUDA kernel 做 chunked lighting evaluation，以控制显存开销。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ICCV_2025/2025_GS_ID_Illumination_Decomposition_on_Gaussian_Splatting_via_Adaptive_Light_Aggregation_and_Diffusion_Guided_Material_Priors.pdf]]
