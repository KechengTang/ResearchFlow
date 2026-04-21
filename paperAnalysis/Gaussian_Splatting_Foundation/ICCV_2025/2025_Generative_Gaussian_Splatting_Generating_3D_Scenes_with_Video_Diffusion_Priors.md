---
title: "Generative Gaussian Splatting: Generating 3D Scenes with Video Diffusion Priors"
venue: ICCV
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3d-scene-generation
  - video-diffusion
  - feature-field
  - multiview-consistency
  - generative-model
  - status/analyzed
core_operator: 将 3D Gaussian feature field 直接嵌入预训练 latent video diffusion model，通过渲染 feature maps 再解码到 latent space，把 3D 表示和强大的视频生成先验耦合起来，以提升生成 3D scenes 的多视角一致性。
primary_logic: |
  先用 3D Gaussian primitives 参数化一个 scene feature field，
  再把该 feature field 渲染成 feature maps 并送入预训练 latent video diffusion 流程，
  随后由 diffusion 生成多视角结果或上采样成 3D radiance field，
  最终在保留视频扩散先验的同时显著提升生成场景的 3D consistency。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_Generative_Gaussian_Splatting_Generating_3D_Scenes_with_Video_Diffusion_Priors.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T16:41
updated: 2026-04-18T16:41
---

# Generative Gaussian Splatting: Generating 3D Scenes with Video Diffusion Priors

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2503.13272](https://arxiv.org/abs/2503.13272)
> - **Summary**: GGS 的关键不是做一个“更会生成”的 3D 模型，而是把 3D Gaussian feature field 真正塞进预训练 video diffusion prior 里，让生成器本身就感知显式 3D 结构。这样它既吃到 2D/video diffusion 的大规模先验，又比纯 2D 多视角拼接更 3D-consistent。
> - **Key Performance**:
>   - 相比没有 3D representation 的相似模型，论文在 RealEstate10K 和 ScanNet++ 上把生成 3D scenes 的 FID 约提升 `20%`。
>   - 论文特别强调 multi-view consistency 与生成 3D scene quality 的双重改善。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

生成式 3D scene synthesis 常见两条路各有缺点:

- 纯 3D 生成模型缺少大规模数据和强先验
- 纯 2D/video diffusion 生成多视角图像又容易缺 3D 一致性

GGS 要解决的是:

**如何在保持强大 video diffusion prior 的同时，让生成结果具备真正的 3D scene consistency。**

### 核心能力定义

- **输入**: noise / generation condition
- **输出**: 更一致的 multi-view images 与生成 3D scenes
- **强项**: generative 3D scene synthesis、video prior 利用、3D consistency
- **弱项**: 更偏 scene generation，不是明确的 editable 4D pipeline

### 真正的挑战来源

- diffusion model 的 latent space 并不天然 3D-consistent
- 直接把 3D representation 塞进 diffusion architecture 通常要重写模型
- diffusion 预测的是 noise，不是直接的 3D scene

### 边界条件

- 方法建立在预训练 latent video diffusion 基础上
- 更适合 generative scene synthesis，不是几何精修型方法
- 重点是生成质量与一致性，而不是实时性

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

GGS 的方法哲学是:

**不要在 latent image space 里强行追 3D 一致性，而要在 diffusion 流程内部引入一个显式 3D feature field。**

Gaussian 在这里承担的是生成先验的结构锚点，而不是事后重建器。

### The "Aha!" Moment

真正的 aha 是:

**3D representation 不必直接建模 latent noise，它可以先在 feature space 中表示 3D scene，再通过 rendering 接入 diffusion 解码流程。**

作者通过这一“feature-field bridge”避开了 diffusion latent 不具备 3D 一致性的根本问题。

### 为什么这个设计有效

Gaussian feature field 提供显式 3D structure，  
video diffusion 提供强大视觉生成先验。  
当 feature maps 从 3D Gaussian 渲染出来时，多视角结果天然被 3D geometry 约束，因此一致性显著提升。

### 对我当前方向的价值

这篇对你当前方向很重要，因为它是 **scene-level generative Gaussian** 的代表作之一。  
如果后续你关心 Gaussian world models、生成式场景先验、或从 generative prior 初始化 editable scenes，这篇很值得参考。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 直接关系不如 Splat4D，但它提供了“生成一个更一致的场景底座，再做后续编辑”的思路。
- **与 3DGS_Editing**: 对静态 scene editing 来说，GGS 解决的是初始场景生成的一致性问题，是前端资产生成。
- **与 feed-forward Gaussians**: 这篇和 feed-forward Gaussian generation 的关系很直接，因为 Gaussian feature field 本身就在生成链路中被前向预测和使用。

### 战略权衡

- 优点: 充分利用视频扩散先验，同时显著增强 3D 一致性
- 代价: 系统更偏生成，训练和推理都依赖大型 diffusion backbone

---

## Part III / Technical Deep Dive

### Pipeline

```text
noise / condition
-> 3D Gaussian feature field
-> render feature maps from 3D representation
-> latent video diffusion prior
-> decode multi-view images or upsample to radiance field
-> generate consistent 3D scenes
```

### 关键模块

#### 1. 3D Gaussian Feature Field

作者不直接生成 RGB，而是先生成 3D feature field。  
这一步是显式 3D consistency 的来源。

#### 2. Feature-Space Integration

把 Gaussian 渲染结果送到 diffusion 流程里，而不是在 latent noise 空间硬套 3D 约束。

#### 3. Flexible Output Head

feature field 既可以渲染成 feature maps 再解码成多视角图像，也可以上采样成 radiance field，说明框架具有较强生成接口弹性。

### 关键实验信号

- 论文明确拿“无 3D representation 的类似模型”做对比，这证明 Gaussian field 的引入本身就是关键因果改动
- FID 改善与 multi-view consistency 改善一起出现，说明不是单纯画面更漂亮，而是 scene 更 3D-consistent

### 少量关键数字

- FID 对生成 3D scenes 约改善 `20%`

### 局限、风险、可迁移点

- **局限**: 方法重点在生成，不保证下游编辑或物理可控性
- **风险**: 若 feature field 和 diffusion prior 对齐不佳，可能出现结构/外观错配
- **可迁移点**: feature-space 3D Gaussian conditioning、diffusion-prior + explicit-scene bridge、scene-level generative Gaussian 都非常值得迁移

### 实现约束

- 依赖预训练 latent video diffusion
- 更偏 scene generation than reconstruction
- 重点是 consistency-aware generation

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_Generative_Gaussian_Splatting_Generating_3D_Scenes_with_Video_Diffusion_Priors.pdf]]
