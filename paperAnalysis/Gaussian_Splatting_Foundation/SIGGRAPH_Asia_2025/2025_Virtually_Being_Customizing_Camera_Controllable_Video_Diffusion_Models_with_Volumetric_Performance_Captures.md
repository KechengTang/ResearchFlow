---
title: "Virtually Being: Customizing Camera-Controllable Video Diffusion Models with Volumetric Performance Captures"
venue: SIGGRAPH Asia
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - video-generation
  - camera-control
  - multi-view-identity
  - volumetric-capture
  - relighting
  - virtual-production
  - multi-subject-generation
  - status/analyzed
core_operator: 不把 4DGS 当最终输出，而是把 volumetric capture + 4DGS reconstruction + relighting 变成多视角、可控相机、可控光照的 customization data pipeline，再通过两阶段训练把 camera-conditioned video diffusion model 定制成具备多视角身份一致性的视频生成器。
primary_logic: |
  先用专业多机位 volumetric captures 重建动态人物，
  再通过 4D Gaussian Splatting 以多种相机轨迹重渲染视频并叠加 relighting 增广，
  形成带相机与光照标注的 subject-specific customization dataset，
  随后先在通用相机数据上预训练 camera-conditioned 视频生成模型，再在定制数据上微调，
  最终实现多视角身份保持、3D camera control、多主体组合与场景交互。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/SIGGRAPH_Asia_2025/2025_Virtually_Being_Customizing_Camera_Controllable_Video_Diffusion_Models_with_Volumetric_Performance_Captures.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T20:35
updated: 2026-04-18T20:35
---

# Virtually Being: Customizing Camera-Controllable Video Diffusion Models with Volumetric Performance Captures

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2510.14179](https://arxiv.org/abs/2510.14179) · [Project Page](https://eyeline-labs.github.io/Virtually-Being/) · [DOI](https://doi.org/10.1145/3757377.3763888)
> - **Metadata Note**: 当前 intake / 文件名使用的是 **“Volumetric Performance Captures”**，但 PDF 正式题名是 **“Multi-View Performance Captures”**。这里复用现有 log 条目与 `pdf_path`，不新建重复记录。
> - **Summary**: 这篇论文很有代表性地展示了另一条 Gaussian 路线：4DGS 不再是最终交付物，而是高质量训练数据生成器。作者把专业 volumetric capture 重建成 4DGS，再渲染出大量带相机轨迹和光照变化标注的视频，用来定制 camera-controllable video diffusion model，从而获得多视角身份保持、3D 相机控制和多主体组合能力。
> - **Key Performance**:
>   - 相机控制上，`TransErr 0.267 / RotErr 0.047`，优于 CameraCtrl 与 AC3D。
>   - 用户研究中，方法在多视角身份保持、面部真实感、文本对齐上分别获得 `81.34% / 70.59% / 74.07%` 偏好。
>   - 加入 relit data 后，用户对光照真实感的偏好达到 `83.9%`；自定义 I2V 模型后，身份保持偏好达到 `65.43%`。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

这篇论文解决的是：

**如何让视频扩散模型在 subject customization 时，既保持多视角身份一致，又能精确服从 3D camera motion。**

普通个性化视频生成通常有两个明显短板：

- 只有单张或少量参考图，跨视角身份不稳
- 即使支持相机控制，也常停留在 2D 平移层面，难以覆盖电影式 3D camera move

作者的答案不是直接重做一个更强的 diffusion model，而是先重做定制数据。

### 它的方法直觉

方法直觉非常重要：

**想要多视角 identity preservation，关键不是只在 loss 上约束，而是给模型看真正带多视角、带相机轨迹、带光照变化的 subject-specific 视频。**

而 4DGS 在这里扮演的是：

- 高保真多视角重渲染器
- camera trajectory 标注器
- 结合 relighting 的数据扩增引擎

### 一句话能力画像

- **输入数据**：studio volumetric captures / CG renders / real-life videos
- **Gaussian 角色**：4DGS data generator，而不是最终部署表示
- **训练策略**：camera pretraining + subject customization 两阶段
- **新增能力**：multi-view identity、3D camera control、lighting adaptability、multi-subject generation

### 对我当前方向的价值

这篇对你当前 Gaussian 方向的意义，不在于“它是不是 4DGS reconstruction 论文”，而在于：

**它展示了 4DGS 可以作为 generative model 的上游 teacher / data engine。**

具体价值包括：

- 对 `4DGS_Editing`：如果未来想训练可编辑的 camera-aware video model，4DGS 可以先生成高质量监督数据。
- 对 `3DGS_Editing`：静态/动态角色编辑都需要 view-consistent identity prior，这篇给出了如何构造这种先验。
- 对 `feed-forward Gaussians`：它说明高质量 Gaussian 世界表征可以反过来训练更快的前馈生成/编辑器。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

真正的新意有三层：

- **把 4DGS 当作 customization data pipeline 的核心组件**，而不是最终输出。
- **两阶段训练**：先学 general camera-conditioned generation，再学 subject-specific multi-view identity。
- **把 relighting、多主体组合、scene customization、I2V customization 全部并进同一条 production-oriented 路线。**

它的贡献不是提出新的 Gaussian primitive，而是改变了 Gaussian 在生成系统中的角色定位。

### The "Aha!" Moment

这篇最值得记住的 aha 是：

**在生成式系统里，4DGS 的价值不一定是“在线可编辑表示”，也可以是“离线高保真 supervision source”。**

这非常像把 NeRF / 4DGS 从 final renderer 变成 data flywheel：

- 先重建真实人和场景
- 再渲染出更适合 diffusion 学习的多视角训练样本
- 再把学到的能力返还给生成模型

这条路径对以后做 camera-aware editing 或 identity-preserving generation 都很重要。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**：不是直接编辑 pipeline，但它非常像 4DGS 编辑器的上游训练器。未来如果要训练 camera-aware 4D editor，这种数据生成方式很有参考价值。
- **与 3DGS_Editing**：可迁移的是 “multi-view identity supervision from Gaussian renders” 这层思想，而不是具体的 4D deformation。
- **与 feed-forward Gaussians**：关系很直接。它本质上就是用高质量 Gaussian 数据去支持更快的前馈视频生成/定制模型。

### 局限、风险、可迁移点

- **局限**：依赖昂贵的 volumetric capture 设备与高质量 4D 重建。
- **风险**：如果 capture domain 和目标 domain 差异太大，定制模型的泛化仍受 base diffusion model 限制。
- **可迁移点**：
  - 4DGS as supervision engine
  - camera-conditioned video generation with explicit trajectory labels
  - relit Gaussian renders for lighting robustness
  - joint training + noise blending 的多主体组合策略

---

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view performance capture
-> 4D Gaussian reconstruction
-> render videos with diverse camera trajectories
-> relight rendered videos for lighting diversity
-> build subject-specific customization dataset
-> pretrain camera-conditioned video model on general data
-> fine-tune on customized data
-> identity-preserving, camera-controllable video generation
```

### 关键模块

#### 1. 4DGS-based Customization Data Pipeline

作者先采集 75 相机面部系统和 160 相机全身系统的数据，再用 4DGS 重建动态人物。  
重点不是重建本身，而是后续能精确重渲染任意相机轨迹并自动带上 camera annotation。

#### 2. Relighting Augmentation

为了让模型学会电影制作里很关键的 lighting variability，作者引入 video relighting model 生成不同 HDR 光照条件。  
这一步对“身份不变但光照变化很大”的场景尤其重要。

#### 3. Two-stage Training

第一阶段学 camera-conditioned generation；  
第二阶段做 subject-specific customization。  
这个分解让模型既保留通用 camera control，又获得定制身份。

#### 4. Joint Training + Noise Blending

多主体生成不是只能 joint finetune，作者还给出 noise blending，在推理时组合独立定制模型。  
这对实际系统很有价值，因为它允许更模块化的 subject composition。

### 关键实验信号

- Baseline comparison 中，方法在用户对多视角 identity 的偏好上远高于 MagicMe、DreamVideo、VideoBooth、MotionBooth、ConsisID。
- Camera control 定量表明它不仅“看起来更像”，而且 camera trajectory 跟随更准。
- relit data、joint-subject data、customized I2V 都通过单独用户研究验证，说明整条 pipeline 是按生产需求逐层构建的，而不是堆概念。

### 对当前研究最可迁移的 operator

- **4DGS renders as controllable supervision**
- **camera-annotated Gaussian re-rendering for customization**
- **relighting-augmented identity data**
- **Gaussian-first, diffusion-second 的 teacher-student pipeline**

如果你后面想把 Gaussian world representation 和 controllable video generation 接起来，这篇会是很关键的桥梁论文。

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/SIGGRAPH_Asia_2025/2025_Virtually_Being_Customizing_Camera_Controllable_Video_Diffusion_Models_with_Volumetric_Performance_Captures.pdf]]
