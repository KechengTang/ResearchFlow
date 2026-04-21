---
title: "Control4D: Efficient 4D Portrait Editing with Text"
venue: CVPR
year: 2024
tags:
  - 4DGS_Editing
  - gaussian-splatting
  - 4dgs
  - portrait-editing
  - text-guided-editing
  - gaussianplanes
  - 4d-generator
  - spatiotemporal-consistency
  - multi-level-guidance
  - status/analyzed
core_operator: 先用 GaussianPlanes 把 4D Gaussian 结构化，再用 4D generator 从不一致的 2D diffusion 编辑结果中学习连续生成空间，以提升时空一致性。
primary_logic: |
  输入多视角动态人像视频与文本指令，
  先用 GaussianPlanes 以 tri-plane canonical 表示和 4D plane flow 结构化动态 Gaussian，
  再用 2D diffusion editor 生成编辑图像，并训练一个 4D generator 从这些不一致编辑图中学习更连续的 4D 生成空间，
  最终得到高保真且时空一致的 4D portrait editing 结果。
pdf_ref: paperPDFs/4DGS_Editing/CVPR_2024/2024_Control4D_Efficient_4D_Portrait_Editing_with_Text.pdf
category: 4DGS_Editing
created: 2026-04-18T13:07
updated: 2026-04-18T13:07
---

# Control4D: Efficient 4D Portrait Editing with Text

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2024 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Shao_Control4D_Efficient_4D_Portrait_Editing_with_Text_CVPR_2024_paper.pdf) · [Project Page](https://control4darxiv.github.io/)
> - **Summary**: Control4D 面向的是早期 4DGS 编辑里最棘手的两个问题: 4D 表示太慢，以及 2D diffusion 监督在时间与视角上不一致。它用 GaussianPlanes 解决前者，用 4D generator 吸收不一致编辑监督解决后者，因此比直接把 Instruct-NeRF2NeRF 扩到 4D 更稳。
> - **Key Performance**:
>   - 在静态场景编辑上，`FID 14.11`，优于 `Tensor4D+GAN 27.81` 与 `GaussianPlanes 49.32`；在动态场景上达到 `FID 18.59`，明显优于 `47.39` 与 `67.58`。
>   - 动态场景 `CLIP Similarity` 达到 `0.3192`，同时显著降低训练成本并提升时空一致性。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

Control4D 针对的是 **文本驱动 4D portrait editing**。难点不在于单帧编辑，而在于：

- 4D 表示本身训练与渲染就很重；
- 2D diffusion 编辑不同时间、不同视角时天然不一致；
- 一旦直接拿这些不一致图去监督 4D 表示，模型很容易发散、模糊或出现时序闪烁。

### 核心能力定义

- **输入**: 多视角动态人像视频与文本编辑指令。
- **输出**: 时空一致的 4D portrait editing 结果。
- **擅长**: 动态人像身份/外观变换这类 text-guided 4D editing。
- **边界**: 任务聚焦 portrait，不是开放场景级的通用 4D scene editing。

## Part II / High-Dimensional Insight

### 方法设计的核心判断

Control4D 的判断很清楚：**4D 编辑不能只把 2D diffusion 结果硬塞回动态表示，而要同时重做“4D 表示接口”和“编辑监督吸收方式”。**

### The "Aha!" Moment

真正的 aha 有两层：

1. **GaussianPlanes**: 用结构化 plane decomposition 去约束本来离散、噪声大的动态 Gaussian。
2. **4D generator**: 不直接拟合不一致编辑图，而是学一个更连续的 4D 生成空间，让 GAN 式判别信号替代生硬监督。

这意味着系统不是直接相信每一张编辑图都对，而是把它们当成 noisy targets，再由 4D generator 学出更平滑的一致表示。

### 为什么这个设计有效？

- GaussianPlanes 让 canonical 空间与时间流都变得结构化，降低 4DGS 在编辑时的噪声。
- 4D generator 把离散的、彼此矛盾的 2D 编辑结果吸收到连续 latent space，减少 blur 和 temporal inconsistency。
- multi-level guidance 让 GAN 训练更稳定，不容易 mode collapse。

### 局限

- 明显偏 portrait 域。
- 仍然依赖 diffusion editor 和 GAN 训练，系统复杂度不低。

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view dynamic portrait video + text
-> GaussianPlanes builds structured 4D Gaussian representation
-> diffusion editor produces edited images
-> 4D generator learns a continuous 4D generation space from edited data
-> discriminator + multi-level guidance stabilize supervision
-> spatiotemporally consistent edited 4D portrait
```

### 关键机制

#### 1. GaussianPlanes representation

canonical 3D Gaussian 用 tri-plane 描述静态属性，时间相关流动再映射到额外的 4D planes。这样比独立 Gaussian 点更稳，也更适合 4D 编辑。

#### 2. 4D generator instead of direct supervision

这是本篇最核心的建模选择。它不是让 4D 场景直接追逐每一张编辑后的 2D 图，而是通过 generator 学连续 4D latent，再由判别器提供更平滑的更新信号。

#### 3. Multi-level guidance

作者用多层特征引导稳定 generator 学习。去掉这部分后，动态场景 FID 会从 `18.59` 明显恶化到 `45.19` 或 `54.29`。

### 关键实验信号

- 静态编辑场景 `FID 14.11 / CLIP 0.3323`，优于多种 4D 扩展基线。
- 动态场景 `FID 18.59 / CLIP 0.3192`，显著优于 Tensor4D、Tensor4D+GAN 和单独 GaussianPlanes。
- 论文展示了 Tensor4D 数据集与 360° 人像场景，说明方法不只在单一模板上工作。

### 实现约束

- 实验主要基于 `Tensor4D` 动态半身视频数据。
- diffusion 部分使用 `Stable Diffusion 1.5`，并结合 `OpenPose` 与 normal `ControlNet`。

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Editing/CVPR_2024/2024_Control4D_Efficient_4D_Portrait_Editing_with_Text.pdf]]
