---
title: "GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing"
venue: ECCV
year: 2024
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - scene-editing
  - controlnet
  - depth-conditioned-editing
  - multi-view-consistency
  - latent-code-alignment
  - localized-editing
  - status/analyzed
core_operator: 用深度条件 ControlNet 保证跨视角几何一致，再用基于参考视角的 latent attention alignment 统一外观，从而一次性编辑所有视图并仅更新一次 3DGS。
primary_logic: |
  输入源 3DGS 与编辑文本，
  先渲染多视角 RGB 与 depth，并通过 DDIM inversion 把每张图映射到与原图对齐的 latent 起点，
  再在深度条件下用 ControlNet 编辑全部视图，并通过 AttnAlign 把当前视图的 self-attention 与参考视图 cross-attention 混合，
  最后用这组一致编辑图像监督原始 3DGS，得到更稳定的 3D 编辑结果。
pdf_ref: paperPDFs/3DGS_Editing/ECCV_2024/2024_GaussCtrl_Multi_View_Consistent_Text_Driven_3D_Gaussian_Splatting_Editing.pdf
category: 3DGS_Editing
created: 2026-04-18T16:20
updated: 2026-04-18T16:20
---

# GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ECCV 2024 PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02153.pdf) · [Project Page](https://gaussctrl.active.vision/)
> - **Summary**: GaussCtrl 把 3DGS 编辑中的一致性问题拆成“几何一致”和“外观一致”两层，分别用 depth-conditioned ControlNet 与 cross-view latent alignment 去处理，因此不必像 IN2N 那样一张一张慢慢平均，而是能一次把所有视图一起编辑完再更新 3D 模型。
> - **Key Performance**:
>   - 在 `6` 个场景中有 `4` 个场景的 `CLIPdir` 最优，例如 `Bear Statue 0.1388`、`Dinosaur 0.1584`、`Stone Horse 0.2268`、`Face 0.1503`。
>   - 单场景编辑约 `9 min`，运行于 `RTX A5000 24GB`；论文同时强调其是对比方法中最快的一类方案。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

GaussCtrl 盯住的是 IN2N 一类方法的老问题：

**把 3D 编辑拆成“单张 2D 编辑 + 3D 更新”虽然可行，但由于每张图都被独立修改，最终 3D 会出现模糊、视角不一致，甚至 face-on-the-back 之类的反常结果。**

因此，这篇论文真正要解决的是：

**如何让 2D diffusion 编辑从一开始就带上跨视角一致性，而不是指望 3D 优化事后补救。**

### 核心能力定义

- **输入**：源 3DGS 场景与文本编辑指令。
- **输出**：编辑后的 3DGS 场景。
- **支持场景**：既能改局部对象，也能改整体环境。
- **主要收益**：比逐图迭代编辑更快，同时减少多视角不一致导致的模糊与伪影。

### 真正的挑战来源

- 2D diffusion 模型单次只看一张图，天然不会维护跨视角几何一致性。
- 即便用深度条件控制住几何，外观和纹理仍可能在不同视角漂。
- 在背面、遮挡或困难视角下，2D 编辑模型特别容易生成“强行正面化”的错误结果。

### 边界条件

- 该方法更偏向几何基本不变、外观与局部语义修改的编辑。
- 深度条件意味着它并不主打大幅度拓扑变化。
- 本文自己也指出 `CLIPdir` 未必总能真实反映视觉质量，因此定量结果要结合可视化理解。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

GaussCtrl 的思路非常工程化：

**几何一致性交给 depth，外观一致性交给 latent alignment。**

相比“只换个更强编辑器”，作者更在意的是把 inconsistent edits 的来源分解掉：

- 几何错位来自没有共享 3D 几何条件。
- 外观漂移来自每张图 latent self-attention 彼此独立。

### The "Aha!" Moment

最关键的 aha 有两个。

第一，**depth map 天然是跨视角一致的几何信号**。作者不再用 InstructPix2Pix，而是转向 ControlNet，并把从 3DGS 渲染出的 depth 一起作为条件输入。这样编辑器至少不会轻易把不同视角改成互相矛盾的几何结构。

第二，**仅有 depth 还不够，因为外观和语义仍可能漂**。因此作者再引入 `AttnAlign`，把当前视图 latent 的 self-attention 与若干 reference views 的 cross-view attention 混合，让各视图往共同的 appearance standard 靠拢。

### 为什么这个设计有效？

- `DDIM inversion` 让编辑从原图对应的 latent 出发，而不是从随机噪声开始，这天然保留了原场景的一致结构与风格。
- 深度条件把编辑约束在现有几何附近，减少“凭空脑补”。
- `AttnAlign` 用参考视角给难视角补上下文，尤其能修复背面视角、遮挡视角的外观崩坏。
- 最终所有视图一起编辑、3DGS 只更新一次，避免了 iterative averaging 带来的慢收敛。

### 战略权衡

- **优点**：流程清晰，几何与外观问题分开处理，效果稳定。
- **代价**：对 reference views、depth 质量和 inversion 质量都有依赖。
- **能力边界**：更适合 appearance / texture / moderate local edits，不是专门为剧烈几何变形设计。

## Part III / Technical Deep Dive

### Pipeline

```text
source 3DGS + text prompt
-> render RGB images and depth maps
-> DDIM inversion to obtain image-aligned latents
-> depth-conditioned ControlNet editing
-> AttnAlign mixes self-attention with reference-view cross-attention
-> decode edited multi-view images
-> optional LangSAM masks for local object editing
-> optimize original 3DGS once with edited images
-> edited 3DGS
```

### 关键模块

#### 1. Depth-conditioned editing

作者把 ControlNet 当作 geometry-consistency carrier。由于 depth 是从同一个 3DGS 渲染出来的，不同视角之间天然几何匹配，所以多视角编辑不再完全依赖文本提示自由发挥。

#### 2. DDIM inversion

这一步在文中很关键。作者明确指出，如果用随机噪声起步，ControlNet 更像“重新生成”，容易把整体风格带跑。把原图 inversion 到 latent 后再编辑，才能保留与源图像对齐的 appearance prior，并显著增强多视角一致性。

#### 3. Attention-based latent code alignment

AttnAlign 把待编辑视图的 attention 与多个 reference views 混合。self-attention 负责保留当前视图局部特性，cross-view attention 负责从参考视角借来更稳定的语义与纹理。这样既不至于所有视图完全一样，也不会每张图各改各的。

#### 4. Local object editing

对局部物体编辑时，作者额外使用 `LangSAM` 生成 mask，过滤背景，使编辑更聚焦在目标对象。

### 关键实验信号

- 定量上，GaussCtrl 在 `6` 个场景里拿到 `4` 个场景最优 `CLIPdir`，并且作者强调它是最快的方案之一。
- 定性上，它最核心的优势是显著减轻 side view 的模糊、背面出现脸等多视角不一致伪影。
- 消融显示，单独使用 InstructPix2Pix 或只用带随机噪声的 ControlNet 都会在困难视角出错；加入 inverted latent 后风格更稳，再加 `AttnAlign` 后跨视角一致性进一步提升。

### 少量关键数字

- Bear Statue：`0.1388`
- Dinosaur：`0.1584`
- Stone Horse：`0.2268`
- Face：`0.1503`
- 单场景时间：约 `9 min`

### 实现约束

- 实现基于 `splatfacto`、`Stable Diffusion v1.5` 和对应 `ControlNet`。
- 参考视角数设为 `Nr = 4`，`AttnAlign` 中混合系数 `λ = 0.6`。
- 运行环境为 `NVIDIA RTX A5000 24GB`。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ECCV_2024/2024_GaussCtrl_Multi_View_Consistent_Text_Driven_3D_Gaussian_Splatting_Editing.pdf]]
