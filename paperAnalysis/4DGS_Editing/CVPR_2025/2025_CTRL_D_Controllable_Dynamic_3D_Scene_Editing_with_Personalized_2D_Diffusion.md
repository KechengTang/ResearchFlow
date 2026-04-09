---
title: "CTRL-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion"
venue: CVPR
year: 2025
tags:
  - 4DGS_Editing
  - gaussian-splatting
  - 4dgs
  - personalized-diffusion
  - instructpix2pix
  - single-reference-editing
  - two-stage-optimization
  - edited-image-buffer
  - local-editing
  - deformable-gaussians
  - status/analyzed
core_operator: 用单张用户编辑参考图像个性化微调 InstructPix2Pix，把局部编辑风格和区域控制蒸馏进 2D 编辑器，再通过两阶段 deformable Gaussian 优化将这种个性化编辑稳定传播到动态 3D 场景。
primary_logic: |
  用户先手工或借助任意 2D 编辑器编辑一张参考帧，
  再用该参考图像和原场景图像个性化微调 InstructPix2Pix 以学习目标区域与风格，
  随后对 deformable Gaussian 场景做两阶段优化，并借助 edited image buffer 加快收敛并提升时序一致性。
pdf_ref: paperPDFs/4DGS_Editing/CVPR_2025/2025_CTRL_D_Controllable_Dynamic_3D_Scene_Editing_with_Personalized_2D_Diffusion.pdf
category: 4DGS_Editing
created: 2026-04-09T16:20
updated: 2026-04-09T16:20
---

# CTRL-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://ihe-kaii.github.io/CTRL-D/) · [arXiv 2412.01792](https://arxiv.org/abs/2412.01792)
> - **Summary**: CTRL-D 的思路和很多纯文本驱动 4D 编辑不同，它不追求让通用扩散模型直接“猜中”你想要的局部编辑，而是让用户先给一张编辑参考图，然后围绕这张图个性化微调 InstructPix2Pix。这样模型学到的不是抽象 prompt，而是具体区域、具体风格、具体编辑方式，最后再通过两阶段 deformable Gaussian 优化把这种编辑传播到整个动态场景。
> - **Key Performance**:
>   - 论文强调其局部编辑可控性明显优于 baseline `Instruct 4D-to-4D`。
>   - 项目页把优势集中在“更精确局部编辑 + 更高时序一致性 + 更少跟踪负担”三点上。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

CTRL-D 针对的是动态 3D/4D 编辑里另一个很痛的点：  
**用户想改的是一个很具体的局部对象或区域，但纯文本驱动方法常常既不够稳定，也不够可控。**

常见问题包括：

- 编辑区域不准，模型会误改周边
- 风格不稳定，不同时间点长得不一样
- 需要用户显式跟踪编辑区域，使用门槛高

CTRL-D 的切入方式非常“交互导向”：  
不要只给 prompt，让用户先给一个编辑好的参考帧，然后让模型围绕这张参考图学习你真正想要的改动。

### 它的方法直觉

核心直觉有两点：

1. **单张高质量参考图，比纯 prompt 更能表达局部编辑意图**
2. **先把 2D 编辑器个性化，再去优化 3D/4D 场景，比直接在 3D 里猜用户意图更稳**

这使得 CTRL-D 不是单纯的“文本到动态场景”，而更像是：

- 用户给一个编辑示例
- 模型学这个示例
- 再把示例风格和局部修改传播到整个动态场景

### 一句话能力画像

- **输入**：原动态场景、一张编辑参考图、文本或编辑意图
- **核心模块**：个性化 InstructPix2Pix + 两阶段 deformable Gaussian 优化
- **优势**：局部控制强、区域不需要显式跟踪、风格传播更准确
- **代价**：依赖参考图质量；本质上仍然是“2D 编辑器驱动 3D 优化”的范式

### 对你的研究最重要的启发

CTRL-D 对你最有价值的启发是：

**如果编辑目标很具体，用户示例图往往比自然语言本身更强。**

这对 4DGS 编辑非常重要，因为很多真正实用的编辑任务不是开放生成，而是：

- 改某个人的衣服
- 改某个对象的材质
- 改某一块局部区域的外观

这些任务里，“个性化编辑先验”往往比“纯文本语义”更关键。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

CTRL-D 最核心的创新不只是用 InstructPix2Pix，而是把动态场景编辑的可控性问题重写成了：

**如何从单张参考编辑图里，学到区域、风格和编辑模式，然后稳定地投射回动态 3D 表示。**

这和很多只依赖 prompt 的方法相比，目标函数本身就更明确。

### The "Aha!" Moment

真正的 aha moment 是：

**CTRL-D 不是让编辑器“理解文本”，而是让编辑器“模仿用户已经做出的那次编辑”。**

一旦这样理解，这篇论文的所有设计都顺了：

- 为什么要先微调 InstructPix2Pix  
  因为要吸收参考编辑图中的区域和风格信息。
- 为什么可以不用显式追踪编辑区域  
  因为区域信息已经隐含在示例编辑里。
- 为什么局部编辑更稳  
  因为目标不是开放式生成，而是参考驱动的迁移。

### 为什么这个思路值得长期记住

它给 4DGS 编辑提出了一个实用分支：

- 一类方法追求纯文本、开放式编辑
- 另一类方法追求用户可控、示例驱动编辑

CTRL-D 很明显属于第二类。  
如果你的未来系统更重视交互而不是完全自动生成，那么 CTRL-D 这条线很值得保留。

### 它的隐含限制

- 参考图本身如果质量不好，个性化微调学到的也是偏的
- 它仍然依赖 2D 编辑器作为前端知识源
- 真正结构性很强的运动/几何变化未必容易通过单张参考图完全表达

---

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic scene
-> edit one reference frame
-> fine-tune InstructPix2Pix with edited reference + original samples
-> optimize deformable Gaussian scene in two stages
-> use edited image buffer for faster convergence and better temporal consistency
-> edited dynamic scene
```

### 关键模块

#### 1. Personalized InstructPix2Pix

论文最关键的入口就是先把 InstructPix2Pix 个性化。  
作者不是直接拿通用 IP2P 去编辑所有帧，而是用：

- 一张参考编辑图
- 原始场景采样图像

来做微调。  
后者的作用是保持原模型先验，不让编辑器因为只看单张示例而过拟合到奇怪风格。

#### 2. Two-stage Optimization

CTRL-D 强调使用两阶段优化逐步编辑动态场景。  
虽然项目页没有展开全部细节，但它明确把两阶段设计和以下目标绑定在一起：

- 收敛更快
- 局部编辑更稳
- 时间一致性更好

这说明作者并不认为“一次性把编辑全压进 3D 优化”是稳妥的，而是更偏向 coarse-to-fine 的渐进式编辑。

#### 3. Edited Image Buffer

这是一个很实用的 engineering 设计。  
论文专门提到它被用来：

- 加速优化收敛
- 提高 temporal consistency

从系统角度看，这相当于在 2D 编辑监督和 3D 优化之间加入了一个缓冲层，降低瞬时监督噪声对动态场景更新的冲击。

### 关键实验信号

- 项目页把对比集中在与 Instruct 4D-to-4D 的比较上，强调局部精确性更强。
- 作者反复强调“不需要跟踪目标区域”，这说明它对可用性很敏感。
- 论文的主要卖点不是完全新奇的编辑效果，而是“让用户真的能控制编辑发生在哪、以什么风格发生”。

### 对你可迁移的 operator

- **reference-image-driven personalization**
- **two-stage optimization for dynamic editing**
- **buffered supervision between 2D edits and 4D scene optimization**

如果你以后想做“科研上更实用”的 4DGS 编辑系统，而不只是 benchmark 上的 text editing，CTRL-D 这条示例驱动路线很值得参考。

---

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Editing/CVPR_2025/2025_CTRL_D_Controllable_Dynamic_3D_Scene_Editing_with_Personalized_2D_Diffusion.pdf]]
