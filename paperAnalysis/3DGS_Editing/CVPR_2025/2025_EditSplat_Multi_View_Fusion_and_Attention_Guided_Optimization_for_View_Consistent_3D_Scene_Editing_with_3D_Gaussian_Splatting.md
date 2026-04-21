---
title: "EditSplat: Multi-View Fusion and Attention-Guided Optimization for View-Consistent 3D Scene Editing with 3D Gaussian Splatting"
venue: CVPR
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - scene-editing
  - multi-view-fusion-guidance
  - attention-guided-trimming
  - local-editing
  - diffusion-guidance
  - optimization-efficiency
  - status/analyzed
core_operator: 用 Multi-view Fusion Guidance 把跨视角融合图像直接注入 diffusion guidance，再用 Attention-Guided Trimming 依据 cross-attention 权重剪掉高注意冗余高斯并只优化语义相关区域。
primary_logic: |
  输入源 3DGS 与文本指令，
  先编辑全视角图像并基于 3DGS 深度把邻近视角投影融合到目标视角，形成带多视角信息的 fusion guidance，
  再通过多条件 classifier-free guidance 同时平衡文本、源图与 fusion 图，引导 2D diffusion 生成一致编辑结果，
  最后把 cross-attention 逆投影到 3D 高斯上，对高注意区域做 pruning 与 selective optimization，以更高效率得到语义更集中的 3D 编辑结果。
pdf_ref: paperPDFs/3DGS_Editing/CVPR_2025/2025_EditSplat_Multi_View_Fusion_and_Attention_Guided_Optimization_for_View_Consistent_3D_Scene_Editing_with_3D_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-18T17:10
updated: 2026-04-18T17:10
---

# EditSplat: Multi-View Fusion and Attention-Guided Optimization for View-Consistent 3D Scene Editing with 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_EditSplat_Multi-View_Fusion_and_Attention-Guided_Optimization_for_View-Consistent_3D_Scene_CVPR_2025_paper.pdf) · [Project Page](https://kuai-lab.github.io/editsplat2024/)
> - **Summary**: EditSplat 同时盯住了两件老问题：一是多视角 2D 编辑彼此不一致，二是预训练高斯保留太多源场景信息，导致编辑优化发力不够。它用 `MFG` 解决前者，用 `AGT` 解决后者，因此比只做 multi-view consistency 或只做 local mask 的方案都更完整。
> - **Key Performance**:
>   - `CLIP dir 0.1431 / CLIP sim 0.2531 / User Study 0.4227`，均高于 `DGE 0.1222 / 0.2394 / 0.2143`、`GaussCtrl 0.0979 / 0.2336 / 0.1987` 和 `GaussianEditor 0.0923 / 0.2243 / 0.1643`。
>   - 单张 `Face` 场景完整编辑约 `6 min`，运行在单张 `RTX A6000` 上。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

EditSplat 真正想解决的是 3DGS 文本编辑里两个互相耦合的问题：

**第一，2D 编辑不一致导致 3D 监督噪声很大；第二，即使监督对了，预训练高斯也因为保留过多源信息而不好改。**

很多前作只解决其中一半：

- 有的重视多视角一致性，但没解决优化效率。
- 有的会做局部编辑，但定位仍然依赖粗糙 2D mask，且改动强度不够。

### 核心能力定义

- **输入**：源 3DGS 与文本编辑指令。
- **输出**：多视角一致、且局部编辑更强、更明确的 3DGS。
- **主要收益**：既提升 consistency，又提升 editability。
- **特别擅长**：需要明显几何或外观变化的语义局部编辑。

### 真正的挑战来源

- 多视角图像如果各自独立经过 2D diffusion，得到的 supervision 会互相冲突。
- 预训练高斯在语义关键区域已经“长死了”，即使有正确梯度也不容易快速变形。
- 传统局部编辑只靠 source image 上的 2D mask，很容易漏掉真正需要强改的语义区域。

### 边界条件

- 仍然依赖 2D diffusion editor 作为图像编辑主干。
- attention-based trimming 需要 cross-attention 与目标 token 之间有比较明确的语义对应。
- 更偏优化式编辑，不是轻量级前向模型。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

EditSplat 的思路很完整：

**先让 2D 编辑监督本身变得更可靠，再让 3D 表示层更容易被这些监督改动。**

这就是为什么它会同时提出两个模块：

- `MFG`：解决监督质量。
- `AGT`：解决表示更新效率。

### The "Aha!" Moment

最关键的 aha 是：

**高注意区域既是最该改的区域，也是最不该完全保留原始高斯的区域。**

作者把 diffusion cross-attention 反投影到 3D 高斯上，得到每个 Gaussian 的累计注意权重。这个信号同时告诉系统两件事：

- 哪些区域在语义上最相关，应当重点编辑。
- 哪些预训练高斯最可能成为“保留原貌的阻力”，应该被剪掉一部分。

于是 `AGT` 不是简单做 ROI mask，而是把 attention 当成优化难度和编辑语义的重要性指标。

### 为什么这个设计有效？

- `MFG` 通过深度投影和多视角融合，先构造每个目标视角的 fused image，再把它作为 guidance 注入 diffusion。这样编辑器不只看到当前 source image，还看到来自邻近视角的补充信息。
- 这种多条件 guidance 通过 `text + source + fused image` 三者平衡，让 2D 编辑同时保留 source fidelity、满足文本意图并兼顾多视角一致性。
- `AGT` 则在 3D 层面把高注意、但又被旧信息束缚的高斯裁掉一部分，并且只让高注意区域承担主要更新，从而让 densification 与优化更有效率。

### 战略权衡

- **优点**：把“监督一致性”和“表示可编辑性”统一纳入一个框架。
- **代价**：比只做 MFG 或只做局部掩码更复杂，且含有若干需要调的 guidance / trimming 超参数。
- **定位**：这是一个面向高质量 text-driven 3DGS editing 的综合框架，尤其重视强编辑场景。

## Part III / Technical Deep Dive

### Pipeline

```text
source 3DGS + text prompt
-> edit all source images once with 2D diffusion
-> project and blend adjacent edited views using 3DGS depth
-> obtain multi-view fused images
-> use text + source + fused image as joint guidance in diffusion
-> get multi-view-consistent edited images
-> inverse render cross-attention to 3D Gaussians
-> AGT prunes top-attention redundant Gaussians and freezes low-attention ones
-> optimize trimmed 3DGS with LPIPS + L1
-> edited 3DGS
```

### 关键模块

#### 1. Multi-view Fusion Guidance

MFG 先把初步编辑后的各视角通过深度图回投、重投影和按深度排序的 alpha blending 融到目标视角，形成一个融合了邻近视角细节的 guidance image。它还用 `ImageReward` 先过滤掉底部 `15%` 的低质量编辑视图，减少错误传播。

#### 2. Multi-condition classifier-free guidance

作者没有加额外参数层，也没有多次 diffusion pass，而是直接改 classifier-free guidance 的组合方式，把：

- 文本 prompt
- source image
- multi-view fused image

一起作为 guidance。这样 diffusion 在生成目标视图时，会被显式拉向多视角一致的视觉证据。

#### 3. Attention-Guided Trimming

通过逆渲染 multi-view-consistent cross-attention，作者为每个 Gaussian 计算累计 attention weight。随后：

- 低注意区域直接不更新梯度。
- 高注意区域中 top-k% 的冗余预训练高斯会被 pruning。

这个操作的本质是提前清掉“最妨碍编辑收敛的旧高斯”，给新 densification 和重构腾空间。

### 关键实验信号

- 定量上，EditSplat 在两个 CLIP 指标和用户偏好上都显著领先。
- MFG 消融显示，没有 multi-view fused guidance 时，2D 编辑会出现 minimal change、blurriness 和 artifacts，最终 3D 结果也更差。
- AGT 消融显示，在 clown nose / mannequin 这类需要明显局部几何变化的场景中，不做 trimming 会残留原脸部结构，做了 AGT 后几何变化更充分。

### 少量关键数字

- `CLIP dir`: `0.1431`
- `CLIP sim`: `0.2531`
- User Study: `0.4227`
- 单场景编辑时间：约 `6 min`

### 实现约束

- 评测覆盖 `8` 个场景、每个场景 `2` 条 prompts。
- 使用 `InstructPix2Pix` 作为 2D editor，运行在单张 `RTX A6000` 上。
- 3D 优化采用 `LPIPS + L1`，权重比 `1:1`。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/CVPR_2025/2025_EditSplat_Multi_View_Fusion_and_Attention_Guided_Optimization_for_View_Consistent_3D_Scene_Editing_with_3D_Gaussian_Splatting.pdf]]
