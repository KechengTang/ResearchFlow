---
title: "Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting"
venue: SIGGRAPH
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - drag-based-editing
  - score-distillation
  - triplane-scaffold
  - control-points
  - localized-editing
  - two-stage-optimization
  - status/analyzed
core_operator: 用 MTP triplane 编码器与 RSP 解码器先构建可拖拽的几何脚手架，再通过 Drag-SDS 把 2D drag diffusion 的控制信号稳定蒸馏进 3DGS，并用 SLE 保持局部连续编辑。
primary_logic: |
  先根据 3D mask 与控制点对确定编辑区域与拖拽目标，
  再通过 MTP + RSP 在第一阶段只优化位置偏移、建立几何脚手架，
  第二阶段再优化颜色等属性并结合 Drag-SDS 与 SLE 细化局部结果，
  最终得到多视角一致的 drag-based 3DGS 编辑结果。
pdf_ref: paperPDFs/3DGS_Editing/SIGGRAPH_2025/2025_Drag_Your_Gaussian_Effective_Drag_Based_Editing_with_Score_Distillation_for_3D_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-09T18:42
updated: 2026-04-09T18:42
---

# Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://quyans.github.io/Drag-Your-Gaussian/) · [arXiv 2501.18672](https://arxiv.org/abs/2501.18672)
> - **Summary**: 这篇工作把 2D drag editing 的“拖拽控制”真正迁移到了 3DGS 上，重点解决的不是简单纹理改色，而是**如何精确拖动 3D 场景局部几何，同时保持多视角一致与局部自然衔接**。
> - **Key Performance**:
>   - 一次拖拽编辑约需 `10 分钟`，论文明确对比 `GS-Editor` 往往需要 `20 分钟以上`。
>   - 用户研究中，作者报告有 `86.1%` 和 `62%` 的投票分别更偏好其编辑效果与场景质量。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？

现有 3DGS 编辑大多是文本驱动，擅长外观变化，不擅长精细几何控制。  
但真实交互里，用户常常想做的是：

- 把头往左转一点
- 把嘴张开
- 把某个局部拖到指定位置

这类操作靠语言并不精确，也很难稳定描述“改多少、往哪改”。  
Drag Your Gaussian 的目标就是把 2D drag paradigm 变成一个**可在 3DGS 上精确操作几何的交互接口**。

### 核心能力定义

- **输入**：一个已重建的 3DGS 场景，3D mask，以及若干对 handle/target control points
- **输出**：满足拖拽意图、且多视角一致的 3D 编辑结果
- **特别能力**：不仅改纹理，还要能做人脸转向、抬头、开口这类几何拖拽

### 真正的挑战来源

- 目标区域里的高斯往往很稀疏，直接靠 diffusion 很难在目标位置“长出合理几何”
- 只冻结局部外区域会造成边界断裂，不冻结又容易全场景一起变
- 2D drag diffusion 本身是图像模型，要把它稳定蒸馏进 3DGS 并保证跨视角一致并不直接

### 边界条件

- 方法依赖 3D mask 与控制点质量
- 当底层 2D guidance 模型本身失败时，3D 编辑也会受限
- 对特别大幅度、拓扑变化极强的几何编辑仍然不稳

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

这篇论文的设计非常清楚：

- **几何先行**：先解决“目标区域怎么被拖到合理位置”
- **纹理后补**：再细化颜色、细节和外观
- **局部连续**：不追求绝对硬切的局部编辑，而追求编辑区与非编辑区之间自然过渡

因此作者采用了“两阶段拖拽”而不是一步到位：

- 第一阶段先建几何脚手架
- 第二阶段再做外观细化

### The "Aha!" Moment

真正的 aha 是：

**3DGS 的 drag editing 之所以难，不只是因为控制信号来自 2D，而是因为目标位置处经常没有足够的几何承载体。**

所以作者没有直接让高斯硬移动完事，而是先引入一个隐式 triplane 几何脚手架：

- 用 `MTP` 编码位置
- 用 `RSP` 预测位置偏移

先把目标几何形态大致搭出来，再让 Drag-SDS 去填细节。

### 为什么这个设计有效

MTP + RSP 的组合，本质上是在 3DGS 这种稀疏显式表示外面包一层更连续的几何调整器。  
这样既能保留 3DGS 的显式渲染优势，又能借助 triplane 的连续性补齐目标区域的几何稀疏问题。

Drag-SDS 则把 Lightning-Drag 的控制能力迁移进 3DGS，但作者没有直接照搬 SDS，而是构造了：

- latent-space loss
- image-space loss
- LoRA-based source prediction

来让拖拽 supervision 更稳定、更局部。

### 战略权衡

- 优点：交互更精确，几何编辑能力明显强于纯文本编辑
- 代价：系统更复杂，依赖 3D mask、控制点投影、两阶段训练与多个网络模块

## Part III / Technical Deep Dive

### Pipeline

```text
3DGS scene + 3D mask + handle/target points
-> MTP encoder + RSP decoder
-> stage 1: geometric scaffold construction
-> stage 2: texture and attribute refinement
-> Drag-SDS with Lightning-Drag guidance
-> SLE for smooth local transition
-> multi-view consistent drag-based 3D editing
```

### 关键模块

#### 1. MTP Encoder

MTP 通过多分辨率 triplane 编码 3D Gaussian 位置，核心作用是给位置调整提供一个更连续、更适合表达形变的几何空间。  
这一步是在给后续 drag 编辑造“连续几何支架”。

#### 2. RSP Decoder

RSP 不只是给 mask 内高斯预测位移，还专门为 mask 外区域设计了补偿分支与正则项。  
这意味着作者不满足于“只改局部”，而是进一步追求**局部编辑时的边界协调**。

#### 3. Two-stage Dragging

- **Stage 1**：冻结 3DGS 属性，只优化 MTP/RSP，先把几何拖拽结构搭起来
- **Stage 2**：解冻更多属性并重新允许 densify/prune，补外观与细节

这个阶段拆分非常关键，因为它把“几何建架”和“外观润色”从优化上解耦了。

#### 4. Soft Local Edit

作者没有把非编辑区完全锁死，而是对编辑区 KNN 邻域里的非编辑高斯给较小学习率。  
这是一个很实用的 operator：**局部编辑不是 binary freeze，而是 soft transition**。

#### 5. Drag-SDS

Drag-SDS 将 Lightning-Drag 的 drag guidance 注入 3DGS，但又额外加入 source prediction 与 image-space/latent-space 协同，缓解普通 SDS 的模糊、过饱和和局部不协调问题。

### 关键实验信号

- 与 `GS-Editor`、`GS-Ctrl`、`SC-GS`、`2D-Lifting` 相比，作者强调自己在几何拖拽上最稳定
- 论文中的对比很清楚：别的方法要么几乎只改颜色，要么多视角不一致，要么背景会被撕裂
- 模块 ablation 也非常有信息量：MTP、RSP、two-stage、SLE、Drag-SDS 每个都不是装饰件，而是都在补一个明确短板

### 少量关键数字

- 单次 dragging 约 `10 分钟`
- 对比基线 `GS-Editor` 往往超过 `20 分钟`
- 用户偏好投票中，作者报告 `86.1%` 和 `62%` 的结果偏好优势

### 对你可迁移的 operator

- **triplane geometric scaffold for sparse 3DGS editing**
- **two-stage geometry-first editing**
- **soft local editing instead of hard freeze**
- **drag-conditioned score distillation for 3D control**

如果你以后想做“更像交互工具”的 3DGS/4DGS 编辑，这篇论文很值得记住，因为它把**控制接口设计**提到了和生成质量同等重要的位置。

### 实现约束

- 实验基于已重建的 vanilla 3DGS 场景
- 作者在 `A100-40G` 上测试
- 3D 控制点会被投影到 2D 送入 drag diffusion guidance

## Local Reading / PDF 参考

![[paperPDFs/3DGS_Editing/SIGGRAPH_2025/2025_Drag_Your_Gaussian_Effective_Drag_Based_Editing_with_Score_Distillation_for_3D_Gaussian_Splatting.pdf]]
